from __future__ import annotations

import gc
import hashlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

import torch


def is_vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


def resolve_generation_backend(requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested not in {"auto", "hf", "vllm"}:
        raise ValueError(f"未知生成后端: {requested!r}，可选: auto / hf / vllm")

    if requested == "auto":
        return "vllm" if is_vllm_available() else "hf"

    if requested == "vllm" and not is_vllm_available():
        raise ImportError(
            "未检测到 vllm，但你要求使用 --gen_backend vllm。\n"
            "请先安装 vllm，或改用 --gen_backend hf。"
        )

    return requested


def release_cuda_resources(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class PromptEmbeddingBuilder:
    """Build prompt embeddings for vLLM prompt-embeds inference."""

    def __init__(
        self,
        qwen_path: str,
        image_token: str = "<sar>",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
    ) -> None:
        try:
            from transformers import AutoTokenizer
            try:
                from transformers import Qwen3VLForConditionalGeneration
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration as Qwen3VLForConditionalGeneration  # noqa: N812
        except Exception as e:
            raise RuntimeError(
                "需要 transformers >= 4.49（含 Qwen2.5-VL）或 >= 4.52（含 Qwen3-VL）。\n"
                "建议：pip install -U transformers accelerate safetensors"
            ) from e

        self.image_token = image_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_path,
            trust_remote_code=trust_remote_code,
        )
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.image_token]}
            )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map="cpu",
        )
        vl_model.resize_token_embeddings(len(self.tokenizer))

        if hasattr(vl_model, "language_model"):
            llm_causal = vl_model.language_model
        elif hasattr(vl_model, "model") and hasattr(vl_model.model, "language_model"):
            llm_causal = vl_model.model.language_model
        else:
            raise AttributeError("无法在 Qwen3-VL 中定位 language_model。")

        if hasattr(llm_causal, "get_input_embeddings"):
            embedding_layer = llm_causal.get_input_embeddings()
        elif hasattr(llm_causal, "model") and hasattr(llm_causal.model, "get_input_embeddings"):
            embedding_layer = llm_causal.model.get_input_embeddings()
        else:
            raise AttributeError("无法在 language_model 中定位 input_embeddings。")

        embedding_weight = embedding_layer.weight.detach().cpu().clone()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embedding_weight,
            freeze=True,
        ).cpu()
        self.embedding_dtype = self.embedding.weight.dtype
        self.hidden_size = int(self.embedding.weight.shape[1])
        self.sar_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.emb_scale = float(self.embedding.weight.abs().mean().item())

        release_cuda_resources(vl_model, llm_causal, embedding_layer, embedding_weight)

    @torch.inference_mode()
    def build_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sar_feats: torch.Tensor,
        projector: torch.nn.Module,
        projector_device: torch.device | str,
    ) -> List[torch.Tensor]:
        prompt_ids_cpu = prompt_ids.detach().to("cpu")
        attention_mask_cpu = attention_mask.detach().to("cpu")
        text_embeds = self.embedding(prompt_ids_cpu)

        projector_device = torch.device(projector_device)
        amp_device = "cuda" if projector_device.type == "cuda" else "cpu"
        with torch.amp.autocast(amp_device, enabled=False):
            sar_token_embeds = projector(sar_feats.to(projector_device).float())

        cur_scale = sar_token_embeds.abs().mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        sar_token_embeds = sar_token_embeds / (cur_scale + 1e-6) * self.emb_scale
        sar_token_embeds = torch.clamp(
            sar_token_embeds,
            min=-3.0 * self.emb_scale,
            max=3.0 * self.emb_scale,
        )
        sar_token_embeds = sar_token_embeds.to(dtype=self.embedding_dtype, device="cpu")

        if prompt_ids_cpu.dim() != 2:
            raise ValueError(f"prompt_ids shape 非法: {tuple(prompt_ids_cpu.shape)}")

        if sar_token_embeds.dim() != 3:
            raise ValueError(f"sar_token_embeds shape 非法: {tuple(sar_token_embeds.shape)}")

        mask = (prompt_ids_cpu == self.sar_token_id)
        counts = mask.sum(dim=1)
        need = sar_token_embeds.shape[1]
        bad = (counts != need)
        if bad.any():
            bad_idx = torch.nonzero(bad, as_tuple=False).view(-1).tolist()
            detail = ", ".join(
                f"sample{i}: count={int(counts[i].item())}, need={need}"
                for i in bad_idx
            )
            raise ValueError(f"<sar> token 数量与视觉 token 数量不匹配: {detail}")

        mask_3d = mask.unsqueeze(-1).expand_as(text_embeds)
        merged = text_embeds.masked_scatter(mask_3d, sar_token_embeds.reshape(-1))

        prompt_lengths = attention_mask_cpu.sum(dim=1).tolist()
        return [
            merged[i, :int(prompt_len)].contiguous()
            for i, prompt_len in enumerate(prompt_lengths)
        ]


class VLLMTextGenerator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        enforce_eager: bool = False,
        enable_lora: bool = False,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            raise RuntimeError(
                "导入 vllm 失败，请检查当前评测环境是否已安装 vllm。"
            ) from e

        llm_kwargs = {
            "model": model_path,
            "enable_prompt_embeds": True,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
            "enable_lora": enable_lora,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self._sampling_params_cls = SamplingParams
        self._llm = LLM(**llm_kwargs)

    def generate(
        self,
        prompt_embeds: List[torch.Tensor],
        max_new_tokens: int,
        lora_request=None,
    ) -> Dict[str, List]:
        sampling_params = self._sampling_params_cls(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
        )
        prompts = [{"prompt_embeds": emb} for emb in prompt_embeds]
        outputs = self._llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )

        texts: List[str] = []
        token_lengths: List[int] = []
        for req_out in outputs:
            if not req_out.outputs:
                texts.append("")
                token_lengths.append(0)
                continue
            best = req_out.outputs[0]
            texts.append(str(best.text).strip())
            token_ids = getattr(best, "token_ids", None)
            token_lengths.append(len(token_ids) if token_ids is not None else 0)

        return {
            "texts": texts,
            "token_lengths": token_lengths,
        }


def build_lora_request(adapter_path: str):
    try:
        from vllm.lora.request import LoRARequest
    except Exception as e:
        raise RuntimeError(
            "导入 vLLM LoRARequest 失败，请确认 vllm 版本支持 LoRA。"
        ) from e

    adapter_dir = Path(adapter_path).resolve()
    request_id = int(hashlib.md5(str(adapter_dir).encode("utf-8")).hexdigest()[:8], 16)
    return LoRARequest(
        str(adapter_dir.name),
        request_id,
        str(adapter_dir),
    )
