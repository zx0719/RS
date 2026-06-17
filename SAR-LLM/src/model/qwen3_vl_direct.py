from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image


@dataclass
class DirectQwen3VLSample:
    dataset: str
    sample_id: str
    image_path: str
    prompt: str
    reference: str
    turns: Optional[List[Tuple[str, str]]] = None


class DirectQwen3VLVLLM:
    """Direct Qwen3-VL multimodal inference backend based on vLLM.

    This class is intentionally independent from SAR projector training code:
    it sends raw images to Qwen3-VL and is used for base-model evaluation.
    GPU selection must be controlled by launch scripts via CUDA_VISIBLE_DEVICES.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = 6144,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = 262144,
        seed: int = 0,
    ) -> None:
        try:
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except Exception as e:
            raise RuntimeError(
                "Direct Qwen3-VL evaluation requires transformers and vllm in the active environment."
            ) from e

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.sampling_params_cls = SamplingParams

        mm_processor_kwargs: Dict[str, int] = {}
        if min_pixels is not None:
            mm_processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            mm_processor_kwargs["max_pixels"] = int(max_pixels)

        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": True,
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "dtype": dtype,
            "enforce_eager": bool(enforce_eager),
            "limit_mm_per_prompt": {"image": 1},
            "seed": int(seed),
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if mm_processor_kwargs:
            llm_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        self.llm = LLM(**llm_kwargs)

    def _chat_prompt(self, sample: DirectQwen3VLSample, stage: str) -> str:
        if stage == "stage1":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample.image_path},
                        {"type": "text", "text": sample.prompt},
                    ],
                }
            ]
        else:
            if not sample.turns:
                raise ValueError("stage2 sample has no turns")
            messages: List[Dict[str, Any]] = []
            last_idx = len(sample.turns) - 1
            for i, (user_text, assistant_text) in enumerate(sample.turns):
                content: List[Dict[str, str]] = []
                if i == 0:
                    content.append({"type": "image", "image": sample.image_path})
                content.append({"type": "text", "text": user_text})
                messages.append({"role": "user", "content": content})
                if i < last_idx:
                    messages.append({"role": "assistant", "content": assistant_text})

        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        with Image.open(path) as img:
            out = img.convert("RGB")
            out.load()
            return out

    def generate_batch(
        self,
        samples: Sequence[DirectQwen3VLSample],
        stage: str,
        max_new_tokens: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        prompts: List[Dict[str, Any]] = []
        metas: List[Tuple[DirectQwen3VLSample, str]] = []
        skipped: List[Dict[str, Any]] = []

        for sample in samples:
            try:
                chat_prompt = self._chat_prompt(sample, stage)
                image = self._load_image(sample.image_path)
            except Exception as e:
                skipped.append(
                    {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "image_path": sample.image_path,
                        "reason": repr(e),
                    }
                )
                continue

            prompts.append(
                {
                    "prompt": chat_prompt,
                    "multi_modal_data": {"image": image},
                }
            )
            metas.append((sample, chat_prompt))

        if not prompts:
            return [], skipped, 0.0

        sampling_params = self.sampling_params_cls(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(max_new_tokens),
        )
        t0 = time.time()
        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        generate_sec = time.time() - t0

        records: List[Dict[str, Any]] = []
        for (sample, chat_prompt), req_out in zip(metas, outputs):
            if not req_out.outputs:
                text = ""
                token_len = 0
            else:
                best = req_out.outputs[0]
                text = str(best.text).strip()
                token_ids = getattr(best, "token_ids", None)
                token_len = len(token_ids) if token_ids is not None else 0
            records.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "prompt": sample.prompt,
                    "chat_prompt": chat_prompt,
                    "reference": sample.reference,
                    "hypothesis": text,
                    "output_tokens": token_len,
                }
            )
        return records, skipped, generate_sec
