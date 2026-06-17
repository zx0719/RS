"""
vlm_describer.py — VLM 图像整体描述模块

调用 Qwen3-VL-4B-Instruct（通过 vLLM OpenAI-compatible API）对 SAR 图像
生成自然语言整体描述，作为情报通报的辅助信息。

设计原则：
- VLM 只负责图像整体描述（场景类型、目标密度、分布特征）
- 不允许 VLM 输出具体数量/类别（这些由 YOLOv8 检测器提供）
- 失败时静默 fallback，不阻塞主流程

用法：
    from modules.report.vlm_describer import VLMDescriber

    describer = VLMDescriber(base_url="http://localhost:8101/v1")
    desc = describer.describe(image_path="/path/to/image.jpg")
    # desc: "图像显示一处港口区域，水域内可见多艘船只停泊，目标分布较密集。"
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# VLM 系统提示：严格限制输出范围，防止幻觉
_VLM_SYSTEM_PROMPT = """你是一名SAR卫星图像分析专家。
你的任务是对输入的SAR图像给出简短的整体场景描述（1~2句话，不超过80字）。

严格限制：
1. 只描述场景类型（港口/机场/海域/陆地等）和目标分布特征（密集/稀疏/集中/分散）
2. 不得给出具体目标数量（数量由专用检测器提供，你无法准确计数）
3. 不得给出具体目标类别（类别由专用检测器提供）
4. 不得推断敌我属性或作战意图
5. 如图像质量差或无法判断，直接说"图像质量有限，场景特征不明显"
"""

_VLM_USER_PROMPT = "请描述这张SAR图像的整体场景特征。"


class VLMDescriber:
    """调用 Qwen3-VL vLLM 服务对 SAR 图像生成整体描述。

    Parameters
    ----------
    base_url:
        vLLM OpenAI-compatible API 地址，例如 "http://localhost:8101/v1"。
        为 None 时直接返回空字符串（离线/测试模式）。
    model_name:
        服务端模型名称，默认 "qwen3-vl-4b"（与启动脚本一致）。
    api_key:
        鉴权密钥，vLLM 默认无鉴权传 "EMPTY"。
    timeout:
        HTTP 请求超时秒数，默认 30。
    max_tokens:
        最大生成 token 数，默认 128（描述不需要太长）。
    """

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str = "qwen3-vl-4b",
        api_key: str = "EMPTY",
        timeout: int = 30,
        max_tokens: int = 128,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def describe(self, image_path: str | Path) -> str:
        """对单张图像生成整体场景描述。

        Parameters
        ----------
        image_path:
            本地图像路径（JPEG/PNG/BMP/GeoTIFF）。

        Returns
        -------
        str
            1~2 句场景描述，失败时返回空字符串。
        """
        if self.base_url is None:
            return ""

        try:
            image_b64 = _encode_image_b64(image_path)
            return self._call_vlm(image_b64)
        except Exception as exc:
            logger.warning("VLM 描述失败（%s），跳过: %s", Path(image_path).name, exc)
            return ""

    def trace_metadata(
        self,
        *,
        image_path: str | Path,
        status: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Return auditable metadata for a VLM scene-description attempt."""
        return {
            "source": "vlm_v1",
            "mode": "api",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "image_path": str(image_path),
            "status": status,
            "description_present": bool(description),
        }

    def describe_batch(
        self, image_paths: list[str | Path]
    ) -> dict[str, str]:
        """批量描述多张图像。

        Returns
        -------
        dict: image_path(str) → description(str)
        """
        return {str(p): self.describe(p) for p in image_paths}

    # ------------------------------------------------------------------
    # 内部调用
    # ------------------------------------------------------------------

    def _call_vlm(self, image_b64: str) -> str:
        """调用 vLLM OpenAI-compatible Vision API。"""
        try:
            import openai  # type: ignore
            client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _VLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                            {"type": "text", "text": _VLM_USER_PROMPT},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            return self._call_vlm_requests(image_b64)

    def _call_vlm_requests(self, image_b64: str) -> str:
        """纯 requests 实现（openai SDK 不可用时的 fallback）。"""
        import requests  # type: ignore

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": _VLM_USER_PROMPT},
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def health_check(self) -> bool:
        """检查 vLLM 服务是否可达。"""
        if self.base_url is None:
            return False
        try:
            import requests  # type: ignore
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _encode_image_b64(image_path: str | Path) -> str:
    """将图像编码为 base64 字符串（JPEG，最大边长 1024px 缩放）。

    大图先缩放再编码，避免 base64 payload 过大导致超时。
    """
    from PIL import Image
    import io

    path = Path(image_path)
    with Image.open(str(path)) as img:
        # 转 RGB（GeoTIFF 可能是单波段或 RGBA）
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        # 限制最大边长 1024px（VLM 输入分辨率足够）
        max_side = 1024
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
