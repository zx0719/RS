"""
collab_config.py — Shared configuration helpers for collaborative LLM usage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RouteThresholds:
    large_refine_class_threshold: int = 2
    large_refine_object_threshold: int = 24
    large_refine_review_threshold: int = 4
    large_model_cluster_threshold: int = 12
    large_model_class_threshold: int = 3
    large_model_object_threshold: int = 120
    large_model_review_threshold: int = 10
    large_model_low_conf_ratio: float = 0.28


@dataclass(frozen=True)
class CollaborativeConfig:
    small_model_name: str
    small_base_url: str | None
    small_model_path: str | None
    large_model_name: str | None
    large_base_url: str | None
    large_model_path: str | None
    vlm_model_name: str | None = None
    vlm_base_url: str | None = None
    cache_dir: str | None = None
    api_key: str | None = None
    timeout: int = 60
    local_device: str = "auto"
    local_max_new_tokens: int = 2048
    require_gpu: bool = False
    allow_template_fallback: bool = True


@dataclass(frozen=True)
class VLMConfig:
    vlm_model_name: str
    vlm_base_url: str | None
    api_key: str | None = None
    timeout: int = 30
    require_vlm: bool = False


def read_env_file(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _env_get(mapping: dict[str, str] | None, key: str, default: str | None = None) -> str | None:
    if mapping is not None and key in mapping and mapping[key] != "":
        return mapping[key]
    return os.getenv(key, default)


def _env_bool(
    mapping: dict[str, str] | None,
    key: str,
    default: bool,
) -> bool:
    value = _env_get(mapping, key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def config_from_env(
    cache_dir: str | None = None,
    environ: dict[str, str] | None = None,
) -> CollaborativeConfig:
    return CollaborativeConfig(
        small_model_name=_env_get(environ, "SAR_SMALL_LLM_MODEL", "Qwen2.5-7B-Instruct") or "Qwen2.5-7B-Instruct",
        small_base_url=_env_get(environ, "SAR_SMALL_LLM_URL"),
        small_model_path=_env_get(environ, "SAR_SMALL_LLM_PATH"),
        large_model_name=_env_get(environ, "SAR_LARGE_LLM_MODEL", "Qwen2.5-72B-Instruct"),
        large_base_url=_env_get(environ, "SAR_LARGE_LLM_URL"),
        large_model_path=_env_get(environ, "SAR_LARGE_LLM_PATH"),
        vlm_model_name=_env_get(environ, "SAR_VLM_MODEL", "qwen3-vl-4b"),
        vlm_base_url=_env_get(environ, "SAR_VLM_URL"),
        cache_dir=cache_dir,
        api_key=_env_get(environ, "SAR_LLM_API_KEY", "EMPTY"),
        timeout=int(_env_get(environ, "SAR_LLM_TIMEOUT", "60") or "60"),
        local_device=_env_get(environ, "SAR_LLM_DEVICE", "auto") or "auto",
        local_max_new_tokens=int(_env_get(environ, "SAR_LLM_MAX_NEW_TOKENS", "2048") or "2048"),
        require_gpu=_env_bool(environ, "SAR_REQUIRE_GPU", False),
        allow_template_fallback=_env_bool(environ, "SAR_ALLOW_TEMPLATE_FALLBACK", True),
    )


def route_thresholds_from_env(environ: dict[str, str] | None = None) -> RouteThresholds:
    return RouteThresholds(
        large_refine_class_threshold=int(_env_get(environ, "SAR_ROUTE_LARGE_REFINE_CLASS_THRESHOLD", "2") or "2"),
        large_refine_object_threshold=int(_env_get(environ, "SAR_ROUTE_LARGE_REFINE_OBJECT_THRESHOLD", "24") or "24"),
        large_refine_review_threshold=int(_env_get(environ, "SAR_ROUTE_LARGE_REFINE_REVIEW_THRESHOLD", "4") or "4"),
        large_model_cluster_threshold=int(_env_get(environ, "SAR_ROUTE_LARGE_MODEL_CLUSTER_THRESHOLD", "12") or "12"),
        large_model_class_threshold=int(_env_get(environ, "SAR_ROUTE_LARGE_MODEL_CLASS_THRESHOLD", "3") or "3"),
        large_model_object_threshold=int(_env_get(environ, "SAR_ROUTE_LARGE_MODEL_OBJECT_THRESHOLD", "120") or "120"),
        large_model_review_threshold=int(_env_get(environ, "SAR_ROUTE_LARGE_MODEL_REVIEW_THRESHOLD", "10") or "10"),
        large_model_low_conf_ratio=float(_env_get(environ, "SAR_ROUTE_LARGE_MODEL_LOW_CONF_RATIO", "0.28") or "0.28"),
    )


def vlm_config_from_env(environ: dict[str, str] | None = None) -> VLMConfig:
    return VLMConfig(
        vlm_model_name=_env_get(environ, "SAR_VLM_MODEL", "qwen3-vl-4b") or "qwen3-vl-4b",
        vlm_base_url=_env_get(environ, "SAR_VLM_URL"),
        api_key=_env_get(environ, "SAR_LLM_API_KEY", "EMPTY"),
        timeout=int(_env_get(environ, "SAR_VLM_TIMEOUT", "30") or "30"),
        require_vlm=_env_bool(environ, "SAR_REQUIRE_VLM", False),
    )


def cache_dir_from_manifest(manifest_path: Path) -> str:
    return str(manifest_path.parent / ".report_cache")


def resolve_collaborative_config(
    *,
    cache_dir: str | None = None,
    env_file: str | Path | None = None,
    small_url: str | None = None,
    small_model: str | None = None,
    small_model_path: str | None = None,
    large_url: str | None = None,
    large_model: str | None = None,
    large_model_path: str | None = None,
    vlm_url: str | None = None,
    vlm_model: str | None = None,
    local_device: str | None = None,
    local_max_new_tokens: int | None = None,
    require_gpu: bool | None = None,
    allow_template_fallback: bool | None = None,
) -> CollaborativeConfig:
    env_values = read_env_file(env_file)
    if small_url:
        env_values["SAR_SMALL_LLM_URL"] = small_url
    if small_model:
        env_values["SAR_SMALL_LLM_MODEL"] = small_model
    if small_model_path:
        env_values["SAR_SMALL_LLM_PATH"] = small_model_path
    if large_url:
        env_values["SAR_LARGE_LLM_URL"] = large_url
    if large_model:
        env_values["SAR_LARGE_LLM_MODEL"] = large_model
    if large_model_path:
        env_values["SAR_LARGE_LLM_PATH"] = large_model_path
    if vlm_url:
        env_values["SAR_VLM_URL"] = vlm_url
    if vlm_model:
        env_values["SAR_VLM_MODEL"] = vlm_model
    if local_device:
        env_values["SAR_LLM_DEVICE"] = local_device
    if local_max_new_tokens is not None:
        env_values["SAR_LLM_MAX_NEW_TOKENS"] = str(local_max_new_tokens)
    if require_gpu is not None:
        env_values["SAR_REQUIRE_GPU"] = "1" if require_gpu else "0"
    if allow_template_fallback is not None:
        env_values["SAR_ALLOW_TEMPLATE_FALLBACK"] = "1" if allow_template_fallback else "0"
    return config_from_env(cache_dir=cache_dir, environ=env_values)


def collaborative_generator_kwargs(config: CollaborativeConfig) -> dict[str, object]:
    return {
        "small_model_name": config.small_model_name,
        "small_base_url": config.small_base_url,
        "small_model_path": config.small_model_path,
        "large_model_name": config.large_model_name,
        "large_base_url": config.large_base_url,
        "large_model_path": config.large_model_path,
        "cache_dir": config.cache_dir,
        "api_key": config.api_key,
        "timeout": config.timeout,
        "local_device": config.local_device,
        "local_max_new_tokens": config.local_max_new_tokens,
        "require_gpu": config.require_gpu,
        "allow_template_fallback": config.allow_template_fallback,
    }


def resolve_vlm_config(
    *,
    env_file: str | Path | None = None,
    vlm_url: str | None = None,
    vlm_model: str | None = None,
    require_vlm: bool | None = None,
) -> VLMConfig:
    env_values = read_env_file(env_file)
    if vlm_url:
        env_values["SAR_VLM_URL"] = vlm_url
    if vlm_model:
        env_values["SAR_VLM_MODEL"] = vlm_model
    if require_vlm is not None:
        env_values["SAR_REQUIRE_VLM"] = "1" if require_vlm else "0"
    return vlm_config_from_env(env_values)
