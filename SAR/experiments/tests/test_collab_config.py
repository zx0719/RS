from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.report.collab_config import (
    cache_dir_from_manifest,
    collaborative_generator_kwargs,
    config_from_env,
    read_env_file,
    resolve_collaborative_config,
    resolve_vlm_config,
    route_thresholds_from_env,
    vlm_config_from_env,
)


def test_config_from_env_reads_shared_collab_settings() -> None:
    env = {
        "SAR_SMALL_LLM_URL": "http://small.local/v1",
        "SAR_SMALL_LLM_MODEL": "small-model",
        "SAR_SMALL_LLM_PATH": "/models/small",
        "SAR_LARGE_LLM_URL": "http://large.local/v1",
        "SAR_LARGE_LLM_MODEL": "large-model",
        "SAR_LARGE_LLM_PATH": "/models/large",
        "SAR_VLM_URL": "http://vlm.local/v1",
        "SAR_VLM_MODEL": "vlm-model",
        "SAR_LLM_API_KEY": "secret",
        "SAR_LLM_TIMEOUT": "45",
        "SAR_LLM_DEVICE": "cpu",
        "SAR_LLM_MAX_NEW_TOKENS": "256",
        "SAR_REQUIRE_GPU": "1",
        "SAR_ALLOW_TEMPLATE_FALLBACK": "0",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = config_from_env(cache_dir="/tmp/cache")
    assert cfg.small_base_url == "http://small.local/v1"
    assert cfg.small_model_name == "small-model"
    assert cfg.small_model_path == "/models/small"
    assert cfg.large_base_url == "http://large.local/v1"
    assert cfg.large_model_name == "large-model"
    assert cfg.large_model_path == "/models/large"
    assert cfg.vlm_base_url == "http://vlm.local/v1"
    assert cfg.vlm_model_name == "vlm-model"
    assert cfg.cache_dir == "/tmp/cache"
    assert cfg.api_key == "secret"
    assert cfg.timeout == 45
    assert cfg.local_device == "cpu"
    assert cfg.local_max_new_tokens == 256
    assert cfg.require_gpu is True
    assert cfg.allow_template_fallback is False


def test_cache_dir_from_manifest_uses_manifest_parent() -> None:
    manifest = Path("/tmp/demo/original_format_docx_manifest.json")
    assert cache_dir_from_manifest(manifest) == "/tmp/demo/.report_cache"


def test_read_env_file_and_route_thresholds_from_mapping(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SAR_ROUTE_LARGE_REFINE_OBJECT_THRESHOLD=10\nSAR_ROUTE_LARGE_MODEL_LOW_CONF_RATIO=0.2\n",
        encoding="utf-8",
    )
    mapping = read_env_file(env_file)
    thresholds = route_thresholds_from_env(mapping)
    assert thresholds.large_refine_object_threshold == 10
    assert thresholds.large_model_low_conf_ratio == 0.2


def test_resolve_collaborative_config_merges_env_file_and_overrides(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SAR_SMALL_LLM_URL=http://small.from.file/v1\nSAR_LARGE_LLM_MODEL=large-from-file\n",
        encoding="utf-8",
    )
    cfg = resolve_collaborative_config(
        cache_dir="/tmp/cache",
        env_file=env_file,
        small_model="small-override",
        large_url="http://large.override/v1",
        vlm_url="http://vlm.override/v1",
    )
    assert cfg.small_base_url == "http://small.from.file/v1"
    assert cfg.small_model_name == "small-override"
    assert cfg.large_model_name == "large-from-file"
    assert cfg.large_base_url == "http://large.override/v1"
    assert cfg.vlm_base_url == "http://vlm.override/v1"


def test_collaborative_generator_kwargs_drop_vlm_fields() -> None:
    cfg = resolve_collaborative_config(
        cache_dir="/tmp/cache",
        small_url="http://small.local/v1",
        small_model_path="/models/small",
        large_url="http://large.local/v1",
        large_model_path="/models/large",
        vlm_url="http://vlm.local/v1",
        local_max_new_tokens=128,
        require_gpu=True,
        allow_template_fallback=False,
    )
    kwargs = collaborative_generator_kwargs(cfg)
    assert "vlm_base_url" not in kwargs
    assert "vlm_model_name" not in kwargs
    assert kwargs["small_base_url"] == "http://small.local/v1"
    assert kwargs["small_model_path"] == "/models/small"
    assert kwargs["large_model_path"] == "/models/large"
    assert kwargs["local_max_new_tokens"] == 128
    assert kwargs["require_gpu"] is True
    assert kwargs["allow_template_fallback"] is False


def test_vlm_config_can_require_vlm(tmp_path: Path) -> None:
    env = {
        "SAR_VLM_URL": "http://vlm.local/v1",
        "SAR_VLM_MODEL": "vlm-model",
        "SAR_REQUIRE_VLM": "1",
    }
    cfg = vlm_config_from_env(env)
    assert cfg.vlm_base_url == "http://vlm.local/v1"
    assert cfg.vlm_model_name == "vlm-model"
    assert cfg.require_vlm is True

    env_file = tmp_path / ".env"
    env_file.write_text("SAR_REQUIRE_VLM=0\n", encoding="utf-8")
    resolved = resolve_vlm_config(env_file=env_file, require_vlm=True)
    assert resolved.require_vlm is True


def test_empty_env_file_values_do_not_hide_process_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SAR_VLM_URL=\n"
        "SAR_SMALL_LLM_PATH=\n"
        "SAR_ALLOW_TEMPLATE_FALLBACK=0\n",
        encoding="utf-8",
    )
    env = {
        "SAR_VLM_URL": "http://vlm.from.process/v1",
        "SAR_SMALL_LLM_PATH": "/models/from-process",
    }

    with patch.dict(os.environ, env, clear=False):
        vlm = resolve_vlm_config(env_file=env_file)
        collab = resolve_collaborative_config(env_file=env_file)

    assert vlm.vlm_base_url == "http://vlm.from.process/v1"
    assert collab.small_model_path == "/models/from-process"
    assert collab.allow_template_fallback is False
