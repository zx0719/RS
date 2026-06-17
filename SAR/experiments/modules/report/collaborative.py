"""
collaborative.py — Coordinated small/large-model report generation.

This module is stage-2 infrastructure for large-scene report generation:
  - small-model first, large-model escalation when needed
  - digest-based cache
  - optional template fallback path for offline development
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .generator import LocalModelGenerator, ReportGenerator
from .large_scene import attach_large_scene_metadata
from .prompt_templates import _fmt_acquisition_time, build_refine_prompt_payload
from .table_builder import TableBuilder

logger = logging.getLogger(__name__)

_ROLE_TO_SOURCE = {
    "small_llm": "small_llm_v1",
    "large_llm": "large_llm_v1",
}
_ROLE_TO_CACHE_SOURCE = {
    "small_llm": "small_llm_cache_v1",
    "large_llm": "large_llm_cache_v1",
}
_REFINE_ROLE = "large_refine"
_REFINE_SOURCE = "large_llm_refine_v1"
_REFINE_CACHE_SOURCE = "large_llm_refine_cache_v1"
_ROUTE_LARGE_REFINE = "large_refine"


class CollaborativeReportGenerator:
    """Coordinate small/large model generation with cache and fallback."""

    def __init__(
        self,
        small_model_name: str = "Qwen2.5-7B-Instruct",
        small_base_url: str | None = None,
        small_model_path: str | None = None,
        large_model_name: str | None = None,
        large_base_url: str | None = None,
        large_model_path: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        cache_dir: str | None = None,
        local_device: str = "auto",
        local_max_new_tokens: int = 2048,
        require_gpu: bool = False,
        allow_template_fallback: bool = True,
    ) -> None:
        self.allow_template_fallback = allow_template_fallback
        if small_model_path:
            self._small_backend = LocalModelGenerator(
                model_path=small_model_path,
                device=local_device,
                max_new_tokens=local_max_new_tokens,
                require_gpu=require_gpu,
                allow_template_fallback=allow_template_fallback,
            )
        else:
            self._small_backend = ReportGenerator(
                model_name=small_model_name,
                base_url=small_base_url,
                api_key=api_key,
                timeout=timeout,
                allow_template_fallback=allow_template_fallback,
            )
        self._large_backend = None
        if large_model_path:
            self._large_backend = LocalModelGenerator(
                model_path=large_model_path,
                device=local_device,
                max_new_tokens=local_max_new_tokens,
                require_gpu=require_gpu,
                allow_template_fallback=allow_template_fallback,
            )
        elif large_model_name or large_base_url:
            self._large_backend = ReportGenerator(
                model_name=large_model_name or small_model_name,
                base_url=large_base_url or small_base_url,
                api_key=api_key,
                timeout=timeout,
                allow_template_fallback=allow_template_fallback,
            )

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._table_builder = TableBuilder()

    @property
    def small_backend(self) -> ReportGenerator:
        return self._small_backend

    @property
    def large_backend(self) -> ReportGenerator | None:
        return self._large_backend

    def generate(self, evidence_package: dict) -> dict:
        pkg = deepcopy(evidence_package)
        if pkg.get("status") != "READY_FOR_NLG":
            raise ValueError(
                f"Evidence Package 状态应为 READY_FOR_NLG，实际为 {pkg.get('status')!r}"
            )

        existing_report_context = deepcopy(pkg.get("report_context", {}))
        pkg = attach_large_scene_metadata(pkg)
        report_context = pkg.setdefault("report_context", {})
        if existing_report_context.get("generation_route"):
            report_context["generation_route"] = existing_report_context["generation_route"]
        if existing_report_context.get("large_scene_digest"):
            report_context["large_scene_digest"] = existing_report_context["large_scene_digest"]
        route = report_context.get("generation_route", "small_llm")
        digest = report_context.get("large_scene_digest", {})
        digest_id = digest.get("digest_id", "no-digest")
        statistics = pkg.get("statistics", {})
        objects = pkg.get("objects", [])

        body: str | None = None
        source_tag = "template_v1"
        selected_backend: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        draft_body: str | None = None
        for role in self._route_plan(route):
            if role == "template":
                if not self.allow_template_fallback:
                    attempts.append({"role": role, "status": "disabled"})
                    continue
                draft_body = draft_body or self._small_backend._template_fallback(pkg)
                body = draft_body
                source_tag = "template_v1"
                attempts.append({"role": role, "status": "used"})
                break

            backend = self._backend_for_role(role)
            if not self._backend_available(backend):
                attempts.append({"role": role, "status": "unavailable"})
                continue
            gpu_ok, gpu_reason = self._backend_gpu_requirement_ok(backend)
            if not gpu_ok:
                attempts.append({"role": role, "status": "unavailable", "reason": gpu_reason})
                continue

            cached_body, cache_reason, cached_backend = self._load_cache(digest_id, role, backend)
            if cached_body is not None:
                ok, reason = backend._post_validate(cached_body, statistics)
                if ok:
                    draft_body = cached_body
                    if self._should_refine(route, role):
                        refined = self._maybe_refine_with_large_model(
                            pkg,
                            draft_body,
                            digest_id,
                            statistics,
                            attempts,
                        )
                        if refined is not None:
                            body, source_tag, selected_backend = refined
                        else:
                            body = draft_body
                            source_tag = _ROLE_TO_CACHE_SOURCE[role]
                            selected_backend = self._selected_backend_metadata(
                                role,
                                source_tag,
                                "cache_hit",
                                backend,
                                cached_backend,
                            )
                        attempts.append({"role": role, "status": "cache_hit"})
                    else:
                        body = cached_body
                        source_tag = _ROLE_TO_CACHE_SOURCE[role]
                        selected_backend = self._selected_backend_metadata(
                            role,
                            source_tag,
                            "cache_hit",
                            backend,
                            cached_backend,
                        )
                        attempts.append({"role": role, "status": "cache_hit"})
                    break
                attempts.append({"role": role, "status": "cache_invalid", "reason": reason})
            elif cache_reason:
                attempts.append({"role": role, "status": "cache_invalid", "reason": cache_reason})

            try:
                candidate = backend._call_llm(pkg)
            except Exception as exc:
                attempts.append({"role": role, "status": "error", "reason": str(exc)})
                continue

            ok, reason = backend._post_validate(candidate, statistics)
            if ok:
                draft_body = candidate
                self._save_cache(digest_id, role, backend, candidate)
                if self._should_refine(route, role):
                    refined = self._maybe_refine_with_large_model(
                        pkg,
                        draft_body,
                        digest_id,
                        statistics,
                        attempts,
                    )
                    if refined is not None:
                        body, source_tag, selected_backend = refined
                    else:
                        body = draft_body
                        source_tag = _ROLE_TO_SOURCE[role]
                        selected_backend = self._selected_backend_metadata(
                            role,
                            source_tag,
                            "used",
                            backend,
                        )
                    attempts.append({"role": role, "status": "used"})
                else:
                    body = candidate
                    source_tag = _ROLE_TO_SOURCE[role]
                    selected_backend = self._selected_backend_metadata(
                        role,
                        source_tag,
                        "used",
                        backend,
                    )
                    attempts.append({"role": role, "status": "used"})
                break

            attempts.append({"role": role, "status": "invalid", "reason": reason})

        if body is None:
            if not self.allow_template_fallback:
                report_context["generation_trace"] = {
                    "route": route,
                    "selected_source": None,
                    "selected_backend": None,
                    "digest_id": digest_id,
                    "available_backends": self.describe_backends(),
                    "attempts": attempts,
                }
                raise RuntimeError(
                    "No collaborative LLM backend produced a valid report and "
                    "template fallback is disabled."
                )
            body = self._small_backend._template_fallback(pkg)
            source_tag = "template_v1"
            attempts.append({"role": "template", "status": "fallback"})

        component_table = self._table_builder.build_component_table(objects)
        equipment_table = self._table_builder.build_equipment_table(objects)

        inp = pkg.get("input", {})
        metadata = inp.get("metadata", {})
        mission = inp.get("mission", {})
        acq_time_raw = metadata.get("acquisition_time", "")
        report_date_cn = _fmt_acquisition_time(acq_time_raw) if acq_time_raw else ""

        report = pkg.setdefault("report", {})
        report["title"] = report.get("title") or "航天通报"
        report["subtitle"] = report.get("subtitle") or f"{mission.get('region_name', '未知区域')}SAR目标监测通报"
        if report_date_cn:
            report["report_date"] = report.get("report_date") or report_date_cn
        report["body"] = body
        report["body_sections"] = [
            {
                "section_name": "summary",
                "content": body,
                "source": source_tag,
            }
        ]
        report["tables"] = {
            "component_table": component_table,
            "equipment_table": equipment_table,
        }

        report_context["generation_trace"] = {
            "route": route,
            "selected_source": source_tag,
            "selected_backend": selected_backend,
            "digest_id": digest_id,
            "available_backends": self.describe_backends(),
            "attempts": attempts,
        }

        pkg["status"] = "REPORT_DRAFTED"
        pkg["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        return pkg

    @staticmethod
    def _route_plan(route: str) -> list[str]:
        if route == _ROUTE_LARGE_REFINE:
            return ["small_llm", "template"]
        if route == "large_llm":
            return ["large_llm", "small_llm", "template"]
        if route == "template":
            return ["template"]
        return ["small_llm", "large_llm", "template"]

    def _backend_for_role(self, role: str) -> ReportGenerator | None:
        if role == "small_llm":
            return self._small_backend
        if role == "large_llm":
            return self._large_backend
        return None

    @staticmethod
    def _backend_available(backend: ReportGenerator | None) -> bool:
        if backend is None:
            return False
        if isinstance(backend, LocalModelGenerator):
            return True
        return backend.base_url is not None

    @staticmethod
    def _backend_gpu_requirement_ok(backend: ReportGenerator | None) -> tuple[bool, str]:
        if not isinstance(backend, LocalModelGenerator) or not backend.require_gpu:
            return True, ""
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return True, ""
        except Exception:
            pass
        return False, "cuda_not_visible"

    def _cache_path(self, digest_id: str, role: str, model_name: str) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_model = re.sub(r"[^\w.-]+", "_", model_name)
        return self.cache_dir / f"{digest_id}_{role}_{safe_model}.json"

    def _load_cache(
        self,
        digest_id: str,
        role: str,
        backend: ReportGenerator,
    ) -> tuple[str | None, str, dict[str, Any] | None]:
        path = self._cache_path(digest_id, role, backend.model_name)
        if path is None or not path.exists():
            return None, "", None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, "cache_unreadable", None
        ok, reason = self._cache_metadata_ok(payload, backend)
        if not ok:
            return None, reason, None
        body = payload.get("body")
        cached_backend = payload.get("backend")
        if not isinstance(cached_backend, dict):
            cached_backend = None
        return (
            (body, "", cached_backend)
            if isinstance(body, str) and body
            else (None, "cache_body_missing", None)
        )

    def _save_cache(
        self,
        digest_id: str,
        role: str,
        backend: ReportGenerator,
        body: str,
    ) -> None:
        path = self._cache_path(digest_id, role, backend.model_name)
        if path is None:
            return
        payload = {
            "digest_id": digest_id,
            "role": role,
            "model_name": backend.model_name,
            "backend": self._backend_cache_metadata(backend),
            "body": body,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _backend_cache_metadata(self, backend: ReportGenerator) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "mode": "local" if isinstance(backend, LocalModelGenerator) else "api",
            "model_name": backend.model_name,
            "base_url": getattr(backend, "base_url", None),
            "model_path": getattr(backend, "model_path", None),
            "require_gpu": getattr(backend, "require_gpu", False),
            "cuda_available": None,
        }
        if isinstance(backend, LocalModelGenerator):
            try:
                import torch  # type: ignore
                metadata["cuda_available"] = torch.cuda.is_available()
            except Exception:
                metadata["cuda_available"] = False
        metadata["local_gpu_verified"] = (
            metadata.get("mode") == "local"
            and metadata.get("require_gpu") is True
            and metadata.get("cuda_available") is True
        )
        return metadata

    def _selected_backend_metadata(
        self,
        role: str,
        source: str,
        status: str,
        backend: ReportGenerator,
        cached_backend: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(cached_backend) if cached_backend else self._backend_cache_metadata(backend)
        metadata.update(
            {
                "role": role,
                "source": source,
                "selection_status": status,
            }
        )
        return metadata

    def _cache_metadata_ok(
        self,
        payload: dict[str, Any],
        backend: ReportGenerator,
    ) -> tuple[bool, str]:
        cached_backend = payload.get("backend")
        if not isinstance(cached_backend, dict):
            if isinstance(backend, LocalModelGenerator) and backend.require_gpu:
                return False, "cache_missing_backend_metadata"
            return True, ""
        if cached_backend.get("model_name") != backend.model_name:
            return False, "cache_model_mismatch"
        if isinstance(backend, LocalModelGenerator):
            if cached_backend.get("model_path") != backend.model_path:
                return False, "cache_model_path_mismatch"
            if backend.require_gpu and not (
                cached_backend.get("require_gpu") is True
                and cached_backend.get("cuda_available") is True
            ):
                return False, "cache_not_gpu_verified"
        return True, ""

    def describe_backends(self) -> dict[str, Any]:
        small_backend = self._backend_cache_metadata(self._small_backend)
        small_backend["allow_template_fallback"] = self.allow_template_fallback
        large_backend = None
        if self._large_backend is not None:
            large_backend = self._backend_cache_metadata(self._large_backend)
            large_backend["allow_template_fallback"] = self.allow_template_fallback
        return {
            "small_llm": small_backend,
            "large_llm": large_backend,
        }

    def health_status(self) -> dict[str, Any]:
        return {
            "small_llm": self._backend_health(self._small_backend),
            "large_llm": self._backend_health(self._large_backend),
        }

    @staticmethod
    def _backend_health(backend: ReportGenerator | None) -> dict[str, Any] | None:
        if backend is None:
            return None
        if isinstance(backend, LocalModelGenerator):
            model_path = Path(backend.model_path)
            cuda_available = False
            try:
                import torch  # type: ignore
                cuda_available = torch.cuda.is_available()
            except Exception:
                cuda_available = False
            path_exists = model_path.exists()
            healthy = path_exists and (cuda_available or not backend.require_gpu)
            return {
                "model_name": backend.model_name,
                "model_path": backend.model_path,
                "require_gpu": backend.require_gpu,
                "cuda_available": cuda_available,
                "healthy": healthy,
                "reason": (
                    "local_model_missing"
                    if not path_exists
                    else "cuda_not_visible"
                    if backend.require_gpu and not cuda_available
                    else "local_model_exists"
                ),
            }
        status = {
            "model_name": backend.model_name,
            "base_url": backend.base_url,
            "healthy": False,
            "reason": "disabled" if backend.base_url is None else "unknown",
        }
        if backend.base_url is None:
            return status
        try:
            import requests  # type: ignore
            resp = requests.get(
                f"{backend.base_url}/models",
                headers={"Authorization": f"Bearer {backend.api_key}"},
                timeout=min(backend.timeout, 5),
            )
            status["healthy"] = resp.status_code == 200
            status["reason"] = f"http_{resp.status_code}"
        except Exception as exc:
            status["reason"] = str(exc)
        return status

    def _should_refine(self, route: str, role: str) -> bool:
        return (
            route == _ROUTE_LARGE_REFINE
            and role == "small_llm"
            and self._backend_available(self._large_backend)
            and self._backend_gpu_requirement_ok(self._large_backend)[0]
        )

    def _maybe_refine_with_large_model(
        self,
        pkg: dict[str, Any],
        draft_body: str,
        digest_id: str,
        statistics: dict[str, Any],
        attempts: list[dict[str, Any]],
    ) -> tuple[str, str] | None:
        backend = self._large_backend
        if backend is None or not self._backend_available(backend):
            attempts.append({"role": _REFINE_ROLE, "status": "unavailable"})
            return None
        gpu_ok, gpu_reason = self._backend_gpu_requirement_ok(backend)
        if not gpu_ok:
            attempts.append({"role": _REFINE_ROLE, "status": "unavailable", "reason": gpu_reason})
            return None

        cached_body, cache_reason, cached_backend = self._load_cache(digest_id, _REFINE_ROLE, backend)
        if cached_body is not None:
            ok, reason = backend._post_validate(cached_body, statistics)
            if ok:
                attempts.append({"role": _REFINE_ROLE, "status": "cache_hit"})
                return (
                    cached_body,
                    _REFINE_CACHE_SOURCE,
                    self._selected_backend_metadata(
                        _REFINE_ROLE,
                        _REFINE_CACHE_SOURCE,
                        "cache_hit",
                        backend,
                        cached_backend,
                    ),
                )
            attempts.append({"role": _REFINE_ROLE, "status": "cache_invalid", "reason": reason})
        elif cache_reason:
            attempts.append({"role": _REFINE_ROLE, "status": "cache_invalid", "reason": cache_reason})

        prompt_payload = build_refine_prompt_payload(pkg, draft_body)
        try:
            refined = backend.call_llm_with_prompt(
                prompt_payload["system_prompt"],
                prompt_payload["user_prompt"],
            )
        except Exception as exc:
            attempts.append({"role": _REFINE_ROLE, "status": "error", "reason": str(exc)})
            return None

        ok, reason = backend._post_validate(refined, statistics)
        if not ok:
            attempts.append({"role": _REFINE_ROLE, "status": "invalid", "reason": reason})
            return None

        self._save_cache(digest_id, _REFINE_ROLE, backend, refined)
        attempts.append({"role": _REFINE_ROLE, "status": "used"})
        return (
            refined,
            _REFINE_SOURCE,
            self._selected_backend_metadata(
                _REFINE_ROLE,
                _REFINE_SOURCE,
                "used",
                backend,
            ),
        )
