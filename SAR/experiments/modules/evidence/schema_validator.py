"""
schema_validator.py — Evidence Package Schema Validation

Validates a dict against the required fields of Evidence Package schema v1.0.
Raises descriptive errors on any violation.

This validator checks structural completeness only — it does NOT call any LLM
or perform semantic reasoning.
"""

from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    """Raised when an Evidence Package fails schema validation."""


def validate_evidence_package(package: dict[str, Any]) -> None:
    """Validate *package* against the required fields of schema v1.0.

    Parameters
    ----------
    package:
        The Evidence Package dict to validate.

    Raises
    ------
    SchemaValidationError
        If any required field is missing or has an unexpected type.
    """
    errors: list[str] = []

    # ── Top-level required fields ────────────────────────────────────────────
    _check_required(package, [
        "schema_version",
        "package_id",
        "task_type",
        "status",
        "created_at",
        "updated_at",
        "input",
        "objects",
        "statistics",
    ], path="", errors=errors)

    # ── package_id format ────────────────────────────────────────────────────
    pkg_id = package.get("package_id", "")
    if not isinstance(pkg_id, str) or not pkg_id.startswith("sar-"):
        errors.append(
            f"package_id must be a string starting with 'sar-'; got: {pkg_id!r}"
        )

    # ── status enum ─────────────────────────────────────────────────────────
    valid_statuses = {
        "RECEIVED", "PREPROCESSED", "DETECTED", "GEOLOCATED", "FUSED",
        "READY_FOR_NLG", "REPORT_DRAFTED", "DOCX_RENDERED",
        "REVIEW_PENDING", "COMPLETED", "PARTIAL_SUCCESS", "ERROR",
    }
    status = package.get("status")
    if status not in valid_statuses:
        errors.append(
            f"status '{status}' is not a recognised value. "
            f"Expected one of: {sorted(valid_statuses)}"
        )

    # ── input block ──────────────────────────────────────────────────────────
    inp = package.get("input", {})
    if isinstance(inp, dict):
        _validate_input_block(inp, errors)
    else:
        errors.append("input must be a dict")

    # ── objects array ────────────────────────────────────────────────────────
    objs = package.get("objects", [])
    if not isinstance(objs, list):
        errors.append("objects must be a list")
    else:
        for i, obj in enumerate(objs):
            _validate_object(obj, i, errors)

    # ── statistics block ─────────────────────────────────────────────────────
    stats = package.get("statistics", {})
    if isinstance(stats, dict):
        _validate_statistics(stats, len(objs) if isinstance(objs, list) else 0, errors)
    else:
        errors.append("statistics must be a dict")

    # ── Report ───────────────────────────────────────────────────────────────
    if errors:
        bullet_list = "\n  - ".join(errors)
        raise SchemaValidationError(
            f"Evidence Package validation failed with {len(errors)} error(s):\n"
            f"  - {bullet_list}"
        )


# ---------------------------------------------------------------------------
# Sub-validators
# ---------------------------------------------------------------------------

def _validate_input_block(inp: dict[str, Any], errors: list[str]) -> None:
    _check_required(inp, ["input_id", "image", "metadata", "mission"],
                    path="input", errors=errors)

    image = inp.get("image", {})
    if isinstance(image, dict):
        _check_required(image, ["uri", "file_name", "format", "sha256"],
                        path="input.image", errors=errors)
    else:
        errors.append("input.image must be a dict")

    metadata = inp.get("metadata", {})
    if isinstance(metadata, dict):
        _check_required(metadata, ["satellite", "sensor", "acquisition_time"],
                        path="input.metadata", errors=errors)
    else:
        errors.append("input.metadata must be a dict")

    mission = inp.get("mission", {})
    if isinstance(mission, dict):
        _check_required(mission, ["region_name", "region_type", "priority"],
                        path="input.mission", errors=errors)
    else:
        errors.append("input.mission must be a dict")


def _validate_object(obj: dict[str, Any], index: int, errors: list[str]) -> None:
    path = f"objects[{index}]"
    if not isinstance(obj, dict):
        errors.append(f"{path} must be a dict")
        return

    _check_required(obj, ["object_id", "geometry"], path=path, errors=errors)

    geometry = obj.get("geometry", {})
    if isinstance(geometry, dict):
        pixel = geometry.get("pixel")
        if pixel is None:
            errors.append(f"{path}.geometry.pixel is required")
        elif isinstance(pixel, dict):
            _check_required(pixel, ["center_x", "center_y"],
                            path=f"{path}.geometry.pixel", errors=errors)
        else:
            errors.append(f"{path}.geometry.pixel must be a dict")
    else:
        errors.append(f"{path}.geometry must be a dict")


def _validate_statistics(
    stats: dict[str, Any], n_objects: int, errors: list[str]
) -> None:
    _check_required(stats, ["totals", "by_class", "by_super_class"],
                    path="statistics", errors=errors)

    totals = stats.get("totals", {})
    if isinstance(totals, dict):
        _check_required(totals, ["all_objects", "ships", "aircraft"],
                        path="statistics.totals", errors=errors)
        all_objects = totals.get("all_objects")
        if isinstance(all_objects, int) and all_objects != n_objects:
            errors.append(
                f"statistics.totals.all_objects ({all_objects}) "
                f"!= len(objects) ({n_objects})"
            )
    else:
        errors.append("statistics.totals must be a dict")

    by_class = stats.get("by_class", [])
    if not isinstance(by_class, list):
        errors.append("statistics.by_class must be a list")


# ---------------------------------------------------------------------------
# Generic helper
# ---------------------------------------------------------------------------

def _check_required(
    d: dict[str, Any],
    keys: list[str],
    path: str,
    errors: list[str],
) -> None:
    """Append an error for every key in *keys* that is absent from *d*."""
    prefix = f"{path}." if path else ""
    for key in keys:
        if key not in d:
            errors.append(f"Required field '{prefix}{key}' is missing")
