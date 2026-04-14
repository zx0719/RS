"""
consistency.py — Evidence Package consistency checker.

Validates internal self-consistency of an Evidence Package:
  - Count totals match actual valid objects
  - Per-class counts match actual objects
  - All VALID objects carry geo coordinates
  - Required top-level fields are present
"""

from __future__ import annotations

from typing import Any


class ConsistencyChecker:
    """Check numerical and structural consistency of an Evidence Package."""

    REQUIRED_FIELDS: tuple[str, ...] = ("package_id", "objects", "statistics", "input")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, evidence_package: dict[str, Any]) -> dict[str, bool]:
        """Run all consistency checks.

        Parameters
        ----------
        evidence_package:
            A fully or partially populated Evidence Package dict.

        Returns
        -------
        dict
            The ``quality.consistency_checks`` block with four boolean fields.
            All failures are also raised as ``ValueError`` so callers can
            surface detailed messages.
        """
        results: dict[str, bool] = {
            "count_consistent": False,
            "class_consistent": False,
            "coordinate_present": False,
            "required_fields_present": False,
        }

        # 1. Required fields
        results["required_fields_present"] = self._check_required_fields(evidence_package)

        # 2. Gather valid objects (continue even if required fields missing)
        objects: list[dict] = evidence_package.get("objects") or []
        valid_objects = [o for o in objects if self._is_valid(o)]

        statistics: dict = evidence_package.get("statistics") or {}

        # 3. Total count consistency
        results["count_consistent"] = self._check_count(statistics, valid_objects)

        # 4. Per-class count consistency
        results["class_consistent"] = self._check_class_counts(statistics, valid_objects)

        # 5. Geo coordinate presence for VALID objects
        results["coordinate_present"] = self._check_coordinates(valid_objects)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid(obj: dict[str, Any]) -> bool:
        """Return True when the object is considered a VALID detection.

        An object is VALID if its ``status`` field equals ``"VALID"`` **or**
        if the field is absent (legacy packages that predate the status field).
        """
        status = obj.get("status")
        return status is None or status in ("VALID", "PARTIAL_SUCCESS")

    def _check_required_fields(self, pkg: dict[str, Any]) -> bool:
        missing = [f for f in self.REQUIRED_FIELDS if f not in pkg]
        if missing:
            raise ValueError(
                f"required_fields_present FAILED — missing top-level fields: {missing}"
            )
        return True

    @staticmethod
    def _check_count(statistics: dict[str, Any], valid_objects: list[dict]) -> bool:
        totals: dict = statistics.get("totals") or {}
        reported_count: int | None = totals.get("all_objects")
        actual_count: int = len(valid_objects)

        if reported_count is None:
            raise ValueError(
                "count_consistent FAILED — statistics.totals.all_objects is missing"
            )
        if reported_count != actual_count:
            raise ValueError(
                f"count_consistent FAILED — statistics.totals.all_objects={reported_count} "
                f"but actual VALID object count={actual_count}"
            )
        return True

    @staticmethod
    def _check_class_counts(
        statistics: dict[str, Any], valid_objects: list[dict]
    ) -> bool:
        # schema v1.0: by_class is a list-of-dicts [{code, name_cn, count}, ...]
        # legacy: by_class is a plain dict {code: count}
        raw_by_class = statistics.get("by_class") or {}
        if isinstance(raw_by_class, list):
            by_class: dict[str, int] = {
                item["code"]: item["count"]
                for item in raw_by_class
                if "code" in item and "count" in item
            }
        else:
            by_class = dict(raw_by_class)

        # Build ground-truth counts from objects
        actual: dict[str, int] = {}
        for obj in valid_objects:
            cls = (obj.get("class") or {}).get("code") or obj.get("class_code") or "unknown"
            actual[cls] = actual.get(cls, 0) + 1

        errors: list[str] = []

        # Check every class reported in statistics
        for cls_code, reported in by_class.items():
            actual_n = actual.get(cls_code, 0)
            if reported != actual_n:
                errors.append(
                    f"class '{cls_code}': reported={reported}, actual={actual_n}"
                )

        # Check for classes present in objects but absent from statistics
        for cls_code, actual_n in actual.items():
            if cls_code not in by_class:
                errors.append(
                    f"class '{cls_code}': present in objects (count={actual_n}) "
                    f"but missing from statistics.by_class"
                )

        if errors:
            raise ValueError("class_consistent FAILED — " + "; ".join(errors))

        return True

    @staticmethod
    def _check_coordinates(valid_objects: list[dict]) -> bool:
        missing_geo: list[str] = []
        for obj in valid_objects:
            # PARTIAL_SUCCESS objects are expected to lack geo — skip them
            if obj.get("status") == "PARTIAL_SUCCESS":
                continue
            obj_id = obj.get("object_id", "<unknown>")
            geometry = obj.get("geometry") or {}
            geo = geometry.get("geo") or {}
            lon = geo.get("center_lon")
            lat = geo.get("center_lat")
            if lon is None or lat is None:
                missing_geo.append(obj_id)

        if missing_geo:
            raise ValueError(
                f"coordinate_present FAILED — VALID objects missing geo coordinates: "
                f"{missing_geo}"
            )
        return True
