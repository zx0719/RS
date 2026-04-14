"""
test_consistency.py — Unit tests for ConsistencyChecker.
"""

from __future__ import annotations

import pytest

from modules.eval import ConsistencyChecker


class TestConsistencyChecker:
    """Tests for ConsistencyChecker.check()."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_count_consistent_passes(self, minimal_evidence_package: dict) -> None:
        """A fully consistent package should pass all checks."""
        checker = ConsistencyChecker()
        result = checker.check(minimal_evidence_package)

        assert result["count_consistent"] is True
        assert result["class_consistent"] is True
        assert result["coordinate_present"] is True
        assert result["required_fields_present"] is True

    # ------------------------------------------------------------------
    # Count mismatch
    # ------------------------------------------------------------------

    def test_count_mismatch_detected(self, evidence_count_mismatch: dict) -> None:
        """statistics.totals.all_objects != len(VALID objects) must be flagged."""
        checker = ConsistencyChecker()
        with pytest.raises(ValueError, match="count_consistent FAILED"):
            checker.check(evidence_count_mismatch)

    # ------------------------------------------------------------------
    # Per-class count mismatch
    # ------------------------------------------------------------------

    def test_class_count_mismatch_detected(
        self, evidence_class_count_mismatch: dict
    ) -> None:
        """by_class count that disagrees with actual objects must be flagged."""
        checker = ConsistencyChecker()
        with pytest.raises(ValueError, match="class_consistent FAILED"):
            checker.check(evidence_class_count_mismatch)

    # ------------------------------------------------------------------
    # Missing geo coordinates
    # ------------------------------------------------------------------

    def test_missing_geo_detected(self, evidence_missing_geo: dict) -> None:
        """VALID objects without geo coordinates must be flagged."""
        checker = ConsistencyChecker()
        with pytest.raises(ValueError, match="coordinate_present FAILED"):
            checker.check(evidence_missing_geo)

    # ------------------------------------------------------------------
    # Missing required fields
    # ------------------------------------------------------------------

    def test_missing_required_field_detected(
        self, minimal_evidence_package: dict
    ) -> None:
        """Removing a required top-level field must be flagged."""
        minimal_evidence_package.pop("package_id")
        checker = ConsistencyChecker()
        with pytest.raises(ValueError, match="required_fields_present FAILED"):
            checker.check(minimal_evidence_package)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_objects_zero_total_passes(
        self, minimal_evidence_package: dict
    ) -> None:
        """Zero objects with totals=0 and empty by_class should pass."""
        minimal_evidence_package["objects"] = []
        minimal_evidence_package["statistics"]["totals"]["all_objects"] = 0
        minimal_evidence_package["statistics"]["by_class"] = {}
        checker = ConsistencyChecker()
        result = checker.check(minimal_evidence_package)
        assert result["count_consistent"] is True
        assert result["class_consistent"] is True
        assert result["coordinate_present"] is True

    def test_class_in_objects_missing_from_statistics_detected(
        self, minimal_evidence_package: dict
    ) -> None:
        """A class present in objects[] but absent from by_class must be flagged."""
        # Add a carrier without updating statistics
        import copy

        carrier = copy.deepcopy(minimal_evidence_package["objects"][0])
        carrier["object_id"] = "obj-000002"
        carrier["class"]["code"] = "carrier"
        carrier["class"]["name_cn"] = "航母"
        minimal_evidence_package["objects"].append(carrier)
        # Update total to 2 so count check passes, but by_class is still {destroyer: 1}
        minimal_evidence_package["statistics"]["totals"]["all_objects"] = 2

        checker = ConsistencyChecker()
        with pytest.raises(ValueError, match="class_consistent FAILED"):
            checker.check(minimal_evidence_package)
