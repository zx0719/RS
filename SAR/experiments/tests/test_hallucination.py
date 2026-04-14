"""
test_hallucination.py — Unit tests for HallucinationDetector.
"""

from __future__ import annotations

import pytest

from modules.eval import HallucinationDetector


class TestHallucinationDetector:
    """Tests for HallucinationDetector.check()."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_no_hallucination_passes(self, minimal_evidence_package: dict) -> None:
        """A fully consistent report should return no hallucination flags."""
        detector = HallucinationDetector()
        result = detector.check(minimal_evidence_package)

        assert result["number_hallucination"] is False
        assert result["class_hallucination"] is False
        assert result["format_compliant"] is True

    # ------------------------------------------------------------------
    # Number hallucination
    # ------------------------------------------------------------------

    def test_number_hallucination_detected(
        self, evidence_with_hallucination: dict
    ) -> None:
        """Body mentioning a count (3) not in statistics must be flagged."""
        detector = HallucinationDetector()
        result = detector.check(evidence_with_hallucination)
        assert result["number_hallucination"] is True

    def test_number_matching_statistics_not_flagged(
        self, minimal_evidence_package: dict
    ) -> None:
        """Numbers that appear in statistics should NOT be flagged."""
        # The base package body says "1 艘" which matches totals.all_objects = 1
        detector = HallucinationDetector()
        result = detector.check(minimal_evidence_package)
        assert result["number_hallucination"] is False

    # ------------------------------------------------------------------
    # Class hallucination
    # ------------------------------------------------------------------

    def test_class_hallucination_detected(
        self, evidence_with_hallucination: dict
    ) -> None:
        """Body mentioning a class (航母/carrier) absent from by_class must be flagged."""
        detector = HallucinationDetector()
        result = detector.check(evidence_with_hallucination)
        assert result["class_hallucination"] is True

    def test_class_present_in_statistics_not_flagged(
        self, minimal_evidence_package: dict
    ) -> None:
        """Mentioning 驱逐舰 when destroyer is in by_class must NOT be flagged."""
        detector = HallucinationDetector()
        result = detector.check(minimal_evidence_package)
        assert result["class_hallucination"] is False

    def test_english_class_hallucination_detected(
        self, minimal_evidence_package: dict
    ) -> None:
        """Body mentioning English class name not in by_class must be flagged."""
        minimal_evidence_package["report"]["body"] = (
            "Detected 1 destroyer and 1 carrier at the harbor."
        )
        # statistics only has destroyer; carrier is a hallucination
        detector = HallucinationDetector()
        result = detector.check(minimal_evidence_package)
        assert result["class_hallucination"] is True

    # ------------------------------------------------------------------
    # Format compliance
    # ------------------------------------------------------------------

    def test_missing_title_not_format_compliant(
        self, minimal_evidence_package: dict
    ) -> None:
        """Report missing title field must fail format compliance."""
        minimal_evidence_package["report"].pop("title")
        detector = HallucinationDetector()
        result = detector.check(minimal_evidence_package)
        assert result["format_compliant"] is False

    def test_missing_date_not_format_compliant(
        self, minimal_evidence_package: dict
    ) -> None:
        """Report missing date field must fail format compliance."""
        minimal_evidence_package["report"].pop("date")
        detector = HallucinationDetector()
        result = detector.check(minimal_evidence_package)
        assert result["format_compliant"] is False

    def test_missing_body_not_format_compliant(
        self, minimal_evidence_package: dict
    ) -> None:
        """Report with empty body must fail format compliance."""
        minimal_evidence_package["report"]["body"] = ""
        detector = HallucinationDetector()
        result = detector.check(minimal_evidence_package)
        assert result["format_compliant"] is False

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_body_no_hallucination(
        self, minimal_evidence_package: dict
    ) -> None:
        """An empty body cannot hallucinate numbers or classes."""
        minimal_evidence_package["report"]["body"] = ""
        detector = HallucinationDetector()
        result = detector.check(minimal_evidence_package)
        assert result["number_hallucination"] is False
        assert result["class_hallucination"] is False

    def test_no_report_block_returns_defaults(
        self, minimal_evidence_package: dict
    ) -> None:
        """Missing report block should not raise — format_compliant should be False."""
        minimal_evidence_package.pop("report")
        detector = HallucinationDetector()
        result = detector.check(minimal_evidence_package)
        assert result["format_compliant"] is False
        assert result["number_hallucination"] is False
        assert result["class_hallucination"] is False
