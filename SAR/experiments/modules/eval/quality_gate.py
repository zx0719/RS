"""
quality_gate.py — Orchestrates all eval checks and fills the ``quality`` block.

Usage::

    from modules.eval import QualityGate

    gate = QualityGate()
    updated_package = gate.evaluate(evidence_package)
    # evidence_package["quality"] is now fully populated
"""

from __future__ import annotations

from typing import Any

from .consistency import ConsistencyChecker
from .hallucination import HallucinationDetector


class QualityGate:
    """Run consistency + hallucination checks and fill ``quality`` in-place."""

    def __init__(self) -> None:
        self._consistency = ConsistencyChecker()
        self._hallucination = HallucinationDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, evidence_package: dict[str, Any]) -> dict[str, Any]:
        """Run all checks, populate ``evidence_package["quality"]``, and return
        the updated package.

        The method never raises; individual check failures are caught, recorded
        as ``False`` in the appropriate sub-block, and the failure message is
        appended to ``quality.errors``.

        Parameters
        ----------
        evidence_package:
            A mutable Evidence Package dict.  The ``quality`` key will be
            created or overwritten.

        Returns
        -------
        dict
            The same *evidence_package* dict with the ``quality`` block filled.
        """
        quality: dict[str, Any] = {
            "consistency_checks": {},
            "nlg_checks": {},
            "review_gate": {
                "needs_human_review": False,
                "reason": [],
            },
            "errors": [],
        }

        # ---- 1. Consistency checks ----------------------------------------
        consistency_results = self._run_consistency(evidence_package, quality)

        # ---- 2. Hallucination checks ----------------------------------------
        hallucination_results = self._run_hallucination(evidence_package, quality)

        # ---- 3. Review gate logic -------------------------------------------
        self._evaluate_review_gate(
            evidence_package, consistency_results, hallucination_results, quality
        )

        evidence_package["quality"] = quality
        return evidence_package

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_consistency(
        self,
        pkg: dict[str, Any],
        quality: dict[str, Any],
    ) -> dict[str, bool]:
        """Run ConsistencyChecker; catch errors and record them."""
        default = {
            "count_consistent": False,
            "class_consistent": False,
            "coordinate_present": False,
            "required_fields_present": False,
        }
        try:
            results = self._consistency.check(pkg)
            quality["consistency_checks"] = results
            return results
        except ValueError as exc:
            quality["errors"].append(f"[consistency] {exc}")
            # Re-run silently to get partial results
            partial = dict(default)
            for key in default:
                try:
                    # Attempt individual sub-checks via separate checker
                    sub_checker = ConsistencyChecker()
                    individual = sub_checker.check(pkg)
                    partial.update(individual)
                    break
                except ValueError:
                    pass
            quality["consistency_checks"] = partial
            return partial

    def _run_hallucination(
        self,
        pkg: dict[str, Any],
        quality: dict[str, Any],
    ) -> dict[str, Any]:
        """Run HallucinationDetector; catch errors and record them."""
        default: dict[str, Any] = {
            "number_hallucination": False,
            "class_hallucination": False,
            "format_compliant": False,
        }
        try:
            results = self._hallucination.check(pkg)
            quality["nlg_checks"] = results
            return results
        except Exception as exc:  # noqa: BLE001
            quality["errors"].append(f"[hallucination] {exc}")
            quality["nlg_checks"] = default
            return default

    @staticmethod
    def _evaluate_review_gate(
        pkg: dict[str, Any],
        consistency: dict[str, bool],
        hallucination: dict[str, Any],
        quality: dict[str, Any],
    ) -> None:
        """Determine whether human review is required and explain why."""
        reasons: list[str] = []

        # --- Low-confidence objects ----------------------------------------
        objects: list[dict] = pkg.get("objects") or []
        low_conf_ids = [
            obj.get("object_id", "<unknown>")
            for obj in objects
            if _is_low_confidence(obj)
        ]
        if low_conf_ids:
            reasons.append(
                f"Low-confidence objects detected: {low_conf_ids}"
            )

        # --- Consistency failures -------------------------------------------
        if not consistency.get("count_consistent", True):
            reasons.append("Object count is inconsistent with statistics.totals")
        if not consistency.get("class_consistent", True):
            reasons.append("Per-class counts are inconsistent with statistics.by_class")
        if not consistency.get("coordinate_present", True):
            reasons.append("One or more VALID objects are missing geo coordinates")
        if not consistency.get("required_fields_present", True):
            reasons.append("Required Evidence Package fields are missing")

        # --- Hallucination flags -------------------------------------------
        if hallucination.get("number_hallucination"):
            reasons.append("Report body contains numbers not present in statistics")
        if hallucination.get("class_hallucination"):
            reasons.append(
                "Report body mentions target classes not present in statistics.by_class"
            )
        if not hallucination.get("format_compliant", True):
            reasons.append("Report is missing required fields (title / body / date)")

        needs_review = len(reasons) > 0
        quality["review_gate"]["needs_human_review"] = needs_review
        quality["review_gate"]["reason"] = reasons


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_LOW_CONF_THRESHOLD = 0.6  # configurable if needed


def _is_low_confidence(obj: dict[str, Any]) -> bool:
    score: dict = obj.get("score") or {}
    conf = score.get("calibrated_confidence") or score.get("confidence")
    if conf is None:
        return False
    return float(conf) < _LOW_CONF_THRESHOLD
