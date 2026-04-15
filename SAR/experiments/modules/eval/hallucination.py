"""
hallucination.py — NLG hallucination detector for intelligence report bodies.

Checks whether the report.body text is consistent with the structured
statistics block:
  - Numbers mentioned in the text must match statistics.totals
  - Class names (EN codes or CN aliases) must not appear unless in statistics.by_class
  - Basic format compliance: title, body, date fields present
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Canonical class mappings
# ---------------------------------------------------------------------------

# English codes that the system recognises
_EN_CODES: tuple[str, ...] = (
    "carrier",
    "destroyer",
    "frigate",
    "replenishment",
    "amphibious",
    "other_vessel",
    "fighter",
    "bomber",
    "transport",
    "aew",
    "helicopter",
    "other_aircraft",
)

# Chinese display names mapped to their canonical code
_CN_TO_CODE: dict[str, str] = {
    "航母": "carrier",
    "驱逐舰": "destroyer",
    "护卫舰": "frigate",
    "补给舰": "replenishment",
    "两栖舰": "amphibious",
    "战斗机": "fighter",
    "轰炸机": "bomber",
    "运输机": "transport",
    "预警机": "aew",
    "直升机": "helicopter",
}

# Build a reverse map: code → [all text tokens that represent it]
_CODE_TO_TOKENS: dict[str, list[str]] = {code: [code] for code in _EN_CODES}
for cn, code in _CN_TO_CODE.items():
    _CODE_TO_TOKENS.setdefault(code, [code]).append(cn)

# All text tokens that map to a known class (used for scanning report body)
_ALL_CLASS_TOKENS: dict[str, str] = {}  # token → canonical code
for code, tokens in _CODE_TO_TOKENS.items():
    for token in tokens:
        _ALL_CLASS_TOKENS[token] = code


class HallucinationDetector:
    """Detect hallucinations between report.body and the statistics block."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, evidence_package: dict[str, Any]) -> dict[str, bool | str]:
        """Run all NLG hallucination checks.

        Parameters
        ----------
        evidence_package:
            An Evidence Package dict that should already contain
            ``statistics`` and ``report`` blocks.

        Returns
        -------
        dict
            The ``quality.nlg_checks`` block:

            .. code-block:: json

               {
                 "number_hallucination": false,
                 "class_hallucination": false,
                 "format_compliant": true
               }

            ``number_hallucination`` / ``class_hallucination`` are ``True``
            when a problem is detected (i.e. True means BAD).
        """
        report: dict = evidence_package.get("report") or {}
        statistics: dict = evidence_package.get("statistics") or {}
        body: str = report.get("body") or ""

        results: dict[str, Any] = {
            "number_hallucination": False,
            "class_hallucination": False,
            "format_compliant": True,
        }

        results["format_compliant"] = self._check_format(report)
        results["number_hallucination"] = self._check_numbers(body, statistics)
        results["class_hallucination"] = self._check_classes(body, statistics)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_format(report: dict[str, Any]) -> bool:
        """Return True when mandatory report fields are present and non-empty."""
        # Accept both "date" (legacy) and "report_date" (schema v1.0 pipeline output)
        has_date = bool(report.get("date") or report.get("report_date"))
        has_title = bool(report.get("title"))
        has_body = bool(report.get("body"))
        return has_title and has_body and has_date

    @staticmethod
    def _extract_count_numbers(text: str) -> list[int]:
        """Extract Arabic integers that look like counts from *text*.

        Deliberately excludes:
        - Integers that are part of a floating-point number (e.g. ``120.265``)
        - Integers that are part of a year-like 4-digit sequence preceded/followed
          by a date separator (heuristic only; year numbers are too large to match
          any plausible count and are filtered by magnitude later)

        The pattern ``(?<![\\.\\d])\\d+(?![\\d\\.])`` matches an integer that is
        not immediately preceded or followed by a digit or a period — this
        correctly skips ``120.265`` while matching standalone ``1``, ``3`` etc.
        """
        return [int(m) for m in re.findall(r"(?<![.\d])\d+(?![.\d])", text)]

    # Keep a simpler alias for internal use
    @staticmethod
    def _extract_numbers(text: str) -> list[int]:
        """Extract standalone integer counts from text.

        Excludes numbers that are:
        - Part of decimal literals (e.g. 120.236)
        - Attached to letters (e.g. GF-6, Qwen3, B-52)
        - Part of identifiers with hyphens (e.g. GF-6)
        """
        # First strip satellite/model identifiers like GF-6, GF-3, Qwen3
        cleaned = re.sub(r"[A-Za-z][-\w]*\d+\w*", "", text)
        cleaned = re.sub(r"\d+\w*[A-Za-z]\w*", "", cleaned)
        return [int(m) for m in re.findall(r"(?<![.\d])\d+(?![.\d])", cleaned)]

    def _check_numbers(
        self, body: str, statistics: dict[str, Any]
    ) -> bool:
        """Return True (hallucination detected) if any number in the body text
        is not explainable by the statistics block.

        Strategy: collect all integers that appear in statistics (totals + per-class),
        then flag any integer in the body that is NOT in that allowed set AND is
        greater than zero.  Numbers embedded in decimal literals (coordinates,
        resolution values, etc.) are excluded from scanning.

        We use a permissive approach: if the total count or any per-class count
        matches a number in the body we do NOT flag it.  Only numbers that cannot
        be accounted for are flagged.
        """
        if not body:
            return False

        totals: dict = statistics.get("totals") or {}
        raw_by_class = statistics.get("by_class") or {}

        # Build the set of numbers that statistics can justify
        allowed: set[int] = set()
        for v in totals.values():
            if v is not None:
                allowed.add(int(v))
        # schema v1.0: by_class is list-of-dicts; legacy: plain dict
        if isinstance(raw_by_class, list):
            for item in raw_by_class:
                if item.get("count") is not None:
                    allowed.add(int(item["count"]))
        else:
            for v in raw_by_class.values():
                if v is not None:
                    allowed.add(int(v))
        # Always allow 0
        allowed.add(0)

        # Strip date patterns like "2026年4月15日" before extracting numbers,
        # so year/month/day digits don't get flagged as hallucinations.
        body_stripped = re.sub(r"\d{4}年\d{1,2}月\d{1,2}日", "", body)
        body_numbers = self._extract_numbers(body_stripped)
        hallucinated = [n for n in body_numbers if n not in allowed and n > 0]
        return len(hallucinated) > 0

    @staticmethod
    def _check_classes(body: str, statistics: dict[str, Any]) -> bool:
        """Return True (hallucination detected) if the body mentions a class
        that does not appear in ``statistics.by_class``.
        """
        if not body:
            return False

        raw_by_class = statistics.get("by_class") or {}

        # Determine which canonical codes are legitimately present
        # schema v1.0: by_class is list-of-dicts; legacy: plain dict
        if isinstance(raw_by_class, list):
            legitimate_codes: set[str] = {
                item["code"] for item in raw_by_class if "code" in item
            }
        else:
            legitimate_codes = set(raw_by_class.keys())

        for token, code in _ALL_CLASS_TOKENS.items():
            if token in body and code not in legitimate_codes:
                return True  # body mentions a class not in statistics

        return False
