"""
test_nlg_training.py — Tests for the NLG training data pipeline.

Covers:
  1. test_parse_caption_extracts_counts
  2. test_parse_caption_handles_no_ships
  3. test_build_synthetic_evidence_valid_schema
  4. test_generate_chinese_report_no_hallucination
  5. test_validate_training_item
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing from experiments root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.build_nlg_training_data import CaptionToEvidenceConverter
from modules.evidence.schema_validator import validate_evidence_package, SchemaValidationError
from modules.report.training_utils import validate_training_item


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def converter() -> CaptionToEvidenceConverter:
    return CaptionToEvidenceConverter(seed=0)


# ---------------------------------------------------------------------------
# Test 1: parse_caption extracts counts correctly
# ---------------------------------------------------------------------------

class TestParseCaptionExtractsCounts:
    """parse_caption should recognise number words and map types to codes."""

    def test_five_cargo_ships(self, converter: CaptionToEvidenceConverter) -> None:
        """'five cargo ships' should yield other_vessel with count 5."""
        parsed = converter.parse_caption(
            "A SAR image shows five cargo ships anchored in the harbor."
        )
        objects = parsed["objects"]
        assert objects, "Expected at least one object entry"
        codes = {o["code"] for o in objects}
        assert "other_vessel" in codes, f"Expected 'other_vessel' in codes, got {codes}"
        total_vessels = sum(o["count"] for o in objects if o["code"] == "other_vessel")
        assert total_vessels == 5, f"Expected count 5, got {total_vessels}"

    def test_digit_count(self, converter: CaptionToEvidenceConverter) -> None:
        """Digit counts ('3 destroyers') should also be extracted."""
        parsed = converter.parse_caption(
            "The scene contains 3 warships moving in formation."
        )
        objects = parsed["objects"]
        assert objects, "Expected at least one object entry"
        total = sum(o["count"] for o in objects)
        assert total >= 3, f"Expected total >= 3, got {total}"

    def test_frigate_recognised(self, converter: CaptionToEvidenceConverter) -> None:
        """'frigate' should map to the 'frigate' class code."""
        parsed = converter.parse_caption(
            "Two frigates are sailing near the coastline."
        )
        codes = {o["code"] for o in parsed["objects"]}
        assert "frigate" in codes, f"Expected 'frigate', got {codes}"

    def test_aircraft_carrier_recognised(self, converter: CaptionToEvidenceConverter) -> None:
        """'aircraft carrier' should map to 'carrier'."""
        parsed = converter.parse_caption(
            "An aircraft carrier is positioned in open water."
        )
        codes = {o["code"] for o in parsed["objects"]}
        assert "carrier" in codes, f"Expected 'carrier', got {codes}"

    def test_multiple_types(self, converter: CaptionToEvidenceConverter) -> None:
        """Multiple ship types in one caption should all be extracted."""
        parsed = converter.parse_caption(
            "Three frigates and two tankers are anchored at the pier."
        )
        codes = {o["code"] for o in parsed["objects"]}
        assert "frigate" in codes, "Expected frigate"
        assert "replenishment" in codes, "Expected replenishment (tanker)"


# ---------------------------------------------------------------------------
# Test 2: parse_caption handles captions with no ships/aircraft
# ---------------------------------------------------------------------------

class TestParseCaptionHandlesNoShips:
    """Captions without recognisable military targets should return empty objects."""

    def test_urban_caption(self, converter: CaptionToEvidenceConverter) -> None:
        """Urban area description contains no ship/aircraft keywords."""
        parsed = converter.parse_caption(
            "An urban area with buildings and roads, no military targets visible."
        )
        assert parsed["objects"] == [], (
            f"Expected empty objects list, got {parsed['objects']}"
        )

    def test_vegetation_caption(self, converter: CaptionToEvidenceConverter) -> None:
        """Vegetation/terrain description should yield no objects."""
        parsed = converter.parse_caption(
            "Dense forest vegetation with rivers and agricultural land."
        )
        assert parsed["objects"] == []

    def test_no_count_no_object(self, converter: CaptionToEvidenceConverter) -> None:
        """Caption mentioning a ship type without a count should yield no objects.

        The parser only records an object when a numeric count can be found
        in the window preceding the keyword.
        """
        parsed = converter.parse_caption(
            "The image depicts a typical harbor environment."
        )
        # 'harbor' is a location hint, not a ship/aircraft, so objects should be empty
        assert parsed["objects"] == []

    def test_spatial_summary_empty_when_no_objects(
        self, converter: CaptionToEvidenceConverter
    ) -> None:
        """Spatial summary should be empty string when no objects are found."""
        parsed = converter.parse_caption("A featureless open ocean scene.")
        assert parsed["objects"] == []
        assert parsed["spatial_summary"] == ""


# ---------------------------------------------------------------------------
# Test 3: build_synthetic_evidence produces a schema-valid package
# ---------------------------------------------------------------------------

class TestBuildSyntheticEvidenceValidSchema:
    """build_synthetic_evidence output should pass schema_validator."""

    def _make_parsed(self) -> dict:
        return {
            "objects": [
                {"code": "destroyer", "count": 2},
                {"code": "frigate", "count": 1},
            ],
            "location_hint": "港口区域",
            "spatial_summary": "分散于港口区域",
        }

    def test_schema_passes(self, converter: CaptionToEvidenceConverter) -> None:
        """A correctly built package should not raise SchemaValidationError."""
        parsed = self._make_parsed()
        evidence = converter.build_synthetic_evidence(
            parsed,
            image_path="/mnt/data/mm_data/SAR/FSAR-Cap/all_images/GF3_SAY_SL_001.png",
        )
        # Should not raise
        validate_evidence_package(evidence)

    def test_status_is_ready_for_nlg(self, converter: CaptionToEvidenceConverter) -> None:
        """Status must be READY_FOR_NLG so the generator accepts it."""
        parsed = self._make_parsed()
        evidence = converter.build_synthetic_evidence(parsed, "GF3_test.png")
        assert evidence["status"] == "READY_FOR_NLG"

    def test_object_count_matches_totals(self, converter: CaptionToEvidenceConverter) -> None:
        """statistics.totals.all_objects must equal len(objects)."""
        parsed = self._make_parsed()
        evidence = converter.build_synthetic_evidence(parsed, "GF3_test.png")
        assert evidence["statistics"]["totals"]["all_objects"] == len(evidence["objects"])

    def test_gf3_satellite_inferred(self, converter: CaptionToEvidenceConverter) -> None:
        """Filename starting with GF3 should produce satellite '高分三号'."""
        parsed = self._make_parsed()
        evidence = converter.build_synthetic_evidence(
            parsed, "/some/path/GF3_HARBOR_001.png"
        )
        sat = evidence["input"]["metadata"]["satellite"]
        assert sat == "高分三号", f"Expected '高分三号', got '{sat}'"

    def test_package_id_format(self, converter: CaptionToEvidenceConverter) -> None:
        """package_id must start with 'sar-'."""
        parsed = self._make_parsed()
        evidence = converter.build_synthetic_evidence(parsed, "test_image.png")
        assert evidence["package_id"].startswith("sar-")

    def test_by_class_sums_to_totals(self, converter: CaptionToEvidenceConverter) -> None:
        """Sum of by_class counts must equal totals.ships + totals.aircraft."""
        parsed = self._make_parsed()
        evidence = converter.build_synthetic_evidence(parsed, "test.png")
        stats = evidence["statistics"]
        by_class_total = sum(e["count"] for e in stats["by_class"])
        expected = stats["totals"]["ships"] + stats["totals"]["aircraft"]
        assert by_class_total == expected


# ---------------------------------------------------------------------------
# Test 4: generate_chinese_report — no hallucination
# ---------------------------------------------------------------------------

class TestGenerateChineseReportNoHallucination:
    """The generated Chinese report should be consistent with the evidence statistics."""

    def _make_evidence(
        self,
        converter: CaptionToEvidenceConverter,
        objects: list[dict],
        location: str = "港口区域",
    ) -> dict:
        parsed = {
            "objects": objects,
            "location_hint": location,
            "spatial_summary": "集中分布于港口",
        }
        return converter.build_synthetic_evidence(parsed, "GF3_test.png")

    def test_destroyer_count_in_report(
        self, converter: CaptionToEvidenceConverter
    ) -> None:
        """Report should mention the correct destroyer count."""
        import re as _re
        evidence = self._make_evidence(converter, [{"code": "destroyer", "count": 3}])
        report = converter.generate_chinese_report(evidence)
        assert report, "Report should not be empty"
        # Check that the number 3 appears (as digit or implied) near a ship unit
        ship_mentions = _re.findall(r"(\d+)\s*艘", report)
        assert ship_mentions, f"No '艘' count found in report: {report}"
        counts = [int(x) for x in ship_mentions]
        assert any(c == 3 for c in counts), (
            f"Expected count 3 in report, found counts {counts}: {report}"
        )

    def test_report_length_in_range(
        self, converter: CaptionToEvidenceConverter
    ) -> None:
        """Report length should be between 50 and 400 characters."""
        evidence = self._make_evidence(
            converter,
            [{"code": "frigate", "count": 2}, {"code": "destroyer", "count": 1}],
        )
        report = converter.generate_chinese_report(evidence)
        assert 50 <= len(report) <= 400, (
            f"Report length {len(report)} out of expected range: {report}"
        )

    def test_report_starts_with_expected_prefix(
        self, converter: CaptionToEvidenceConverter
    ) -> None:
        """Template fallback report should start with '据'."""
        evidence = self._make_evidence(converter, [{"code": "other_vessel", "count": 4}])
        report = converter.generate_chinese_report(evidence)
        assert report.startswith("据"), (
            f"Report should start with '据', got: {report[:20]}"
        )

    def test_no_extra_class_names(
        self, converter: CaptionToEvidenceConverter
    ) -> None:
        """Report must not mention class names that are not in by_class."""
        evidence = self._make_evidence(converter, [{"code": "frigate", "count": 2}])
        report = converter.generate_chinese_report(evidence)

        # These classes are NOT in the evidence, so they must not appear in the report
        forbidden_classes = ["航空母舰", "驱逐舰", "补给舰", "战斗机", "轰炸机"]
        for cls in forbidden_classes:
            assert cls not in report, (
                f"Hallucinated class '{cls}' found in report: {report}"
            )

    def test_aircraft_count_in_report(
        self, converter: CaptionToEvidenceConverter
    ) -> None:
        """Report should mention the correct fighter count."""
        import re as _re
        evidence = self._make_evidence(
            converter,
            [{"code": "fighter", "count": 6}],
            location="机场",
        )
        report = converter.generate_chinese_report(evidence)
        aircraft_mentions = _re.findall(r"(\d+)\s*架", report)
        assert aircraft_mentions, f"No '架' count found in report: {report}"
        counts = [int(x) for x in aircraft_mentions]
        assert any(c == 6 for c in counts), (
            f"Expected count 6 in report, found counts {counts}: {report}"
        )


# ---------------------------------------------------------------------------
# Test 5: validate_training_item
# ---------------------------------------------------------------------------

class TestValidateTrainingItem:
    """validate_training_item should correctly accept valid and reject invalid items."""

    def _make_valid_item(self) -> dict:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一名军事情报分析助手，负责根据SAR卫星侦察数据生成标准军事情报通报正文。",
                },
                {
                    "role": "user",
                    "content": "请根据以下 <EVIDENCE> 生成SAR卫星情报通报正文。\n<EVIDENCE>\n卫星：高分三号\n</EVIDENCE>",
                },
                {
                    "role": "assistant",
                    "content": "据高分三号卫星某日侦察，某港口区域共发现舰船3艘，其中驱逐舰2艘、护卫舰1艘，目标集中分布于港口区域。",
                },
            ]
        }

    def test_valid_item_passes(self) -> None:
        """A correctly formed training item should pass validation."""
        item = self._make_valid_item()
        ok, reason = validate_training_item(item)
        assert ok is True, f"Expected valid, got reason: {reason}"
        assert reason == ""

    def test_missing_messages_key(self) -> None:
        """Item without 'messages' key should fail."""
        ok, reason = validate_training_item({"data": []})
        assert ok is False
        assert "messages" in reason.lower()

    def test_wrong_role_order(self) -> None:
        """Messages with wrong role order should fail."""
        item = self._make_valid_item()
        # Swap system and user
        item["messages"][0], item["messages"][1] = item["messages"][1], item["messages"][0]
        ok, reason = validate_training_item(item)
        assert ok is False
        assert "role" in reason.lower()

    def test_empty_assistant_content(self) -> None:
        """Empty assistant content should fail."""
        item = self._make_valid_item()
        item["messages"][2]["content"] = ""
        ok, reason = validate_training_item(item)
        assert ok is False

    def test_too_short_assistant_content(self) -> None:
        """Very short assistant content (< min length) should fail."""
        item = self._make_valid_item()
        item["messages"][2]["content"] = "短"  # 1 char
        ok, reason = validate_training_item(item)
        assert ok is False
        assert "short" in reason.lower() or "chars" in reason.lower()

    def test_non_dict_item(self) -> None:
        """A non-dict item should fail."""
        ok, reason = validate_training_item("not a dict")  # type: ignore[arg-type]
        assert ok is False
        assert "dict" in reason.lower()

    def test_too_few_messages(self) -> None:
        """Fewer than 3 messages should fail."""
        item = self._make_valid_item()
        item["messages"] = item["messages"][:2]
        ok, reason = validate_training_item(item)
        assert ok is False
        assert "3" in reason

    def test_empty_system_content(self) -> None:
        """Empty system message content should fail."""
        item = self._make_valid_item()
        item["messages"][0]["content"] = "   "  # whitespace only
        ok, reason = validate_training_item(item)
        assert ok is False
