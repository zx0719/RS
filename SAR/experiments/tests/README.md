# Eval QA Test Suite

Quality evaluation and consistency tests for the SAR intelligence reporting pipeline.

## Running Tests

```bash
cd /home/zhuxiang/RS/SAR/experiments
pytest tests/
```

Verbose output with per-test status:

```bash
pytest tests/ -v
```

Run a single test file:

```bash
pytest tests/test_consistency.py -v
pytest tests/test_hallucination.py -v
pytest tests/test_integration.py -v
```

## Test Structure

| File | Description |
|------|-------------|
| `conftest.py` | Shared fixtures (minimal package, hallucination sample, missing-geo sample) |
| `test_consistency.py` | Unit tests for `ConsistencyChecker` |
| `test_hallucination.py` | Unit tests for `HallucinationDetector` |
| `test_integration.py` | End-to-end stub tests using mock M1–M5 pipeline functions |

## Fixtures

| Fixture | Description |
|---------|-------------|
| `minimal_evidence_package` | 1 destroyer, harbor scene — fully consistent |
| `evidence_with_hallucination` | report.body claims 3 ships / mentions 航母 not in statistics |
| `evidence_missing_geo` | VALID object has empty `geometry.geo` |
| `evidence_count_mismatch` | `statistics.totals.all_objects = 2` but only 1 VALID object |
| `evidence_class_count_mismatch` | `by_class.destroyer = 2` but only 1 destroyer object |

## Modules Under Test

```
modules/eval/
├── __init__.py           exports ConsistencyChecker, HallucinationDetector, QualityGate
├── consistency.py        internal count / class / geo / field checks
├── hallucination.py      regex-based NLG vs statistics checks
└── quality_gate.py       orchestrator — fills quality block and review_gate
```
