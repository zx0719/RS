"""modules.eval — Evidence Package quality evaluation sub-package.

Public exports::

    from modules.eval import ConsistencyChecker, HallucinationDetector, QualityGate
"""

from .consistency import ConsistencyChecker
from .hallucination import HallucinationDetector
from .quality_gate import QualityGate

__all__ = [
    "ConsistencyChecker",
    "HallucinationDetector",
    "QualityGate",
]
