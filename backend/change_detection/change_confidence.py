"""
Change Confidence Score (CCS) Engine: False-Alarm Suppression
Heuristic, multi-factor fusion metric combining:
1. Raw Open-CD change evidence (change_score)
2. s2cloudless clear-sky fraction (1 - cloud_score)
3. AROSICS sub-pixel registration quality (reg_quality_score)
4. Multi-temporal persistence consistency (temp_consistency_score)

CCS Formula:
    CCS = 0.4 * change_score + 0.2 * (1 - cloud_score) + 
          0.2 * reg_quality_score + 0.2 * temp_consistency_score

Confidence Categories:
    - High Confidence: CCS >= 0.70 → "confirmed_change"
    - Needs Review: 0.40 <= CCS < 0.70 → "review_flagged"
    - Suppressed: CCS < 0.40 → "low_confidence"

Explicitly NOT a calibrated probability. Displayed only as a heuristic score.
"""
import logging
from typing import Tuple, Literal

from backend.config import (
    CCS_W1_CHANGE, CCS_W2_CLEAR_SKY, CCS_W3_REGISTRATION, CCS_W4_CONSISTENCY
)

logger = logging.getLogger(__name__)


class ConfidenceScore(float):
    """Numeric CCS value that also supports legacy tuple unpacking."""

    def __new__(cls, value: float, category: str):
        instance = super().__new__(cls, value)
        instance.category = category
        return instance

    def __iter__(self):
        yield float(self)
        yield self.category


def compute_change_confidence_score(
    change_evidence: float,
    cloud_score: float,
    registration_quality: float,
    temporal_consistency: float = 1.0
) -> Tuple[float, Literal["confirmed_change", "review_flagged", "low_confidence"]]:
    """
    Compute heuristic Change Confidence Score (CCS) from multiple quality metrics.
    
    This score combines:
    1. Raw change evidence from the deep learning model (40% weight)
    2. Clear-sky quality: (1 - cloud_score) (20% weight)
    3. Registration alignment quality (20% weight)
    4. Temporal consistency across multi-date observations (20% weight)
    
    The CCS is NOT a calibrated probability; it's a heuristic fusion score.
    
    Args:
        change_evidence: Raw change score from Open-CD [0.0, 1.0]
        cloud_score: Fraction of cloudy pixels [0.0, 1.0]
        registration_quality: Registration alignment quality [0.0, 1.0]
        temporal_consistency: Multi-temporal persistence metric [0.0, 1.0] (default 1.0)
    
    Returns:
        (ccs_score, confidence_category) where:
            ccs_score: Combined heuristic confidence [0.0, 1.0]
            confidence_category: "confirmed_change" | "review_flagged" | "low_confidence"
    
    Examples:
        >>> ccs, cat = compute_change_confidence_score(0.85, 0.05, 0.95, 1.0)
        >>> # High confidence change: all metrics are good
        >>> ccs, cat = compute_change_confidence_score(0.72, 0.40, 0.80, 0.8)
        >>> # Needs review: cloud contamination present
        >>> ccs, cat = compute_change_confidence_score(0.35, 0.60, 0.65, 0.5)
        >>> # Suppressed: low change evidence + clouds
    """
    # Clamp inputs to [0.0, 1.0]
    change_evidence = float(max(0.0, min(1.0, change_evidence)))
    cloud_score = float(max(0.0, min(1.0, cloud_score)))
    registration_quality = float(max(0.0, min(1.0, registration_quality)))
    temporal_consistency = float(max(0.0, min(1.0, temporal_consistency)))
    
    # Compute clear-sky quality (1 - cloud_score)
    clear_sky_quality = 1.0 - cloud_score
    
    # Weighted combination
    ccs_score = (
        CCS_W1_CHANGE * change_evidence +
        CCS_W2_CLEAR_SKY * clear_sky_quality +
        CCS_W3_REGISTRATION * registration_quality +
        CCS_W4_CONSISTENCY * temporal_consistency
    )
    
    # Clamp to [0.0, 1.0]
    ccs_score = float(max(0.0, min(1.0, ccs_score)))
    
    # Categorize based on thresholds
    if ccs_score >= 0.70:
        category = "confirmed_change"
    elif ccs_score >= 0.40:
        category = "review_flagged"
    else:
        category = "low_confidence"
    
    logger.debug(
        f"CCS computed: score={ccs_score:.3f}, category={category}, "
        f"(change={change_evidence:.2f}, clear={clear_sky_quality:.2f}, "
        f"reg={registration_quality:.2f}, temp_cons={temporal_consistency:.2f})"
    )
    
    return ConfidenceScore(ccs_score, category)


def categorize_confidence(ccs_score: float) -> Literal["confirmed_change", "review_flagged", "low_confidence"]:
    """
    Categorize CCS score into confidence levels.
    
    Args:
        ccs_score: CCS score [0.0, 1.0]
    
    Returns:
        Confidence category
    """
    if ccs_score >= 0.70:
        return "high_confidence"
    elif ccs_score >= 0.40:
        return "needs_review"
    else:
        return "suppressed"
