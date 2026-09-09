"""
Multi-Temporal Earliest-Change-Date Estimator & Consistency Engine
Walks chronologically ordered time-series acquisitions at a fixed location_key.
Determines onset date by checking persistent threshold elevation vs. transient spikes.

Algorithm:
1. Filter out heavily clouded observations
2. Walk forward through timeline
3. Mark transition point when change_score crosses threshold
4. Verify persistence: next N observations also exceed threshold (reduces false positives)
5. Return earliest consistent change date
"""
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def estimate_earliest_change_date(
    timeline: List[Dict[str, Any]],
    threshold: float = 0.40,
    persistence_lookhead: int = 2,
    max_cloud_fraction: float = 0.30
) -> Tuple[Optional[str], float, float]:
    """
    Estimate the earliest date when persistent change occurred at a location.
    
    Multi-temporal consistency filtering:
    1. Discards observations with cloud_score > max_cloud_fraction
    2. Identifies transition point where change_score crosses threshold
    3. Verifies persistence: next N observations also exceed threshold
    4. Returns earliest confirmed change date or None if no persistent change detected
    
    Args:
        timeline: List of dicts ordered chronologically by date:
            [
                {"date": "2024-03-01", "change_score": 0.05, "cloud_score": 0.05},
                {"date": "2024-10-15", "change_score": 0.12, "cloud_score": 0.08},
                {"date": "2025-05-20", "change_score": 0.72, "cloud_score": 0.02},
                {"date": "2025-11-10", "change_score": 0.78, "cloud_score": 0.04}
            ]
        threshold: Change score threshold to mark transition (default 0.40)
        persistence_lookhead: Number of subsequent observations to verify consistency (default 2)
        max_cloud_fraction: Discard observations with cloud_score > this value (default 0.30)
    
    Returns:
        earliest_date: ISO date string (YYYY-MM-DD) or None if no persistent change
        max_change_score: Maximum change score in timeline
        change_persistence: Fraction of observations above threshold after earliest change
    """
    if not timeline:
        return None, 0.0, 0.0
    
    # Sort by date
    sorted_timeline = sorted(timeline, key=lambda x: str(x.get('date', '')))
    
    # Filter out heavily clouded observations
    clean_timeline = [
        obs for obs in sorted_timeline
        if obs.get('cloud_score', 0.0) <= max_cloud_fraction
    ]
    
    if not clean_timeline:
        logger.warning("All observations heavily clouded; no clean timeline")
        max_score = max(obs.get('change_score', 0.0) for obs in sorted_timeline)
        return None, max_score, 0.0
    
    # Find maximum change score
    max_change_score = max(obs.get('change_score', 0.0) for obs in clean_timeline)
    
    # Walk forward to find earliest transition with persistence verification
    earliest_date = None
    for i, obs in enumerate(clean_timeline):
        if obs.get('change_score', 0.0) > threshold:
            # Check persistence: verify that next 'persistence_lookhead' observations also exceed threshold
            subsequent = clean_timeline[i:i + persistence_lookhead + 1]
            
            # Count how many of the subsequent observations exceed threshold
            exceeding_count = sum(1 for o in subsequent if o.get('change_score', 0.0) > threshold)
            
            # If majority of subsequent observations exceed threshold, mark as earliest change
            if exceeding_count >= min(persistence_lookhead, len(subsequent)):
                earliest_date = obs.get('date')
                break
    
    # Compute change persistence (fraction of observations above threshold after earliest change)
    if earliest_date:
        earliest_idx = next(
            (i for i, o in enumerate(clean_timeline) if o.get('date') == earliest_date),
            -1
        )
        if earliest_idx >= 0:
            remaining = clean_timeline[earliest_idx:]
            change_persistence = (
                sum(1 for o in remaining if o.get('change_score', 0.0) > threshold) / len(remaining)
                if remaining else 0.0
            )
        else:
            change_persistence = 0.0
    else:
        change_persistence = 0.0
    
    logger.debug(
        f"Earliest change date: {earliest_date}, max_change={max_change_score:.3f}, "
        f"persistence={change_persistence:.3f}"
    )
    
    return earliest_date, max_change_score, change_persistence


def filter_temporal_outliers(
    timeline: List[Dict[str, Any]],
    change_score_key: str = 'change_score',
    window_size: int = 3,
    percentile_threshold: float = 75.0
) -> List[Dict[str, Any]]:
    """
    Filter out single-date outliers (spikes) in change scores using percentile-based windowing.
    
    Robust to cloud-induced transient spikes by examining local temporal context.
    
    Args:
        timeline: Chronologically ordered list of observations
        change_score_key: Key in each observation dict for the score to filter
        window_size: Rolling window size for local percentile computation
        percentile_threshold: Percentile threshold for outlier detection (75 = 3rd quartile)
    
    Returns:
        Filtered timeline with outliers replaced or removed
    """
    if len(timeline) < window_size:
        return timeline
    
    filtered = []
    
    for i, obs in enumerate(timeline):
        # Compute local window percentile
        start = max(0, i - window_size // 2)
        end = min(len(timeline), i + window_size // 2 + 1)
        window = timeline[start:end]
        
        scores = [o.get(change_score_key, 0.0) for o in window]
        local_percentile = np.percentile(scores, percentile_threshold)
        
        obs_score = obs.get(change_score_key, 0.0)
        
        # Keep observations that are within local percentile or represent genuine changes
        if obs_score <= local_percentile or obs_score > local_percentile * 1.5:
            filtered.append(obs)
    
    return filtered


def compute_temporal_consistency_score(
    timeline: List[Dict[str, Any]],
    change_score_key: str = 'change_score',
    window_size: int = 5
) -> float:
    """
    Compute temporal consistency score (0.0 to 1.0).
    
    Measures how stable/consistent the change score is over time.
    Higher values indicate more persistent change; lower values indicate noise or spikes.
    
    Formula:
        consistency = 1.0 - (temporal_variance / max_possible_variance)
    
    Args:
        timeline: Chronologically ordered observations
        change_score_key: Key for score values
        window_size: Window for local variance computation
    
    Returns:
        Consistency score [0.0, 1.0]
    """
    if len(timeline) < 2:
        return 1.0
    
    scores = [obs.get(change_score_key, 0.0) for obs in timeline]
    
    # Compute rolling variance
    local_variances = []
    for i in range(len(scores) - window_size + 1):
        window = scores[i:i + window_size]
        local_var = np.var(window)
        local_variances.append(local_var)
    
    if not local_variances:
        return 1.0
    
    # Average local variance
    mean_variance = np.mean(local_variances)
    
    # Max possible variance (uniform distribution [0, 1])
    max_variance = 0.25  # Var of Uniform(0, 1)
    
    # Consistency = low variance
    consistency = 1.0 - min(1.0, mean_variance / max_variance)
    
    return float(consistency)

    for i in range(1, n):
        score = float(sorted_timeline[i].get("change_score", 0.0))
        if score >= threshold:
            # Verify if it remains elevated in all subsequent dates
            subsequent_elevated = True
            for j in range(i + 1, n):
                sub_score = float(sorted_timeline[j].get("change_score", 0.0))
                if sub_score < (threshold * 0.6):
                    subsequent_elevated = False
                    break
            
            if subsequent_elevated:
                return sorted_timeline[i].get("date"), temporal_consistency, 0.85
            else:
                # Down-weighted candidate
                pass

    # Fallback to the date of maximum change if no monotonic step-function found
    max_idx = max(range(1, n), key=lambda idx: float(sorted_timeline[idx].get("change_score", 0.0)))
    return sorted_timeline[max_idx].get("date"), temporal_consistency, 0.40
