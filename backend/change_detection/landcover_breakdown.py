"""Land-cover percentages derived from aligned Sentinel-2 source bands."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from backend.quality.arosics_register import arosics_aligner

try:
    import rasterio
except ImportError:
    rasterio = None


def _normalize_bands(bands: np.ndarray) -> np.ndarray:
    values = bands.astype(np.float32)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if max_value > 255.0:
        values /= 10000.0
    elif max_value > 1.0:
        values /= 255.0
    return np.clip(values, 0.0, 1.0)


def load_sentinel_bands(filepath: str) -> np.ndarray:
    """Load a staged Sentinel-2 scene in its stored band order."""
    if rasterio is None or not filepath or not Path(filepath).exists():
        raise FileNotFoundError(f"Sentinel-2 source is unavailable: {filepath}")
    with rasterio.open(filepath) as dataset:
        bands = dataset.read()
    if bands.shape[0] < 8:
        raise ValueError(f"Expected B04/B08 bands in {filepath}, found {bands.shape[0]} bands")
    return _normalize_bands(bands)


def _resize_to_shape(values: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if values.shape[-2:] == shape:
        return values
    y_indices = np.linspace(0, values.shape[-2] - 1, shape[0]).round().astype(int)
    x_indices = np.linspace(0, values.shape[-1] - 1, shape[1]).round().astype(int)
    return values[..., y_indices[:, None], x_indices]


def compute_landcover_breakdown(
    before_filepath: str,
    after_filepath: str,
    change_mask: Optional[np.ndarray] = None,
    water_threshold: float = 0.1,
    vegetation_threshold: float = 0.3,
) -> Dict[str, float]:
    """Compute water, vegetation, and built-up percentages for two aligned scenes."""
    before = load_sentinel_bands(before_filepath)
    after = load_sentinel_bands(after_filepath)
    common_shape = (
        min(before.shape[1], after.shape[1]),
        min(before.shape[2], after.shape[2]),
    )
    before = before[:, :common_shape[0], :common_shape[1]]
    after = after[:, :common_shape[0], :common_shape[1]]

    aligned_after, _, _ = arosics_aligner.register(before, after)
    green_before, red_before, nir_before = before[1], before[2], before[6]
    green_after, red_after, nir_after = aligned_after[1], aligned_after[2], aligned_after[6]

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return numerator / np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)

    water_before = ratio(green_before - nir_before, green_before + nir_before) > water_threshold
    water_after = ratio(green_after - nir_after, green_after + nir_after) > water_threshold
    vegetation_before = ratio(nir_before - red_before, nir_before + red_before) > vegetation_threshold
    vegetation_after = ratio(nir_after - red_after, nir_after + red_after) > vegetation_threshold

    excluded_before = water_before | vegetation_before
    excluded_after = water_after | vegetation_after
    builtup_before = ~excluded_before
    builtup_after = ~excluded_after

    if change_mask is not None:
        changed = np.asarray(change_mask) > 0
        changed = _resize_to_shape(changed, common_shape)
        eligible_change = changed & ~(
            water_before | water_after | vegetation_before | vegetation_after
        )
        builtup_after = builtup_after | eligible_change

    def percent(mask: np.ndarray) -> float:
        return round(float(np.mean(mask) * 100.0), 2)

    water_before_pct = percent(water_before)
    water_after_pct = percent(water_after)
    vegetation_before_pct = percent(vegetation_before)
    vegetation_after_pct = percent(vegetation_after)
    builtup_before_pct = percent(builtup_before)
    builtup_after_pct = percent(builtup_after)

    return {
        "water_pct_before": water_before_pct,
        "water_pct_after": water_after_pct,
        "water_pct_change": round(water_after_pct - water_before_pct, 2),
        "vegetation_pct_before": vegetation_before_pct,
        "vegetation_pct_after": vegetation_after_pct,
        "vegetation_pct_change": round(vegetation_after_pct - vegetation_before_pct, 2),
        "builtup_pct_before": builtup_before_pct,
        "builtup_pct_after": builtup_after_pct,
        "builtup_pct_change": round(builtup_after_pct - builtup_before_pct, 2),
    }