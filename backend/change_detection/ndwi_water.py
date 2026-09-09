"""
Normalized Difference Water Index (NDWI) Change Detection Engine
NDWI = (Green - NIR) / (Green + NIR)
Identifies water extent variations (flooding, river shifts, reservoir changes)
independently from structural/building changes.

NDWI is particularly sensitive to water bodies:
  - Open water: NDWI > 0.3
  - Vegetation: NDWI ≈ 0.4-0.8 (but distinguishable by NDVI)
  - Soil/urban: NDWI < 0.1
  - Snow/ice: NDWI ≈ 0.0

This module computes temporal NDWI changes to detect water extent dynamics.
"""
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class NDWIWaterDetector:
    """
    Normalized Difference Water Index (NDWI) detector for water extent changes.
    
    Computes NDWI = (Green - NIR) / (Green + NIR) and derives:
    - Water extent masks
    - Water-specific change detection
    - Delta NDWI (change in water body area)
    """
    
    def __init__(self, water_threshold: float = 0.3, vegetation_ndvi_threshold: float = 0.4):
        """
        Initialize NDWI detector.
        
        Args:
            water_threshold: NDWI threshold for classifying pixels as water (default 0.3)
            vegetation_ndvi_threshold: NDVI threshold to exclude dense vegetation (default 0.4)
        """
        self.water_threshold = water_threshold
        self.vegetation_ndvi_threshold = vegetation_ndvi_threshold
        logger.debug(f"NDWIWaterDetector initialized: water_threshold={water_threshold}")
    
    def compute_ndwi(
        self,
        green_band: np.ndarray,
        nir_band: np.ndarray
    ) -> np.ndarray:
        """
        Compute NDWI raster from Green and NIR bands.
        
        Formula: NDWI = (Green - NIR) / (Green + NIR)
        
        Args:
            green_band: Green band reflectance [H, W] in [0.0, 1.0]
            nir_band: NIR (B08) band reflectance [H, W] in [0.0, 1.0]
        
        Returns:
            NDWI map [H, W] in [-1.0, 1.0]
        """
        green = green_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        denom = green + nir
        ndwi = np.divide(
            (green - nir), denom,
            where=(denom > 1e-6),
            out=np.zeros_like(denom, dtype=np.float32)
        )
        
        return ndwi
    
    def compute_ndvi(
        self,
        red_band: np.ndarray,
        nir_band: np.ndarray
    ) -> np.ndarray:
        """
        Compute NDVI for separating vegetation from water.
        
        Formula: NDVI = (NIR - Red) / (NIR + Red)
        
        Args:
            red_band: Red band reflectance [H, W] in [0.0, 1.0]
            nir_band: NIR band reflectance [H, W] in [0.0, 1.0]
        
        Returns:
            NDVI map [H, W] in [-1.0, 1.0]
        """
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        denom = nir + red
        ndvi = np.divide(
            (nir - red), denom,
            where=(denom > 1e-6),
            out=np.zeros_like(denom, dtype=np.float32)
        )
        
        return ndvi
    
    def detect_water_mask(
        self,
        green_band: np.ndarray,
        nir_band: np.ndarray,
        red_band: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Detect water extent using NDWI with optional NDVI filtering to exclude vegetation.
        
        Args:
            green_band: Green band [H, W] in [0.0, 1.0]
            nir_band: NIR band [H, W] in [0.0, 1.0]
            red_band: Optional Red band for NDVI filtering [H, W]
        
        Returns:
            water_mask: Binary mask [H, W] uint8 (1 = water, 0 = non-water)
        """
        ndwi = self.compute_ndwi(green_band, nir_band)
        water_mask = (ndwi > self.water_threshold).astype(np.uint8)
        
        # Optionally filter out dense vegetation (high NDVI + high NDWI → likely mixed water+veg)
        if red_band is not None:
            ndvi = self.compute_ndvi(red_band, nir_band)
            dense_veg = (ndvi > self.vegetation_ndvi_threshold)
            water_mask[dense_veg] = 0
        
        return water_mask
    
    def compute_water_extent_change(
        self,
        green_before: np.ndarray,
        nir_before: np.ndarray,
        green_after: np.ndarray,
        nir_after: np.ndarray,
        red_before: Optional[np.ndarray] = None,
        red_after: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute water extent change between two dates.
        
        Detects water expansion/retreat by comparing water masks and NDWI deltas.
        
        Args:
            green_before, nir_before: T1 Green and NIR bands
            green_after, nir_after: T2 Green and NIR bands
            red_before, red_after: Optional Red bands for NDVI filtering
        
        Returns:
            Dictionary with:
                'water_mask_before': Binary water extent at T1
                'water_mask_after': Binary water extent at T2
                'water_delta': T2 - T1 water fraction change
                'ndwi_delta': Mean (T2 - T1) NDWI change
                'expansion_pixels': Count of pixels transitioned to water
                'retreat_pixels': Count of pixels transitioned from water
                'water_change_score': Normalized change metric [0.0, 1.0]
        """
        water_before = self.detect_water_mask(green_before, nir_before, red_before)
        water_after = self.detect_water_mask(green_after, nir_after, red_after)
        
        ndwi_before = self.compute_ndwi(green_before, nir_before)
        ndwi_after = self.compute_ndwi(green_after, nir_after)
        ndwi_delta = ndwi_after - ndwi_before
        
        # Compute change statistics
        water_before_frac = np.mean(water_before)
        water_after_frac = np.mean(water_after)
        water_delta = water_after_frac - water_before_frac
        
        # Count pixels that transitioned
        expansion = np.sum((water_after == 1) & (water_before == 0))
        retreat = np.sum((water_after == 0) & (water_before == 1))
        total_pixels = water_before.size
        
        # Normalize change score to [0, 1]
        max_change = max(expansion, retreat) / (total_pixels + 1e-6)
        water_change_score = min(1.0, abs(water_delta) * 5.0 + max_change)  # Amplify for visibility
        
        logger.debug(
            f"Water extent change: delta_frac={water_delta:.3f}, "
            f"expansion={expansion}, retreat={retreat}"
        )
        
        return {
            'water_mask_before': water_before,
            'water_mask_after': water_after,
            'water_delta': float(water_delta),
            'ndwi_delta': float(np.mean(ndwi_delta)),
            'expansion_pixels': int(expansion),
            'retreat_pixels': int(retreat),
            'water_change_score': float(water_change_score)
        }


# Global singleton instance
ndwi_detector = NDWIWaterDetector()
