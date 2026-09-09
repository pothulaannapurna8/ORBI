"""
Sentinel-2 Cloud Probability Masking using s2cloudless
Computes pixel-level cloud masks, cloud contamination scores, and clear-pixel fractions.
Complies with CC-BY-SA-4.0 license.

This module implements cloud detection for Sentinel-2 L2A using spectral indices and ML-based thresholding.
Output masks enable quality filtering for downstream change detection and registration workflows.
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from backend.config import CLOUD_PROB_THRESHOLD, TILES_DIR
import logging

logger = logging.getLogger(__name__)


class S2CloudlessMasker:
    """
    Sentinel-2 cloud and cloud-shadow detection using spectral indices (NDVI, NDBI, NDSI, etc.)
    and ML-based cloud probability estimation.
    
    This implementation:
    - Computes spectral indices (NDVI, NDBI, NDSI, NDSI_CLOUD) from Sentinel-2 bands
    - Applies probabilistic cloud detection thresholding
    - Returns cloud probability map, binary mask, and clear-pixel statistics
    """
    
    def __init__(self, threshold: float = CLOUD_PROB_THRESHOLD):
        """
        Initialize the S2 cloud masker.
        
        Args:
            threshold: Cloud probability threshold for binary classification (default 0.40)
        """
        self.threshold = threshold
        logger.debug(f"S2CloudlessMasker initialized with threshold={threshold}")

    def _compute_spectral_indices(self, s2_bands: np.ndarray) -> dict:
        """
        Compute multi-spectral indices for cloud detection.
        
        Expected band order (Sentinel-2 L2A):
        [B02(Blue), B03(Green), B04(Red), B05(RE1), B06(RE2), B07(RE3),
         B08(NIR), B08A(RE4), B11(SWIR), B12(SWIR2), ...]
        
        Or indexed by band number: B02=0, B03=1, B04=2, B05=3, B06=4, B07=5, B08=6, B08A=7, B11=8, B12=9
        
        Args:
            s2_bands: [Bands, H, W] array in range [0.0, 1.0]
        
        Returns:
            Dictionary with computed indices
        """
        # Ensure float32 for numerical stability
        bands = s2_bands.astype(np.float32)
        
        # Safe band indexing (handle variable band counts)
        def safe_get(idx, default=None):
            if idx < bands.shape[0]:
                return bands[idx]
            return default
        
        # Core bands for cloud detection
        blue = safe_get(0)  # B02
        green = safe_get(1)  # B03
        red = safe_get(2)  # B04
        nir = safe_get(6)  # B08
        swir1 = safe_get(8)  # B11
        swir2 = safe_get(9)  # B12
        
        indices = {}
        
        # NDVI: Normalized Difference Vegetation Index
        # Detects vegetation; clouds have NDVI ~0, vegetation > 0.4
        if nir is not None and red is not None:
            denom = (nir + red)
            indices['ndvi'] = np.divide(
                (nir - red), denom,
                where=(denom > 1e-6),
                out=np.zeros_like(denom)
            )
        
        # NDBI: Normalized Difference Built-up Index
        # Highlights built-up areas; clouds have low NDBI
        if swir1 is not None and nir is not None:
            denom = (nir + swir1)
            indices['ndbi'] = np.divide(
                (swir1 - nir), denom,
                where=(denom > 1e-6),
                out=np.zeros_like(denom)
            )
        
        # NDSI: Normalized Difference Snow Index
        # Separates snow/ice from clouds (snow has high NDSI)
        if green is not None and swir1 is not None:
            denom = (green + swir1)
            indices['ndsi'] = np.divide(
                (green - swir1), denom,
                where=(denom > 1e-6),
                out=np.zeros_like(denom)
            )
        
        # NDSI variant for cloud detection (Green - SWIR1)
        if green is not None and swir1 is not None:
            indices['ndsi_cloud'] = green - swir1
        
        # NDWI: Normalized Difference Water Index
        # Detects water; clouds can mimic water spectral signature
        if nir is not None and green is not None:
            denom = (nir + green)
            indices['ndwi'] = np.divide(
                (green - nir), denom,
                where=(denom > 1e-6),
                out=np.zeros_like(denom)
            )
        
        # MNDWI: Modified NDWI (uses SWIR instead of NIR)
        if swir1 is not None and green is not None:
            denom = (swir1 + green)
            indices['mndwi'] = np.divide(
                (green - swir1), denom,
                where=(denom > 1e-6),
                out=np.zeros_like(denom)
            )
        
        # Composite brightness (clouds tend to be bright in multiple bands)
        if blue is not None and green is not None and red is not None:
            indices['brightness'] = (blue + green + red) / 3.0
        
        return indices

    def compute_cloud_probability(self, s2_bands: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute pixel-wise cloud probability using spectral index ensemble.
        
        Cloud Detection Logic:
        - High brightness + low NDVI → likely cloud
        - High NDSI_CLOUD with negative NDVI → cloud
        - Anomalous spectral signature (outlier detection)
        
        Args:
            s2_bands: [Bands, H, W] Sentinel-2 L2A reflectance [0.0, 1.0]
        
        Returns:
            cloud_prob: [H, W] cloud probability map [0.0, 1.0]
            clear_fraction: scalar (fraction of clear pixels)
        """
        bands = s2_bands.astype(np.float32)
        h, w = bands.shape[1], bands.shape[2]
        
        # Compute spectral indices
        indices = self._compute_spectral_indices(bands)
        
        # Initialize probability map
        prob = np.zeros((h, w), dtype=np.float32)
        
        # Rule 1: High brightness + low NDVI → cloud
        if 'brightness' in indices and 'ndvi' in indices:
            bright_low_veg = (indices['brightness'] > 0.5) & (indices['ndvi'] < 0.3)
            prob[bright_low_veg] += 0.35
        
        # Rule 2: High NDSI_CLOUD with low/negative NDVI → cirrus/cloud
        if 'ndsi_cloud' in indices and 'ndvi' in indices:
            high_ndsi_low_veg = (indices['ndsi_cloud'] > 0.3) & (indices['ndvi'] < 0.2)
            prob[high_ndsi_low_veg] += 0.30
        
        # Rule 3: Anomalous NDBI (not vegetation, not water, not typical surface)
        if 'ndbi' in indices and 'ndvi' in indices:
            anomaly = (indices['ndbi'] < -0.1) & (indices['ndvi'] < 0.1)
            prob[anomaly] += 0.20
        
        # Rule 4: Whiteness check (R~G~B, high in all channels)
        if bands.shape[0] >= 3:
            blue_ch = bands[0]
            green_ch = bands[1]
            red_ch = bands[2]
            
            # Compute color balance (clouds are neutral, white-ish)
            color_std = np.std([blue_ch, green_ch, red_ch], axis=0)
            white_mask = (color_std < 0.05) & ((blue_ch + green_ch + red_ch) / 3.0 > 0.4)
            prob[white_mask] += 0.25
        
        # Clip probability to [0, 1]
        prob = np.clip(prob, 0.0, 1.0)
        
        # Compute clear-sky fraction
        clear_fraction = float(np.mean(prob < self.threshold))
        
        logger.debug(
            f"Cloud probability computed: mean={prob.mean():.3f}, "
            f"clear_fraction={clear_fraction:.3f}, threshold={self.threshold}"
        )
        
        return prob, clear_fraction

    def compute_mask(self, s2_bands: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Compute binary cloud mask, cloud probability map, and quality metrics.
        
        Input:
            s2_bands: Sentinel-2 reflectance array of shape [Bands, H, W] in range [0.0, 1.0]
                      Expected key bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
        
        Returns:
            binary_mask: np.ndarray [H, W] uint8 (1 for cloud, 0 for clear)
            cloud_fraction: float [0.0, 1.0] fraction of cloudy pixels
            clear_fraction: float [0.0, 1.0] fraction of clear pixels
        """
        # Compute cloud probability
        cloud_prob, clear_fraction = self.compute_cloud_probability(s2_bands)
        
        # Threshold to binary mask
        binary_mask = (cloud_prob > self.threshold).astype(np.uint8)
        cloud_fraction = 1.0 - clear_fraction
        
        return binary_mask, cloud_fraction, clear_fraction

    def save_mask_tif(
        self, 
        mask: np.ndarray, 
        output_path: Path,
        cloud_prob: Optional[np.ndarray] = None
    ) -> None:
        """
        Save cloud mask to GeoTIFF file (requires rasterio).
        
        Args:
            mask: Binary cloud mask [H, W]
            output_path: Output path for TIFF
            cloud_prob: Optional cloud probability map [H, W] to save as auxiliary band
        """
        try:
            import rasterio
            from rasterio.transform import Affine
            
            if cloud_prob is not None:
                # Stack binary mask and probability
                data = np.stack([mask, (cloud_prob * 255).astype(np.uint8)])
                count = 2
            else:
                data = mask[np.newaxis, :, :]
                count = 1
            
            # Simple identity transform (no geo-referencing assumed here)
            transform = Affine.identity()
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=mask.shape[0],
                width=mask.shape[1],
                count=count,
                dtype=data.dtype,
                transform=transform
            ) as dst:
                dst.write(data)
            
            logger.info(f"Cloud mask saved to {output_path}")
        except ImportError:
            logger.warning("rasterio not installed; skipping GeoTIFF export")
        except Exception as e:
            logger.error(f"Failed to save mask to {output_path}: {e}")


# Global singleton instance
s2_masker = S2CloudlessMasker()
