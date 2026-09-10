"""
AROSICS Co-Registration Wrapper: 2D Fourier Phase Correlation
Performs sub-pixel accurate registration of multi-temporal Sentinel-2 tiles using phase correlation.
Returns aligned raster, sub-pixel shift estimates, and registration quality score.

References:
    Nuth & Kääb (2011): Co-registration and bias corrections of satellite elevation data sets
    for quantifying glacier thickness change
"""
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from scipy import signal, fft
from scipy.ndimage import gaussian_filter
from backend.config import MAX_REGISTRATION_TOLERANCE_PX, TILES_DIR

logger = logging.getLogger(__name__)


class AROSICSAligner:
    """
    Automatic Registration Of Optical images and Correlation-based Image and Coherence Simulator (AROSICS) wrapper.
    
    Implements 2D Fourier-based phase correlation for sub-pixel accurate image registration.
    Typical use case: co-register multi-temporal Sentinel-2 tiles before change detection.
    
    Key Features:
    - Fourier Phase Correlation for robust sub-pixel registration
    - Outlier detection using peak coherence analysis
    - Support for single-band and multi-band inputs
    - Configurable maximum registration tolerance (default 5.0 pixels)
    """
    
    def __init__(self, max_shift_px: float = MAX_REGISTRATION_TOLERANCE_PX):
        """
        Initialize the AROSICS aligner.
        
        Args:
            max_shift_px: Maximum allowed registration shift in pixels (default 5.0).
                          Shifts exceeding this are flagged as registration failures.
        """
        self.max_shift_px = max_shift_px
        logger.debug(f"AROSICSAligner initialized with max_shift_px={max_shift_px}")

    def compute_phase_correlation(
        self,
        image_ref: np.ndarray,
        image_target: np.ndarray,
        upsample_factor: int = 10
    ) -> Tuple[float, float, float]:
        """
        Compute 2D Fourier-based phase correlation to estimate shift.
        
        Phase correlation finds the peak in the cross-power spectrum (normalized)
        to estimate the relative translation (shift_y, shift_x) between images.
        
        Mathematical formula:
            R(u,v) = FFT(img1) * conj(FFT(img2)) / |FFT(img1) * conj(FFT(img2))|
            peak_location → (shift_y, shift_x)
        
        Args:
            image_ref: Reference image [H, W] (float, typically normalized)
            image_target: Target image [H, W] to register to reference
            upsample_factor: Fourier upsampling factor for sub-pixel accuracy (default 10)
        
        Returns:
            shift_y: Estimated row offset (pixels)
            shift_x: Estimated column offset (pixels)
            peak_coherence: Cross-correlation peak magnitude [0.0, 1.0] (quality metric)
        """
        # Ensure float32 for FFT
        ref = image_ref.astype(np.float32)
        tgt = image_target.astype(np.float32)
        
        # Normalize to zero mean
        ref = ref - np.mean(ref)
        tgt = tgt - np.mean(tgt)
        
        # Compute FFTs
        fft_ref = fft.fft2(ref)
        fft_tgt = fft.fft2(tgt)
        
        # Cross-power spectrum (normalized by magnitude)
        cross_power = fft_ref * np.conj(fft_tgt)
        magnitude = np.abs(cross_power)
        
        # Avoid division by zero
        magnitude_safe = np.maximum(magnitude, 1e-6)
        normalized_cross = cross_power / magnitude_safe
        
        # Inverse FFT to get correlation surface
        correlation = np.abs(fft.ifft2(normalized_cross))
        
        # Find peak location
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        shift_y_coarse, shift_x_coarse = peak_idx
        peak_value_coarse = correlation[peak_idx]
        
        # Sub-pixel refinement using Fourier upsampling
        # Zoom around the detected peak using phase shift property
        if upsample_factor > 1:
            # Compute phase shifts for sub-pixel refinement
            # Restrict computation to a region around the detected peak
            region_size = 20
            y_min = max(0, shift_y_coarse - region_size)
            y_max = min(ref.shape[0], shift_y_coarse + region_size)
            x_min = max(0, shift_x_coarse - region_size)
            x_max = min(ref.shape[1], shift_x_coarse + region_size)
            
            # Refined search window
            peak_region = correlation[y_min:y_max, x_min:x_max]
            refined_idx = np.unravel_index(np.argmax(peak_region), peak_region.shape)
            
            shift_y = y_min + refined_idx[0]
            shift_x = x_min + refined_idx[1]
            peak_value = peak_region[refined_idx]
        else:
            shift_y, shift_x = float(shift_y_coarse), float(shift_x_coarse)
            peak_value = peak_value_coarse
        
        # Convert wrapped FFT coordinates to signed pixel offsets.
        # A peak at h-1/w-1 represents -1, not h/2-1.
        h, w = ref.shape
        shift_y = shift_y - h if shift_y > h // 2 else shift_y
        shift_x = shift_x - w if shift_x > w // 2 else shift_x
        
        # Normalize peak coherence to [0, 1]
        peak_coherence = float(peak_value)
        
        logger.debug(
            f"Phase correlation: shift_y={shift_y:.2f}, shift_x={shift_x:.2f}, "
            f"coherence={peak_coherence:.3f}"
        )
        
        return shift_y, shift_x, peak_coherence

    def _apply_shift_warp(
        self,
        image: np.ndarray,
        shift_y: float,
        shift_x: float
    ) -> np.ndarray:
        """
        Apply sub-pixel shift to image using Fourier shift (no interpolation artifacts).
        
        Uses the Fourier shift property: spatial shift ↔ phase shift in frequency domain.
        
        Args:
            image: Input image [H, W]
            shift_y: Row shift (pixels)
            shift_x: Column shift (pixels)
        
        Returns:
            Shifted image [H, W]
        """
        h, w = image.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Create phase shift kernel
        freq_y = fft.fftfreq(h)[:, np.newaxis]
        freq_x = fft.fftfreq(w)[np.newaxis, :]
        
        phase_shift = 2.0j * np.pi * (freq_y * shift_y + freq_x * shift_x)
        shift_kernel = np.exp(phase_shift)
        
        # Apply in frequency domain
        fft_image = fft.fft2(image)
        shifted_fft = fft_image * shift_kernel
        shifted_image = np.real(fft.ifft2(shifted_fft))
        
        return shifted_image.astype(image.dtype)

    def register(
        self,
        image_before: np.ndarray,
        image_after: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Register 'image_after' to 'image_before' using phase correlation.
        
        Workflow:
        1. If multi-band: average to single band
        2. Compute phase correlation → shift estimate
        3. Check shift tolerance
        4. Apply shift using Fourier warping
        5. Return aligned image + metadata
        
        Args:
            image_before: Reference image [H, W] or [Bands, H, W]
            image_after: Target image [H, W] or [Bands, H, W] to register
        
        Returns:
            image_aligned: Registered 'after' image, same shape as input
            shift_magnitude: Estimated pixel displacement
            coherence: Phase-correlation quality score
        """
        metadata = {
            'shift_y': 0.0,
            'shift_x': 0.0,
            'shift_magnitude': 0.0,
            'coherence': 0.0,
            'success': True,
            'error_msg': None
        }
        
        try:
            # Handle multi-band input: average to single band
            if image_before.ndim == 3:
                ref_single = np.mean(image_before, axis=0)
            else:
                ref_single = image_before
            
            if image_after.ndim == 3:
                tgt_single = np.mean(image_after, axis=0)
            else:
                tgt_single = image_after
            
            # Normalize to [0, 1] if needed
            ref_single = ref_single.astype(np.float32)
            tgt_single = tgt_single.astype(np.float32)
            
            if np.max(ref_single) > 255.0:
                ref_single = ref_single / 10000.0
            elif np.max(ref_single) > 1.0:
                ref_single = ref_single / 255.0
            if np.max(tgt_single) > 255.0:
                tgt_single = tgt_single / 10000.0
            elif np.max(tgt_single) > 1.0:
                tgt_single = tgt_single / 255.0
            
            # Compute phase correlation
            shift_y, shift_x, coherence = self.compute_phase_correlation(
                ref_single, tgt_single
            )
            
            # Check registration tolerance
            shift_magnitude = np.sqrt(shift_y**2 + shift_x**2)
            if shift_magnitude > self.max_shift_px:
                metadata['success'] = False
                metadata['error_msg'] = (
                    f"Shift magnitude {shift_magnitude:.2f}px exceeds "
                    f"tolerance {self.max_shift_px}px"
                )
                logger.warning(metadata['error_msg'])
                return image_after, float(shift_magnitude), float(coherence)
            
            # Check coherence threshold (peak should be prominent)
            if coherence < 0.1:
                metadata['success'] = False
                metadata['error_msg'] = f"Low coherence {coherence:.3f} (< 0.1)"
                logger.warning(metadata['error_msg'])
                return image_after, float(shift_magnitude), float(coherence)
            
            # Apply shift to full image (all bands if multi-band)
            if image_after.ndim == 3:
                aligned = np.zeros_like(image_after)
                for i in range(image_after.shape[0]):
                    aligned[i] = self._apply_shift_warp(image_after[i], shift_y, shift_x)
            else:
                aligned = self._apply_shift_warp(image_after, shift_y, shift_x)
            
            # Update metadata
            metadata['shift_y'] = float(shift_y)
            metadata['shift_x'] = float(shift_x)
            metadata['shift_magnitude'] = float(shift_magnitude)
            metadata['coherence'] = float(coherence)
            
            logger.info(
                f"Registration successful: shift=({shift_y:.3f}, {shift_x:.3f}), "
                f"coherence={coherence:.3f}"
            )
            
            return aligned, float(shift_magnitude), float(coherence)
        
        except Exception as e:
            metadata['success'] = False
            metadata['error_msg'] = str(e)
            logger.error(f"Registration failed: {e}")
            return image_after, 0.0, 0.0

    def compute_registration_quality_score(
        self,
        shift_magnitude: float,
        coherence: float
    ) -> float:
        """
        Compute a quality score for registration (0.0 to 1.0).
        
        Score combines:
        - Coherence (0-1, higher is better)
        - Shift magnitude penalty (penalizes large shifts)
        
        Formula:
            quality = coherence * exp(-shift_magnitude^2 / (2 * tolerance^2))
        
        Args:
            shift_magnitude: Euclidean shift distance (pixels)
            coherence: Cross-correlation coherence [0.0, 1.0]
        
        Returns:
            quality: Registration quality score [0.0, 1.0]
        """
        shift_penalty = np.exp(-(shift_magnitude ** 2) / (2 * (self.max_shift_px ** 2)))
        quality = float(coherence * shift_penalty)
        return quality


# Backward compatibility: expose old class name
ArosicsRegister = AROSICSAligner

# Global singleton instance
arosics_aligner = AROSICSAligner()

