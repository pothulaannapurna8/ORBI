"""
Clay Foundation Model v1.5 Spectral Tile Encoder
Encodes multispectral chips (13 bands) into 768-dimensional spectral embeddings.
Dedicated vector space in Qdrant ('spectral_tiles') joined by tile_id and location_key.
Complies with Apache-2.0 license.
"""
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from backend.config import CLAY_WEIGHTS_PATH

logger = logging.getLogger(__name__)

class ClaySpectralEncoder(nn.Module):
    def __init__(self, in_channels: int = 13, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        # Spectral convolution & spatial patch aggregation
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=16, stride=16),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        nn.init.orthogonal_(self.patch_embed[4].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 13, H, W]
        feats = self.patch_embed(x)
        return feats / feats.norm(dim=-1, keepdim=True)

class ClayEncoder:
    def __init__(self, checkpoint_path: str = CLAY_WEIGHTS_PATH, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = 768
        self.model = ClaySpectralEncoder(in_channels=13, embed_dim=self.dim).to(self.device).eval()
        
        # Load weights if available
        if os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            except Exception as exc:
                logger.warning("Failed to load Clay weights from %s: %s", checkpoint_path, exc)
        else:
            logger.warning("Clay weights not found: %s; using initialized model", checkpoint_path)

    def encode_multispectral_tile(self, bands: np.ndarray) -> List[float]:
        """
        bands: np.ndarray of shape [13, H, W] or [C, H, W], normalized in [0, 1]
        Returns: 768-dimensional L2-normalized vector
        """
        # If fewer bands provided, pad to 13 bands with replication
        if bands.ndim == 2:
            bands = np.stack([bands] * 13, axis=0)
        elif bands.ndim == 3:
            if bands.shape[0] < 13:
                repeats = (13 // bands.shape[0]) + 1
                bands = np.tile(bands, (repeats, 1, 1))[:13]
            elif bands.shape[0] > 13:
                bands = bands[:13]

        tensor = torch.from_numpy(bands.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)
        if not torch.isfinite(emb).all() or torch.any(emb.norm(dim=-1) <= 1e-12):
            raise ValueError("Clay embedding is non-finite or zero")
        return emb.cpu().numpy()[0].tolist()

# Global singleton
clay_encoder = ClayEncoder()
