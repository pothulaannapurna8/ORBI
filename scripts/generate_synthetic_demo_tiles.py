"""
Demo Data Generator: Realistic Multi-Temporal Sentinel-2 Scenes
Creates multi-date paired scenes for 2 AOIs:
1. 'AOI_URBAN_GROWTH' (Centroid: 12.9716 N, 77.5946 E - Bangalore)
   Dates: 2024-03-01 (Baseline), 2024-09-15, 2025-05-10 (Construction onset), 2025-11-20 (Complete structures)
2. 'AOI_RIVER_BASIN' (Centroid: 13.0827 N, 80.2707 E - River/Coast)
   Dates: 2024-03-01 (Dry season), 2024-10-20 (Monsoon water expansion), 2025-04-15 (Receding)

Simulates 13-band Sentinel-2 reflectance (B01-B12) with realistic spectral indices:
- NIR (B08) high on vegetation, low on water
- Red (B04) / Green (B03) / Blue (B02) for true color RGB
- High brightness on newly built roofs in 2025
- Synthetic cloud contamination on 2024-09-15 to validate false-alarm suppression!
"""
import os
import sys
import uuid
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import RAW_DIR
from backend.ingestion.tiler import tiler

def create_synthetic_s2_scene(
    name: str,
    date_str: str,
    aoi_type: str,
    has_cloud: bool = False,
    has_new_buildings: bool = False,
    has_water_expansion: bool = False
) -> tuple[str, np.ndarray]:
    """
    Generates a 1024x1024 Sentinel-2 13-band scene [13, 1024, 1024].
    Yields 4 tiles of 512x512.
    """
    H, W = 1024, 1024
    # Bands: 0:B1, 1:B2(Blue), 2:B3(Green), 3:B4(Red), 4:B5, 5:B6, 6:B7, 7:B8(NIR), 8:B8A, 9:B9, 10:B10(Cirrus), 11:B11(SWIR1), 12:B12(SWIR2)
    bands = np.zeros((13, H, W), dtype=np.float32)

    # Base background: Soil & sparse vegetation
    bands[1] = 0.15 + np.random.normal(0, 0.02, (H, W)) # Blue
    bands[2] = 0.22 + np.random.normal(0, 0.02, (H, W)) # Green
    bands[3] = 0.20 + np.random.normal(0, 0.02, (H, W)) # Red
    bands[7] = 0.45 + np.random.normal(0, 0.03, (H, W)) # NIR (Vegetation)

    if aoi_type == "river":
        # Draw a winding river channel through center
        for x in range(W):
            river_y = int(512 + 150 * np.sin(x / 120.0))
            width = 70 if not has_water_expansion else 140
            y_low = max(0, river_y - width)
            y_high = min(H, river_y + width)
            # Water signature: very low NIR, moderate Blue/Green
            bands[1, y_low:y_high, x] = 0.25 # Blue
            bands[2, y_low:y_high, x] = 0.35 # Green
            bands[3, y_low:y_high, x] = 0.10 # Red
            bands[7, y_low:y_high, x] = 0.03 # NIR

    elif aoi_type == "urban":
        # Baseline roads and small settlement
        bands[1, 400:650, 400:650] = 0.28
        bands[2, 400:650, 400:650] = 0.30
        bands[3, 400:650, 400:650] = 0.32
        bands[7, 400:650, 400:650] = 0.18

        # Introduce high-contrast newly constructed buildings in 2025
        if has_new_buildings:
            for by in [200, 280, 360, 700, 780]:
                for bx in [200, 280, 360, 700, 780]:
                    # High reflectance concrete / metal rooftop signature
                    bands[1, by:by+50, bx:bx+50] = 0.65
                    bands[2, by:by+50, bx:bx+50] = 0.70
                    bands[3, by:by+50, bx:bx+50] = 0.72
                    bands[7, by:by+50, bx:bx+50] = 0.55

    # Simulated cloud contamination on single date
    if has_cloud:
        # Puffy high reflectance cloud patch in top-right quadrant
        cy, cx = 250, 800
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        cloud_mask = dist < 180
        bands[1, cloud_mask] += 0.55 # Blue scatter
        bands[3, cloud_mask] += 0.50
        bands[10, cloud_mask] += 0.60 # Cirrus
        bands[11, cloud_mask] += 0.40 # SWIR

    # Fill remaining auxiliary bands
    bands[0] = bands[1] * 0.9 # Aerosol
    bands[4] = (bands[3] + bands[7]) / 2.0
    bands[5] = bands[7] * 0.95
    bands[6] = bands[7] * 0.98
    bands[8] = bands[7] * 1.02
    bands[9] = bands[1] * 0.5 # Water vapor
    bands[11] = bands[3] * 0.8
    bands[12] = bands[3] * 0.6

    bands = np.clip(bands, 0.0, 1.0)

    # Save RGB preview
    rgb_arr = np.clip(np.stack([bands[3], bands[2], bands[1]], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    filename = f"{name}_{date_str}.png"
    filepath = Path(RAW_DIR) / filename
    Image.fromarray(rgb_arr).save(str(filepath))

    return str(filepath), bands

def populate_demo_dataset():
    print("Generating synthetic Sentinel-2 multi-temporal scenes...")
    scenes_meta = [
        # AOI 1: Urban Growth (Centroid: 12.9716, 77.5946)
        ("AOI_URBAN", "2024-03-01", "urban", False, False, False, 12.9716, 77.5946),
        ("AOI_URBAN", "2024-09-15", "urban", True, False, False, 12.9716, 77.5946),  # Cloud contamination date
        ("AOI_URBAN", "2025-05-10", "urban", False, True, False, 12.9716, 77.5946),  # Construction visible
        ("AOI_URBAN", "2025-11-20", "urban", False, True, False, 12.9716, 77.5946),  # Fully built structures

        # AOI 2: River Basin (Centroid: 13.0827, 80.2707)
        ("AOI_RIVER", "2024-03-01", "river", False, False, False, 13.0827, 80.2707),
        ("AOI_RIVER", "2024-10-20", "river", False, False, True, 13.0827, 80.2707),   # Monsoon flood / expansion
        ("AOI_RIVER", "2025-04-15", "river", False, False, False, 13.0827, 80.2707)
    ]

    total_tiles = 0
    for name, date_str, aoi_type, cloud, new_bldg, water_exp, lat, lon in scenes_meta:
        path, bands = create_synthetic_s2_scene(name, date_str, aoi_type, cloud, new_bldg, water_exp)
        tiles = tiler.process_and_index_scene(
            scene_filepath=path,
            acquisition_datetime=f"{date_str}T10:00:00Z",
            source_id=f"{name}_{date_str}",
            center_lat=lat,
            center_lon=lon,
            sensor="Sentinel-2 L2A",
            raw_bands=bands
        )
        total_tiles += len(tiles)
        print(f"  Indexed {name} ({date_str}): {len(tiles)} tiles")

    print(f"\nCompleted! Total tiles indexed across multi-temporal archive: {total_tiles}")

if __name__ == "__main__":
    populate_demo_dataset()
