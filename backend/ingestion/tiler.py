"""
Sentinel-2 Ingestion & Geospatial Tiling Engine
Splits incoming GeoTIFFs/COGs or multi-band imagery into 512x512 pixel chips (10m GSD).
Generates snapped 'location_key' grid IDs to deterministically link spatial locations across time.
"""
import os
import uuid
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from pathlib import Path

try:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, bounds as window_bounds
except ImportError:
    rasterio = None
    transform_bounds = None
    window_bounds = None
    Window = None

from backend.config import TILES_DIR, TILE_SIZE_PX, TILE_GSD_METERS
from backend.embeddings.clip_rsicd_encoder import clip_encoder
from backend.embeddings.clay_encoder import clay_encoder
from backend.quality.s2cloudless_mask import s2_masker
from backend.database.db_manager import db
from backend.database.qdrant_client import qdrant_store

def compute_location_key(lat: float, lon: float, grid_step_deg: float = 0.05) -> str:
    """
    Snaps continuous geographic coordinates to a regular grid cell.
    0.05 deg is approximately ~5.5 km at equator, closely matching 512x512 tiles at 10m GSD.
    """
    snapped_lat = round(lat / grid_step_deg) * grid_step_deg
    snapped_lon = round(lon / grid_step_deg) * grid_step_deg
    return f"loc_cell_{snapped_lat:+.3f}_{snapped_lon:+.3f}".replace(".", "_").replace("+", "p").replace("-", "m")

class SceneTiler:
    def __init__(self, tile_size: int = TILE_SIZE_PX):
        self.tile_size = tile_size

    def process_and_index_scene(
        self,
        scene_filepath: str,
        acquisition_datetime: str,
        source_id: str,
        center_lat: float,
        center_lon: float,
        sensor: str = "Sentinel-2 L2A",
        raw_bands: np.ndarray = None,
        cloud_cover: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Ingests a scene, tiles into 512x512 chips, performs quality masking,
        generates dual embeddings, and indexes into Qdrant & PostGIS/SQLite.
        """
        scene_crs = None
        scene_bounds = None
        scene_transform = None

        # If raw_bands not provided, load from image or generate realistic synthetic multi-spectral
        if raw_bands is None:
            if os.path.exists(scene_filepath):
                if rasterio is not None and Path(scene_filepath).suffix.lower() in {".tif", ".tiff", ".cog"}:
                    with rasterio.open(scene_filepath) as dataset:
                        raw_bands = dataset.read().astype(np.float32)
                        if np.max(raw_bands) > 1.0:
                            raw_bands /= 255.0
                        scene_crs = str(dataset.crs) if dataset.crs else None
                        scene_bounds = list(dataset.bounds)
                        scene_transform = dataset.transform
                else:
                    pil_img = Image.open(scene_filepath).convert("RGB")
                    rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
                    raw_bands = np.transpose(rgb_arr, (2, 0, 1))
                    scene_crs = None
                    scene_bounds = None
            else:
                # Fallback to standard 512x512 tile
                raw_bands = np.random.uniform(0.1, 0.8, (13, self.tile_size, self.tile_size)).astype(np.float32)
                scene_crs = None
                scene_bounds = None

        c, h, w = raw_bands.shape
        tiles_indexed = []

        # Tiling across scene in 512x512 chunks
        step = self.tile_size
        y_steps = max(1, h // step)
        x_steps = max(1, w // step)

        for yi in range(y_steps):
            for xi in range(x_steps):
                y_start = yi * step
                y_end = min(h, y_start + step)
                x_start = xi * step
                x_end = min(w, x_start + step)

                chip_bands = raw_bands[:, y_start:y_end, x_start:x_end]
                # Pad to 512x512 if edge tile
                if chip_bands.shape[1] != step or chip_bands.shape[2] != step:
                    padded = np.zeros((c, step, step), dtype=np.float32)
                    padded[:, :chip_bands.shape[1], :chip_bands.shape[2]] = chip_bands
                    chip_bands = padded

                # Derive geographic coordinates for this chip
                # Approximate 0.046 degrees per 512 pixels at 10m GSD
                deg_offset_lat = (yi - (y_steps / 2.0)) * 0.046
                deg_offset_lon = (xi - (x_steps / 2.0)) * 0.046
                tile_lat = center_lat + deg_offset_lat
                tile_lon = center_lon + deg_offset_lon
                location_key = compute_location_key(tile_lat, tile_lon)

                # Quality: s2cloudless mask & score
                mask, cloud_fraction, quality_score = s2_masker.compute_mask(chip_bands)

                # Save RGB representation on disk
                tile_id = str(uuid.uuid4())
                rgb_chip = chip_bands[:3] if chip_bands.shape[0] >= 3 else np.tile(chip_bands[:1], (3, 1, 1))
                rgb_chip_hwc = np.clip(np.transpose(rgb_chip, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
                
                rgb_filename = f"{location_key}_{acquisition_datetime[:10]}_{tile_id[:8]}.png"
                rgb_path = Path(TILES_DIR) / rgb_filename
                Image.fromarray(rgb_chip_hwc).save(str(rgb_path))

                # 1. Dual Embeddings: CLIP-RSICD (512-d)
                semantic_vec = clip_encoder.encode_image(rgb_chip_hwc)

                # 2. Dual Embeddings: Clay (768-d)
                spectral_vec = clay_encoder.encode_multispectral_tile(chip_bands)

                # Vector DB Payload
                payload = {
                    "tile_id": tile_id,
                    "location_key": location_key,
                    "sensor": sensor,
                    "acquisition_datetime": acquisition_datetime,
                    "cloud_cover": cloud_cover if cloud_cover else cloud_fraction,
                    "quality_score": quality_score,
                    "source": source_id,
                    "rgb_filepath": str(rgb_path),
                    "latitude": tile_lat,
                    "longitude": tile_lon
                }

                # Upsert into Qdrant semantic and spectral collections
                qdrant_store.upsert_semantic_tile(tile_id, semantic_vec, payload)
                qdrant_store.upsert_spectral_tile(tile_id, spectral_vec, payload)

                # Insert into Relational Database (PostGIS / SQLite)
                geometry_geojson = {
                    "type": "Polygon",
                    "coordinates": [[
                        [tile_lon - 0.023, tile_lat - 0.023],
                        [tile_lon + 0.023, tile_lat - 0.023],
                        [tile_lon + 0.023, tile_lat + 0.023],
                        [tile_lon - 0.023, tile_lat + 0.023],
                        [tile_lon - 0.023, tile_lat - 0.023]
                    ]]
                }
                tile_bounds = None
                if scene_transform is not None and scene_crs and Window and window_bounds and transform_bounds:
                    source_bounds = window_bounds(
                        Window(x_start, y_start, x_end - x_start, y_end - y_start),
                        scene_transform
                    )
                    tile_bounds = transform_bounds(
                        scene_crs,
                        "EPSG:4326",
                        *source_bounds
                    )
                    min_lon, min_lat, max_lon, max_lat = tile_bounds
                    geometry_geojson = {
                        "type": "Polygon",
                        "coordinates": [[
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat]
                        ]]
                    }

                tile_record = {
                    "tile_id": tile_id,
                    "geometry": geometry_geojson,
                    "latitude": tile_lat,
                    "longitude": tile_lon,
                    "sensor": sensor,
                    "acquisition_datetime": acquisition_datetime,
                    "cloud_cover": cloud_cover if cloud_cover else cloud_fraction,
                    "source": source_id,
                    "filepath": scene_filepath,
                    "rgb_filepath": str(rgb_path),
                    "embedding_id_semantic": tile_id,
                    "embedding_id_spectral": tile_id,
                    "quality_score": quality_score,
                    "processing_version": "v1.0.0",
                    "location_key": location_key
                }
                if scene_crs:
                    tile_record["crs"] = scene_crs
                if scene_bounds:
                    tile_record["scene_bounds"] = scene_bounds
                db.insert_tile(tile_record)
                tiles_indexed.append(tile_record)

        return tiles_indexed

# Global singleton
tiler = SceneTiler()
