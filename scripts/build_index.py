"""
Index Builder Script
Rebuilds or initializes Qdrant vector collections and PostGIS/SQLite metadata
from all staged Sentinel-2 scene files in data/raw.
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import RAW_DIR
from backend.ingestion.tiler import tiler
from backend.database.qdrant_client import qdrant_store
from backend.database.db_manager import db

def rebuild_full_index():
    print("Initializing Qdrant collections...")
    qdrant_store._ensure_collections()

    raw_files = list(Path(RAW_DIR).glob("*.png")) + list(Path(RAW_DIR).glob("*.tif"))
    print(f"Found {len(raw_files)} raw scenes to index.")

    total_indexed = 0
    for f in raw_files:
        name = f.stem
        # Extract timestamp if available
        parts = name.split("_")
        date_str = parts[1] if len(parts) > 1 and len(parts[1]) == 10 else "2024-01-01"
        lat = 12.9716 if "URBAN" in name else 13.0827
        lon = 77.5946 if "URBAN" in name else 80.2707

        tiles = tiler.process_and_index_scene(
            scene_filepath=str(f),
            acquisition_datetime=f"{date_str}T10:00:00Z",
            source_id=name,
            center_lat=lat,
            center_lon=lon
        )
        total_indexed += len(tiles)
        print(f"  Processed {name} -> {len(tiles)} chips")

    print(f"\nIndex rebuild complete. Total indexed tiles: {total_indexed}")

if __name__ == "__main__":
    rebuild_full_index()
