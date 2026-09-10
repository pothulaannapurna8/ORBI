import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("QDRANT_STORAGE_DIR", str(ROOT / "data" / "qdrant_ingest_test"))
os.environ.setdefault("QDRANT_LOCAL_PATH", str(ROOT / "data" / "qdrant_ingest_test"))
TEST_QDRANT_DIR = Path(os.environ["QDRANT_STORAGE_DIR"])

from backend.config import INCOMING_DIR, SQLITE_DB_PATH
from backend.database.qdrant_client import qdrant_store
from backend.ingestion.tiler import compute_location_key
from backend.ingestion.watch_folder import HAS_WATCHDOG, SceneEventHandler, IncrementalIngestionService


def _tile_location_counts():
    connection = sqlite3.connect(str(SQLITE_DB_PATH))
    rows = connection.execute(
        "SELECT location_key, COUNT(*) FROM tile GROUP BY location_key"
    ).fetchall()
    connection.close()
    return {location_key: count for location_key, count in rows}


def _qdrant_count(collection_name: str) -> int:
    return qdrant_store.client.count(collection_name=collection_name, exact=True).count


def _expected_scene_locations(center_lat: float, center_lon: float):
    expected = set()
    tile_lat = center_lat - 0.023
    tile_lon = center_lon - 0.023
    expected.add(compute_location_key(tile_lat, tile_lon))
    return expected


@pytest.mark.integration
def test_watch_folder_incremental_ingestion():
    """Programmatically drop a GeoTIFF into the incoming folder and assert only the affected grid cells are indexed."""
    if not HAS_WATCHDOG:
        pytest.fail("watchdog is required for the incremental ingestion integration test")

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)

    center_lat = 18.5214
    center_lon = 73.8567
    expected_locations = _expected_scene_locations(center_lat, center_lon)

    before_tile_counts = _tile_location_counts()
    before_semantic = _qdrant_count("semantic_tiles")
    before_spectral = _qdrant_count("spectral_tiles")

    scene_name = f"watch_incoming_{uuid.uuid4().hex}.tif"
    scene_path = INCOMING_DIR / scene_name
    staged_path = INCOMING_DIR / f"dropped_{uuid.uuid4().hex}.tif"

    try:
        bands = np.zeros((13, 512, 512), dtype=np.uint16)
        bands[0:3, :, :] = 1800
        bands[3:7, :, :] = 2200
        bands[7:10, :, :] = 2600
        bands[10:13, :, :] = 3000

        transform = from_origin(center_lon - 0.0256, center_lat + 0.0256, 0.0001, 0.0001)
        with rasterio.open(
            scene_path,
            mode="w",
            driver="GTiff",
            width=512,
            height=512,
            count=13,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(bands)

        service = IncrementalIngestionService(str(INCOMING_DIR))
        handler = SceneEventHandler(service)

        # Simulate the watch-folder event by dropping a file into the monitored incoming directory.
        shutil.copy2(scene_path, staged_path)
        event = SimpleNamespace(is_directory=False, src_path=str(staged_path))
        handler.on_created(event)

        deadline = time.monotonic() + 90
        new_locations = set()

        while time.monotonic() < deadline:
            after_tile_counts = _tile_location_counts()
            new_locations = {
                location_key
                for location_key, count in after_tile_counts.items()
                if count > before_tile_counts.get(location_key, 0)
            }

            if new_locations and new_locations.issubset(expected_locations):
                after_semantic = _qdrant_count("semantic_tiles")
                after_spectral = _qdrant_count("spectral_tiles")
                if after_semantic >= before_semantic + 1 and after_spectral >= before_spectral + 1:
                    break

            time.sleep(0.5)
        else:
            filepath_key = str(staged_path.resolve())
            pytest.fail(
                "watch-folder processing did not complete within the timeout; "
                f"locations={sorted(new_locations)}, "
                f"expected={sorted(expected_locations)}, "
                f"semantic={_qdrant_count('semantic_tiles')} (before={before_semantic}), "
                f"spectral={_qdrant_count('spectral_tiles')} (before={before_spectral}), "
                f"errors={handler.errors.get(filepath_key)}"
            )

        assert new_locations, "the ingestion pipeline did not create any new location_key updates"
        assert new_locations <= expected_locations, "ingestion touched grid cells outside the incoming scene footprint"
        assert not any(location_key not in expected_locations for location_key in new_locations)

        after_semantic = _qdrant_count("semantic_tiles")
        after_spectral = _qdrant_count("spectral_tiles")
        assert after_semantic > before_semantic
        assert after_spectral > before_spectral

        for location_key in new_locations:
            connection = sqlite3.connect(str(SQLITE_DB_PATH))
            tile_count = connection.execute(
                "SELECT COUNT(*) FROM tile WHERE location_key = ?",
                (location_key,),
            ).fetchone()[0]
            connection.close()
            assert tile_count > 0, f"location_key {location_key} was not persisted to SQLite"

        assert len(new_locations) <= len(expected_locations)
        assert len(new_locations) > 0

    finally:
        for path in [scene_path, staged_path]:
            if path.exists():
                path.unlink()

        if TEST_QDRANT_DIR.exists():
            shutil.rmtree(TEST_QDRANT_DIR, ignore_errors=True)
