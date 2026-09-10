"""Stage real Sentinel-2 L2A multispectral windows from Earth Search STAC."""
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import RAW_DIR

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
START_DATE = "2023-01-01T00:00:00Z"
END_DATE = "2026-12-31T23:59:59Z"
MAX_CLOUD = 30
CHUNK_BYTES = 8 * 1024 * 1024
AOIS = {
    "urban_edge": [77.55, 12.93, 77.65, 13.02],
    "coastline": [80.20, 12.95, 80.35, 13.10],
    "agriculture_edge": [77.20, 12.70, 77.35, 12.85],
}
# B10 is not exposed by Earth Search's Sentinel-2 L2A COG collection.
BAND_ASSETS = [
    ("B02", "blue"), ("B03", "green"), ("B04", "red"),
    ("B05", "rededge1"), ("B06", "rededge2"),
    ("B07", "rededge3"), ("B08", "nir"), ("B8A", "nir08"),
    ("B11", "swir16"), ("B12", "swir22"), ("B01", "coastal"), ("B09", "nir09"),
]


def search_items(bbox):
    response = requests.post(
        STAC_SEARCH,
        json={
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": f"{START_DATE}/{END_DATE}",
            "limit": 100,
            "query": {"eo:cloud_cover": {"lt": MAX_CLOUD}},
        },
        timeout=(10, 60),
    )
    response.raise_for_status()
    return response.json()["features"]


def choose_pair(items):
    """Choose lowest-cloud same-grid pair closest to one year apart."""
    groups = {}
    for item in items:
        groups.setdefault(item["properties"].get("grid:code"), []).append(item)
    candidates = []
    for grid_items in groups.values():
        ordered = sorted(grid_items, key=lambda item: (item["properties"]["datetime"], item["id"]))
        for index, first in enumerate(ordered):
            first_date = datetime.fromisoformat(first["properties"]["datetime"].replace("Z", "+00:00"))
            for second in ordered[index + 1:]:
                second_date = datetime.fromisoformat(second["properties"]["datetime"].replace("Z", "+00:00"))
                days = (second_date - first_date).days
                if 180 <= days <= 550:
                    cloud = max(first["properties"].get("eo:cloud_cover", 100), second["properties"].get("eo:cloud_cover", 100))
                    candidates.append((cloud, abs(days - 365), first["id"], second["id"], first, second))
    if not candidates:
        raise RuntimeError("No same-grid Sentinel-2 pair 6-18 months apart was found")
    selected = min(candidates)
    return selected[-2], selected[-1]


def download_range(url, destination):
    """Resumable 8 MB fallback when GDAL cannot read the remote COG window."""
    head = requests.head(url, timeout=(10, 30), allow_redirects=True)
    head.raise_for_status()
    expected = int(head.headers.get("Content-Length", "0"))
    offset = destination.stat().st_size if destination.exists() else 0
    if expected and offset > expected:
        destination.unlink()
        offset = 0
    while not expected or offset < expected:
        end = min(offset + CHUNK_BYTES - 1, expected - 1) if expected else offset + CHUNK_BYTES - 1
        for attempt in range(5):
            try:
                response = requests.get(url, headers={"Range": f"bytes={offset}-{end}"}, stream=True, timeout=(10, 120))
                response.raise_for_status()
                with destination.open("ab") as output:
                    for chunk in response.iter_content(CHUNK_BYTES):
                        if chunk:
                            output.write(chunk)
                offset = destination.stat().st_size
                print(f"  fallback download {destination.name}: {offset}/{expected or '?'} bytes", flush=True)
                break
            except (requests.RequestException, OSError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    if destination.stat().st_size == 0 or (expected and destination.stat().st_size != expected):
        raise RuntimeError(f"Truncated asset {destination}: {destination.stat().st_size}/{expected}")
    return destination


def read_window(url, bbox, fallback_dir):
    """Read only the AOI window from a remote COG, with a resumable fallback."""
    try:
        with rasterio.open(url) as dataset:
            bounds = transform_bounds("EPSG:4326", dataset.crs, *bbox, densify_pts=21)
            window = from_bounds(*bounds, transform=dataset.transform).intersection(Window(0, 0, dataset.width, dataset.height))
            width, height = max(1, math.ceil(window.width)), max(1, math.ceil(window.height))
            data = dataset.read(1, window=window, out_shape=(height, width), resampling=Resampling.bilinear)
            return data.astype(np.uint16), dataset.window_transform(window), dataset.crs
    except Exception as exc:
        print(f"  remote window read failed: {exc}; using ranged fallback", flush=True)
        local = download_range(url, fallback_dir / Path(url).name)
        with rasterio.open(local) as dataset:
            bounds = transform_bounds("EPSG:4326", dataset.crs, *bbox, densify_pts=21)
            window = from_bounds(*bounds, transform=dataset.transform).intersection(Window(0, 0, dataset.width, dataset.height))
            width, height = max(1, math.ceil(window.width)), max(1, math.ceil(window.height))
            data = dataset.read(1, window=window, out_shape=(height, width), resampling=Resampling.bilinear)
            return data.astype(np.uint16), dataset.window_transform(window), dataset.crs


def stage_item(item, aoi_name, bbox):
    item_id = item["id"]
    date = item["properties"]["datetime"][:10]
    assets = item["assets"]
    missing = [asset for _, asset in BAND_ASSETS if asset not in assets]
    if missing:
        raise RuntimeError(f"{item_id} missing assets: {missing}")
    cache = RAW_DIR / "_asset_cache" / item_id
    cache.mkdir(parents=True, exist_ok=True)
    arrays = []
    transform = None
    crs = None
    for band_name, asset_name in BAND_ASSETS:
        data, band_transform, band_crs = read_window(assets[asset_name]["href"], bbox, cache)
        print(f"  {band_name}: {data.shape} {data.dtype} {band_crs}", flush=True)
        if transform is None:
            transform, crs = band_transform, band_crs
        arrays.append(data)
    target_shape = arrays[1].shape
    aligned = []
    for data in arrays:
        if data.shape != target_shape:
            data = np.asarray(Image.fromarray(data).resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR))
        aligned.append(data.astype(np.uint16))
    output_path = RAW_DIR / f"{aoi_name}_{item_id}_{date}_bands.tif"
    stack = np.stack(aligned)
    with rasterio.open(output_path, "w", driver="GTiff", height=stack.shape[1], width=stack.shape[2], count=stack.shape[0], dtype="uint16", crs=crs, transform=transform, compress="deflate") as dataset:
        dataset.write(stack)
    metadata = dict(item)
    metadata["orbi_aoi"] = aoi_name
    metadata["orbi_band_order"] = [band for band, _ in BAND_ASSETS]
    metadata["orbi_missing_l2a_assets"] = ["B10"]
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Staged {output_path} ({output_path.stat().st_size} bytes), CRS={crs}", flush=True)
    return output_path


def main():
    staged = []
    for aoi_name, bbox in AOIS.items():
        first, second = choose_pair(search_items(bbox))
        print(f"{aoi_name}: selected {first['id']} and {second['id']}", flush=True)
        staged.extend(stage_item(item, aoi_name, bbox) for item in (first, second))
    print(f"Staged {len(staged)} full-band scene windows", flush=True)


if __name__ == "__main__":
    main()
