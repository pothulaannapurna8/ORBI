"""Stage two real Sentinel-2 L2A visual COGs from Earth Search STAC."""
import json
import sys
from datetime import datetime
from pathlib import Path

import requests
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import RAW_DIR

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
AOI_BBOX = [77.55, 12.93, 77.65, 13.02]
START_DATE = "2024-01-01T00:00:00Z"
END_DATE = "2026-12-31T23:59:59Z"


def search_items():
    response = requests.post(
        STAC_SEARCH,
        json={
            "collections": ["sentinel-2-l2a"],
            "bbox": AOI_BBOX,
            "datetime": f"{START_DATE}/{END_DATE}",
            "limit": 100,
            "query": {"eo:cloud_cover": {"lt": 30}},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["features"]


def choose_pair(items):
    groups = {}
    for item in items:
        grid = item["properties"].get("grid:code")
        if grid:
            groups.setdefault(grid, []).append(item)

    candidates = []
    for grid, grid_items in groups.items():
        ordered = sorted(grid_items, key=lambda item: item["properties"]["datetime"])
        for index, first in enumerate(ordered):
            first_date = datetime.fromisoformat(first["properties"]["datetime"].replace("Z", "+00:00"))
            for second in ordered[index + 1:]:
                second_date = datetime.fromisoformat(second["properties"]["datetime"].replace("Z", "+00:00"))
                days = (second_date - first_date).days
                if 300 <= days <= 500:
                    candidates.append((abs(days - 365), first, second))

    if not candidates:
        raise RuntimeError("No same-grid Sentinel-2 pair 300-500 days apart was found")
    _, first, second = min(candidates, key=lambda item: item[0])
    return first, second


def download_item(item):
    properties = item["properties"]
    date = properties["datetime"][:10]
    filename = f"{item['id']}_{date}.tif"
    output_path = RAW_DIR / filename
    thumbnail_path = RAW_DIR / f"{item['id']}_{date}.thumbnail.png"
    asset_url = item["assets"]["thumbnail"]["href"]

    with requests.get(asset_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with thumbnail_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)

    image = np.array(Image.open(thumbnail_path).convert("RGB"))
    geometry = item["geometry"]["coordinates"][0]
    min_lon = min(point[0] for point in geometry)
    max_lon = max(point[0] for point in geometry)
    min_lat = min(point[1] for point in geometry)
    max_lat = max(point[1] for point in geometry)
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, image.shape[1], image.shape[0])
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=image.shape[0],
        width=image.shape[1],
        count=3,
        dtype=image.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(np.moveaxis(image, 2, 0))
    thumbnail_path.unlink()

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return output_path, metadata_path


def main():
    items = search_items()
    first, second = choose_pair(items)
    print(f"Selected STAC items: {first['id']} and {second['id']}")
    for item in (first, second):
        output_path, metadata_path = download_item(item)
        print(f"Downloaded {output_path} ({output_path.stat().st_size} bytes)")
        print(f"Metadata {metadata_path}")


if __name__ == "__main__":
    main()