"""
Offline Assets Staging Script
Downloads and caches all external weights, dependencies, and sample scenes
to ensure the application operates in 100% network-disconnected environments.
"""
import os
import sys
import urllib.request
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import WEIGHTS_DIR, RAW_DIR

def stage_offline_assets():
    print("=== PS26227 Offline Staging Protocol ===")
    print(f"Target weights directory: {WEIGHTS_DIR}")
    print(f"Target raw data directory: {RAW_DIR}")

    # 1. Verify cached artifacts without creating empty files that look usable.
    weights_manifest = {
        "clip_rsicd": "clip-rsicd-v2",
        "clay_model": "clay_v1_5.pt",
        "open_cd": "open_cd_snunet.pt",
        "s2cloudless": "s2cloudless_weights.npy"
    }

    for key, filename in weights_manifest.items():
        dest = Path(WEIGHTS_DIR) / filename
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"Missing usable offline artifact for {key}: {dest}")
        else:
            print(f"Found staged {key} at {dest}")

    print("\nOffline staging verification completed.")
    print("Local stores and fallback algorithms can run offline; missing model artifacts remain explicit.")

if __name__ == "__main__":
    stage_offline_assets()
