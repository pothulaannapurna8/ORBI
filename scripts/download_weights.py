"""Download and verify pretrained PS26227 model weights.

Usage:
    python scripts/download_weights.py

Configuration:
    CLIP_HF_REPO       Hugging Face repo for CLIP-RSICD.
    OPEN_CD_WEIGHTS_URL Direct URL to a trained open_cd_snunet.pt checkpoint.
    OPEN_CD_HF_REPO    Optional Hugging Face repo containing open_cd_snunet.pt.
    OPEN_CD_HF_FILE   Filename inside OPEN_CD_HF_REPO (default: open_cd_snunet.pt).
    *_SHA256           Optional SHA-256 integrity checks.

The script is resumable at the repository level: existing non-empty model
files/directories are kept. For offline use, place downloaded artifacts at the
paths below and rerun with --offline to verify them without network access.
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "data" / "weights"
CLIP_DIR = WEIGHTS_DIR / "clip_rsicd"
LEGACY_CLIP_DIR = WEIGHTS_DIR / "clip-rsicd-v2"
OPEN_CD_DIR = WEIGHTS_DIR / "open_cd"
LEGACY_OPEN_CD_FILE = WEIGHTS_DIR / "open_cd_snunet.pt"
CLIP_REPO = os.getenv("CLIP_HF_REPO", "flax-community/clip-rsicd-v2")
OPEN_CD_URL = os.getenv("OPEN_CD_WEIGHTS_URL", "")
OPEN_CD_REPO = os.getenv("OPEN_CD_HF_REPO", "")
OPEN_CD_FILE = os.getenv("OPEN_CD_HF_FILE", "open_cd_snunet.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_hash: Optional[str] = None) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty weight file: {path}")
    actual = sha256(path)
    print(f"Verified {path} ({path.stat().st_size} bytes, sha256={actual})")
    if expected_hash and actual.lower() != expected_hash.lower():
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected_hash}, got {actual}")


def download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        expected = int(response.headers.get("Content-Length", "0"))
        received = 0
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                received += len(chunk)
                if expected:
                    print(f"Downloading {destination.name}: {received}/{expected} bytes", flush=True)
    if received == 0 or (expected and received != expected):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete download for {destination}: {received}/{expected} bytes")
    temporary.replace(destination)


def stage_clip(offline: bool) -> None:
    staged_dir = next(
        (
            directory
            for directory in (CLIP_DIR, LEGACY_CLIP_DIR)
            if directory.exists()
            and any(path.is_file() and path.stat().st_size > 0 for path in directory.rglob("*"))
        ),
        None,
    )
    if staged_dir:
        print(f"CLIP-RSICD already staged at {staged_dir}")
        config_path = staged_dir / "config.json"
        if not config_path.exists() or config_path.stat().st_size == 0:
            raise RuntimeError(f"CLIP-RSICD staging is incomplete; missing {config_path}")
        return
    elif offline:
        print(
            "CLIP-RSICD weights are not staged; the runtime will use its "
            "deterministic offline encoder fallback."
        )
        return
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub or run offline with pre-staged CLIP weights") from exc
        CLIP_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=CLIP_REPO, local_dir=str(CLIP_DIR), local_dir_use_symlinks=False)
        print(f"Downloaded CLIP-RSICD from {CLIP_REPO} to {CLIP_DIR}")
    required = [CLIP_DIR / "config.json"]
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"CLIP-RSICD staging is incomplete; missing {path}")


def stage_open_cd(offline: bool) -> None:
    destination = OPEN_CD_DIR / OPEN_CD_FILE
    if not destination.exists() and LEGACY_OPEN_CD_FILE.exists() and LEGACY_OPEN_CD_FILE.stat().st_size > 0:
        destination = LEGACY_OPEN_CD_FILE
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Open-CD checkpoint already staged at {destination}")
    elif offline:
            print(
                "Open-CD checkpoint is not staged; the runtime will use its "
                "deterministic Otsu fallback."
            )
            return
    elif OPEN_CD_URL:
        download_url(OPEN_CD_URL, destination)
    elif OPEN_CD_REPO:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub or provide OPEN_CD_WEIGHTS_URL") from exc
        downloaded = hf_hub_download(repo_id=OPEN_CD_REPO, filename=OPEN_CD_FILE, local_dir=str(OPEN_CD_DIR))
        if Path(downloaded) != destination:
            shutil.copy2(downloaded, destination)
    else:
        raise RuntimeError(
            "No Open-CD source configured. Set OPEN_CD_WEIGHTS_URL to the approved "
            "LEVIR-CD/S2Looking open_cd_snunet.pt URL or set OPEN_CD_HF_REPO."
        )
    verify_file(destination, os.getenv("OPEN_CD_SHA256"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Only verify already-staged weights")
    args = parser.parse_args()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        stage_clip(args.offline)
        stage_open_cd(args.offline)
    except Exception as exc:
        print(f"Weight staging failed: {exc}", file=sys.stderr)
        return 1
    print("All requested weight artifacts are staged and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
