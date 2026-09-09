"""
Incremental Ingestion Engine & Folder Watcher
Monitors incoming directory for new Sentinel-2 scenes.
Updates Qdrant and PostGIS incrementally and triggers temporal analysis
ONLY for affected location_keys, avoiding whole-archive recomputation.

Architecture:
1. Watchdog monitors data/incoming/ for new GeoTIFF/COG files
2. On detection: Load scene → Tile → Mask → Encode → Upsert Qdrant → Insert PostGIS
3. Query affected location_keys and re-run change detection (multi-temporal analysis)
4. Update change_result table with earliest change dates and confidence scores
5. Publish events for real-time dashboard updates

Dependencies: watchdog, rasterio, numpy, PIL, torch (embeddings)
"""
import logging
import os
import time
import threading
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    Observer = None

    class FileSystemEventHandler:
        """Fallback base class when optional watchdog support is unavailable."""

        pass

from backend.config import INCOMING_DIR, TILE_SIZE_PX
from backend.ingestion.tiler import SceneTiler, compute_location_key
from backend.database.db_manager import db
from backend.database.qdrant_client import qdrant_store
from backend.quality.s2cloudless_mask import s2_masker
from backend.quality.arosics_register import arosics_aligner
from backend.change_detection.open_cd_wrapper import open_cd_detector
from backend.change_detection.ndwi_water import ndwi_detector
from backend.change_detection.change_confidence import compute_change_confidence_score
from backend.change_detection.earliest_change import estimate_earliest_change_date

logger = logging.getLogger(__name__)


class IncrementalIngestionService:
    """
    Manages incremental scene ingestion and multi-temporal change detection.
    
    Key capabilities:
    - Single-scene processing (programmatic or from watcher)
    - Location-key-based change re-evaluation
    - Temporal timeline building for multi-temporal analysis
    - Cloud detection & suppression of false alarms
    """
    
    def __init__(self, watch_dir: str = str(INCOMING_DIR)):
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.tiler = SceneTiler(tile_size=TILE_SIZE_PX)
        self.observer = None
        self.processed_files: Set[str] = set()
        logger.info(f"Initialized IncrementalIngestionService, watching: {self.watch_dir}")

    def ingest_single_scene(
        self,
        filepath: str,
        acquisition_date: str = "2025-11-20",
        source_name: str = "S2_INCOMING",
        center_lat: float = 12.9716,
        center_lon: float = 77.5946
    ) -> Dict[str, Any]:
        """
        Incrementally processes one scene file.
        
        Pipeline:
        1. Tile and index scene into Qdrant + PostGIS
        2. For each affected location_key, query temporal history
        3. Pair earliest and latest tiles, run co-registration + change detection
        4. Compute multi-temporal confidence scores
        5. Update change_result table
        
        Args:
            filepath: Path to GeoTIFF or COG scene file
            acquisition_date: ISO date string (YYYY-MM-DD)
            source_name: Data source identifier
            center_lat, center_lon: Approximate scene center
        
        Returns:
            {
                "status": "success" | "error",
                "new_tiles_count": int,
                "affected_locations": List[str],
                "updated_change_analyses": int,
                "errors": List[str]
            }
        """
        errors = []
        
        try:
            # Step 1: Tile & index the new scene
            logger.info(f"[Ingest] Starting incremental ingest for: {filepath}")
            
            new_tiles = self.tiler.process_and_index_scene(
                scene_filepath=filepath,
                acquisition_datetime=f"{acquisition_date}T10:30:00Z",
                source_id=source_name,
                center_lat=center_lat,
                center_lon=center_lon
            )
            
            if not new_tiles:
                error_msg = "No tiles produced from scene"
                logger.error(error_msg)
                errors.append(error_msg)
                return {
                    "status": "error",
                    "new_tiles_count": 0,
                    "affected_locations": [],
                    "updated_change_analyses": 0,
                    "errors": errors
                }
            
            affected_locations = list(set(t["location_key"] for t in new_tiles))
            updated_changes = []
            
            # Step 2: Rerun temporal change analysis ONLY for affected location_keys
            for loc_key in affected_locations:
                try:
                    history_tiles = db.get_tiles_by_location(loc_key)
                    
                    if len(history_tiles) < 2:
                        logger.debug(f"[{loc_key}] Insufficient history tiles for change detection")
                        continue
                    
                    # Sort by date
                    history_tiles = sorted(history_tiles, key=lambda x: x.get("acquisition_datetime", ""))
                    ref_tile = history_tiles[0]
                    latest_tile = history_tiles[-1]
                    
                    logger.debug(
                        f"[{loc_key}] Running change analysis: {ref_tile['tile_id'][:6]} "
                        f"→ {latest_tile['tile_id'][:6]}"
                    )
                    
                    # Load RGB images for change detection
                    ref_img_path = ref_tile.get("rgb_filepath")
                    tgt_img_path = latest_tile.get("rgb_filepath")
                    
                    if not (ref_img_path and os.path.exists(ref_img_path) and 
                            tgt_img_path and os.path.exists(tgt_img_path)):
                        logger.warning(f"[{loc_key}] Image files not found")
                        continue
                    
                    ref_img = np.array(Image.open(ref_img_path).convert("RGB"))
                    tgt_img = np.array(Image.open(tgt_img_path).convert("RGB"))
                    
                    # Co-registration
                    try:
                        reg_tgt, shift_px, reg_quality = arosics_aligner.register(ref_img, tgt_img)
                    except Exception as e:
                        logger.warning(f"[{loc_key}] Registration failed: {e}")
                        reg_tgt = tgt_img
                        shift_px = 0.0
                        reg_quality = 0.5
                    
                    # Open-CD change detection
                    change_mask, change_score = open_cd_detector.detect_change(ref_img, reg_tgt)
                    
                    # Build timeline for earliest change estimation
                    timeline = []
                    for t in history_tiles:
                        t_img_path = t.get("rgb_filepath")
                        if t_img_path and os.path.exists(t_img_path):
                            try:
                                t_img = np.array(Image.open(t_img_path).convert("RGB"))
                                _, t_score = open_cd_detector.detect_change(ref_img, t_img)
                                timeline.append({
                                    "date": t["acquisition_datetime"][:10],
                                    "change_score": float(t_score),
                                    "cloud_score": float(t.get("cloud_cover", 0.0))
                                })
                            except Exception as e:
                                logger.warning(f"[{loc_key}] Timeline image loading failed: {e}")
                                continue
                    
                    if len(timeline) < 2:
                        logger.debug(f"[{loc_key}] Insufficient timeline for estimation")
                        continue
                    
                    # Estimate earliest change date
                    earliest_date, temporal_consistency, _ = estimate_earliest_change_date(
                        timeline,
                        threshold=0.40,
                        persistence_lookhead=2,
                        max_cloud_fraction=0.30
                    )
                    
                    # Compute Change Confidence Score
                    cloud_score = float(latest_tile.get("cloud_cover", 0.0))
                    ccs, _ = compute_change_confidence_score(
                        change_evidence=float(change_score),
                        cloud_score=cloud_score,
                        registration_quality=reg_quality,
                        temporal_consistency=temporal_consistency
                    )
                    
                    # Save change mask PNG
                    mask_filename = f"mask_{loc_key}_{ref_tile['tile_id'][:8]}_{latest_tile['tile_id'][:8]}.png"
                    mask_path = Path(ref_tile.get("rgb_filepath", "")).parent / mask_filename
                    
                    try:
                        Image.fromarray((change_mask * 255).astype(np.uint8)).save(str(mask_path))
                    except Exception as e:
                        logger.warning(f"[{loc_key}] Mask save failed: {e}")
                        mask_path = None
                    
                    # Create change record
                    change_record = {
                        "location_key": loc_key,
                        "before_tile_id": ref_tile["tile_id"],
                        "after_tile_id": latest_tile["tile_id"],
                        "earliest_change_date": earliest_date,
                        "change_type": "construction" if ccs >= 0.40 else "no_change",
                        "change_score": float(change_score),
                        "confidence": float(ccs),
                        "evidence_paths": {
                            "before_png": ref_img_path,
                            "after_png": tgt_img_path,
                            "change_mask_png": str(mask_path) if mask_path else None
                        },
                        "registration_shift_px": shift_px,
                        "cloud_mask_quality": 1.0 - cloud_score,
                        "registration_quality": float(reg_quality),
                        "ccs_breakdown": {
                            "change_evidence": float(change_score),
                            "cloud_score": cloud_score,
                            "registration_quality": float(reg_quality),
                            "temporal_consistency": float(temporal_consistency)
                        },
                        "model_version": "Open-CD-SNUNet-v1.2",
                        "processing_version": "v1.0.0"
                    }
                    
                    # Insert or update in database
                    try:
                        change_id = db.insert_change_result(change_record)
                        change_record["change_id"] = change_id
                        updated_changes.append(change_record)
                        logger.info(
                            f"[{loc_key}] Change result created: {change_id}, "
                            f"CCS={ccs:.3f}, earliest_date={earliest_date}"
                        )
                    except Exception as e:
                        logger.error(f"[{loc_key}] Database insert failed: {e}")
                        errors.append(f"DB insert failed for {loc_key}: {e}")
                
                except Exception as e:
                    error_msg = f"[{loc_key}] Temporal analysis error: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            logger.info(
                f"[Ingest] Complete: {len(new_tiles)} tiles, "
                f"{len(affected_locations)} locations, "
                f"{len(updated_changes)} changes"
            )
            
            return {
                "status": "success",
                "new_tiles_count": len(new_tiles),
                "affected_locations": affected_locations,
                "updated_change_analyses": len(updated_changes),
                "errors": errors
            }
        
        except Exception as e:
            error_msg = f"Ingest pipeline error: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                "status": "error",
                "new_tiles_count": 0,
                "affected_locations": [],
                "updated_change_analyses": 0,
                "errors": [error_msg]
            }

    def start_watcher(self) -> Optional[Observer]:
        """
        Start the watchdog observer to monitor the incoming directory.
        
        Returns:
            Observer instance or None if watchdog not available
        """
        if not HAS_WATCHDOG:
            logger.warning("watchdog not installed; folder monitoring disabled")
            return None
        
        event_handler = SceneEventHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        logger.info(f"Watchdog observer started for: {self.watch_dir}")
        return self.observer

    def stop_watcher(self):
        """Stop the watchdog observer."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Watchdog observer stopped")


class SceneEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler for detecting new scene files.
    Filters for .tif, .tiff, .cog extensions and triggers ingestion.
    """
    
    def __init__(self, ingest_service: IncrementalIngestionService):
        self.ingest_service = ingest_service
        self.processing = set()

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        
        # Check file extension
        if filepath.suffix.lower() not in ['.tif', '.tiff', '.cog', '.geotiff']:
            return
        
        # Avoid processing the same file multiple times
        if str(filepath) in self.processing:
            return
        
        logger.info(f"[Watch] Detected new scene: {filepath.name}")
        
        # Add small delay to ensure file is complete
        time.sleep(2)
        
        # Check if file is still being written
        if not self._is_file_stable(filepath):
            logger.info(f"[Watch] File still being written, will retry: {filepath.name}")
            return
        
        # Process in background thread to avoid blocking watcher
        thread = threading.Thread(
            target=self._process_scene,
            args=(filepath,),
            daemon=True
        )
        thread.start()

    def _is_file_stable(self, filepath: Path, check_interval: float = 0.5) -> bool:
        """Check if file size is stable (not being written)."""
        try:
            size1 = filepath.stat().st_size
            time.sleep(check_interval)
            size2 = filepath.stat().st_size
            return size1 == size2
        except Exception:
            return False

    def _process_scene(self, filepath: Path):
        """Process scene in background thread."""
        self.processing.add(str(filepath))
        
        try:
            # Extract acquisition date from filename or use current date
            acquisition_date = self._extract_date_from_filename(filepath.name)
            
            logger.info(f"[Watch] Ingesting: {filepath.name} (date: {acquisition_date})")
            
            result = self.ingest_service.ingest_single_scene(
                filepath=str(filepath),
                acquisition_date=acquisition_date,
                source_name="Sentinel-2_Incoming"
            )
            
            if result["status"] == "success":
                logger.info(
                    f"[Watch] Ingest successful: {result['new_tiles_count']} tiles, "
                    f"{result['updated_change_analyses']} changes"
                )
            else:
                logger.error(f"[Watch] Ingest failed: {result.get('errors', [])}")
        
        except Exception as e:
            logger.error(f"[Watch] Error processing {filepath.name}: {e}", exc_info=True)
        
        finally:
            self.processing.discard(str(filepath))

    @staticmethod
    def _extract_date_from_filename(filename: str) -> str:
        """
        Extract ISO date from Sentinel-2 filename.
        S2A_MSIL2A_20240315T053601_N0511_R104_T43PEQ_20240315T090646.zip
        → 2024-03-15
        """
        match = re.search(r'(\d{8})T', filename)
        if match:
            date_str = match.group(1)
            return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _archive_file(filepath: Path):
        """Optional: Move processed file to archive directory."""
        try:
            archive_dir = filepath.parent / "archive"
            archive_dir.mkdir(exist_ok=True)
            archive_path = archive_dir / filepath.name
            filepath.rename(archive_path)
            logger.info(f"Archived: {archive_path}")
        except Exception as e:
            logger.warning(f"Archive failed: {e}")


# Global singleton instances
incremental_service = IncrementalIngestionService()

# Optional: Auto-start watcher (disable in tests)
def start_background_watcher():
    """Start the background watch service."""
    try:
        observer = incremental_service.start_watcher()
        if observer:
            logger.info("Background watcher active")
            return observer
    except Exception as e:
        logger.error(f"Failed to start watcher: {e}")
    return None


if __name__ == "__main__":
    # CLI: Run watcher in foreground for manual testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    observer = start_background_watcher()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        incremental_service.stop_watcher()
        logger.info("Watcher stopped")
