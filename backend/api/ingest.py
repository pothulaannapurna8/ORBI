"""
Incremental Ingestion API Endpoint
Supports live scene ingestion from file path or archive.
Pipeline: Tile → Mask → Encode (CLIP/Clay) → Upsert Qdrant → Insert PostGIS.
Updates vector DB and spatial database without full-archive rebuilds.

Endpoints:
- POST /ingest: Ingest scene by path, returning summary of tiles created
"""
import logging
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Query, File, UploadFile, Form
from pydantic import BaseModel, Field

from backend.config import INCOMING_DIR
from backend.database.db_manager import db
from backend.ingestion.tiler import tiler
from backend.quality.s2cloudless_mask import s2_masker
from backend.embeddings.clip_rsicd_encoder import clip_encoder
from backend.embeddings.clay_encoder import clay_encoder
from backend.database.qdrant_client import qdrant_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestRequest(BaseModel):
    """Request body for scene ingestion."""
    scene_path: str = Field(
        ...,
        description="Absolute path to scene GeoTIFF or NetCDF file"
    )
    acquisition_date: Optional[str] = Field(
        default=None,
        description="ISO date (YYYY-MM-DD); extracted from filename if omitted"
    )
    source_name: Optional[str] = Field(
        default="Sentinel-2 L2A",
        description="Data source (e.g., Sentinel-2 L2A, Landsat 8, custom)"
    )
    center_lat: Optional[float] = Field(
        default=None,
        description="Latitude of scene center (extracted from metadata if omitted)"
    )
    center_lon: Optional[float] = Field(
        default=None,
        description="Longitude of scene center (extracted from metadata if omitted)"
    )
    tile_size: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Tile size in pixels (default 512×512)"
    )
    encoding_model: str = Field(
        default="clip",
        description="Encoding model: 'clip' (RGB) or 'clay' (multi-spectral)"
    )
    skip_cloud_mask: bool = Field(
        default=False,
        description="Skip cloud masking if True"
    )


class TileIngestResponse(BaseModel):
    """Response model for ingest operation."""
    status: str
    ingest_id: str
    scene_path: str
    tiles_created: int
    tiles_encoded: int
    tiles_upserted_qdrant: int
    tiles_upserted_postgis: int
    average_cloud_cover: float
    encoding_model: str
    total_pixels_processed: int
    errors: List[str]
    ingest_start_time: str
    ingest_end_time: str
    duration_seconds: float


@router.post("", response_model=TileIngestResponse)
async def ingest_scene(req: IngestRequest) -> Dict[str, Any]:
    """
    Ingest a satellite scene into the retrieval system.
    
    Complete pipeline:
    1. Validate input scene path
    2. Tile scene into 512×512 patches (with overlap if requested)
    3. Apply cloud masking (S2Cloudless spectral indices)
    4. Encode tiles (CLIP RGB or Clay multi-spectral)
    5. Upsert to Qdrant (semantic_tiles collection)
    6. Insert tile metadata to PostGIS (tile table)
    
    Args:
        req: IngestRequest with scene_path and optional metadata
    
    Returns:
        {
            "status": "success" | "partial_failure",
            "ingest_id": str (UUID),
            "tiles_created": int,
            "tiles_encoded": int,
            "tiles_upserted_qdrant": int,
            "tiles_upserted_postgis": int,
            "average_cloud_cover": float,
            "encoding_model": str,
            "total_pixels_processed": int,
            "errors": [str],
            "duration_seconds": float
        }
    
    Raises:
        HTTPException 404: Scene file not found
        HTTPException 400: Invalid request parameters
        HTTPException 500: Processing error
    """
    ingest_start = datetime.now()
    ingest_id = str(uuid.uuid4())
    
    errors = []
    tiles_created = 0
    tiles_encoded = 0
    tiles_upserted_qdrant = 0
    tiles_upserted_postgis = 0
    total_cloud_cover = 0.0
    total_pixels_processed = 0
    
    try:
        # Validate scene path
        scene_path = Path(req.scene_path)
        if not scene_path.exists():
            logger.error(f"Scene file not found: {req.scene_path}")
            raise HTTPException(status_code=404, detail=f"Scene not found: {req.scene_path}")
        
        logger.info(f"[{ingest_id}] Ingesting scene: {req.scene_path}")
        
        # Step 1: Load scene raster
        try:
            import rasterio
            import numpy as np
            
            with rasterio.open(str(scene_path)) as src:
                # Read multi-band raster (Sentinel-2 L2A = 11 bands)
                bands = src.read()
                crs = src.crs
                transform = src.transform
                metadata = src.meta
                
                logger.debug(f"Loaded scene: bands={bands.shape}, crs={crs}")
        except Exception as e:
            error_msg = f"Failed to load raster: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Step 2: Tile scene
        try:
            tiles = tiler.tile_scene(
                raster=bands,
                tile_size=req.tile_size,
                transform=transform,
                crs=crs
            )
            tiles_created = len(tiles)
            logger.info(f"[{ingest_id}] Created {tiles_created} tiles")
        except Exception as e:
            error_msg = f"Tiling failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            tiles_created = 0
        
        # Step 3: Process each tile (mask, encode, upsert)
        for i, tile in enumerate(tiles):
            try:
                tile_id = str(uuid.uuid4())
                tile_bands = tile.get("bands")  # Dict of band arrays
                
                # Cloud masking
                if not req.skip_cloud_mask:
                    try:
                        # Extract necessary bands: blue, green, red, nir, swir
                        cloud_mask, cloud_fraction, _ = s2_masker.compute_mask(
                            tile_bands
                        )
                        total_cloud_cover += cloud_fraction
                    except Exception as e:
                        logger.warning(f"Cloud masking failed for tile {tile_id}: {e}")
                        cloud_fraction = 0.0
                else:
                    cloud_fraction = 0.0
                
                # Encoding
                embedding_vector = None
                if req.encoding_model.lower() == "clip":
                    try:
                        # RGB composite
                        rgb = np.stack([
                            tile_bands.get(3, np.zeros((req.tile_size, req.tile_size))),  # Red
                            tile_bands.get(2, np.zeros((req.tile_size, req.tile_size))),  # Green
                            tile_bands.get(1, np.zeros((req.tile_size, req.tile_size)))   # Blue
                        ])
                        embedding_vector = clip_encoder.encode_tile(rgb)
                        tiles_encoded += 1
                    except Exception as e:
                        logger.warning(f"CLIP encoding failed for tile {tile_id}: {e}")
                
                elif req.encoding_model.lower() == "clay":
                    try:
                        embedding_vector = clay_encoder.encode_tile(tile_bands)
                        tiles_encoded += 1
                    except Exception as e:
                        logger.warning(f"Clay encoding failed for tile {tile_id}: {e}")
                
                # Upsert to Qdrant
                if embedding_vector is not None:
                    try:
                        qdrant_store.upsert(
                            tile_id=tile_id,
                            embedding=embedding_vector.tolist(),
                            metadata={
                                "scene_path": str(req.scene_path),
                                "tile_bounds": tile.get("bounds"),
                                "cloud_cover": cloud_fraction,
                                "acquisition_date": req.acquisition_date or datetime.now().isoformat()[:10]
                            }
                        )
                        tiles_upserted_qdrant += 1
                    except Exception as e:
                        logger.warning(f"Qdrant upsert failed for tile {tile_id}: {e}")
                        errors.append(f"Qdrant upsert: {e}")
                
                # Insert to PostGIS
                try:
                    bounds = tile.get("bounds")
                    db.insert_tile(
                        tile_id=tile_id,
                        scene_path=str(req.scene_path),
                        bounds=bounds,
                        cloud_cover=cloud_fraction,
                        acquisition_datetime=req.acquisition_date or datetime.now().isoformat(),
                        source_name=req.source_name,
                        embedding_model=req.encoding_model
                    )
                    tiles_upserted_postgis += 1
                except Exception as e:
                    logger.warning(f"PostGIS insert failed for tile {tile_id}: {e}")
                    errors.append(f"PostGIS: {e}")
                
                total_pixels_processed += req.tile_size * req.tile_size
            
            except Exception as e:
                error_msg = f"Tile {i} processing failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
        
        # Calculate statistics
        avg_cloud_cover = (
            total_cloud_cover / tiles_created if tiles_created > 0 else 0.0
        )
        
        ingest_end = datetime.now()
        duration_seconds = (ingest_end - ingest_start).total_seconds()
        
        # Determine status
        status = "success" if len(errors) == 0 else "partial_failure"
        
        logger.info(
            f"[{ingest_id}] Ingest complete: {tiles_upserted_qdrant}/{tiles_created} "
            f"tiles encoded/upserted in {duration_seconds:.2f}s"
        )
        
        return {
            "status": status,
            "ingest_id": ingest_id,
            "scene_path": str(req.scene_path),
            "tiles_created": tiles_created,
            "tiles_encoded": tiles_encoded,
            "tiles_upserted_qdrant": tiles_upserted_qdrant,
            "tiles_upserted_postgis": tiles_upserted_postgis,
            "average_cloud_cover": round(avg_cloud_cover, 3),
            "encoding_model": req.encoding_model,
            "total_pixels_processed": total_pixels_processed,
            "errors": errors,
            "ingest_start_time": ingest_start.isoformat(),
            "ingest_end_time": ingest_end.isoformat(),
            "duration_seconds": round(duration_seconds, 2)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{ingest_id}] Unhandled ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@router.post("/validate")
async def validate_scene(req: IngestRequest) -> Dict[str, Any]:
    """
    Pre-flight validation for a scene without full ingestion.
    
    Checks:
    - File exists and is readable
    - Raster has expected bands
    - CRS and geotransform are valid
    
    Args:
        req: IngestRequest
    
    Returns:
        {
            "valid": bool,
            "scene_path": str,
            "band_count": int,
            "dimensions": (height, width),
            "crs": str,
            "errors": [str]
        }
    """
    try:
        import rasterio
        
        scene_path = Path(req.scene_path)
        
        if not scene_path.exists():
            return {
                "valid": False,
                "scene_path": str(req.scene_path),
                "errors": ["File not found"]
            }
        
        with rasterio.open(str(scene_path)) as src:
            return {
                "valid": True,
                "scene_path": str(req.scene_path),
                "band_count": src.count,
                "dimensions": (src.height, src.width),
                "crs": str(src.crs),
                "errors": []
            }
    
    except Exception as e:
        return {
            "valid": False,
            "scene_path": str(req.scene_path),
            "errors": [str(e)]
        }




@router.post("/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    acquisition_date: str = Form("2025-11-20"),
    source_name: str = Form("Sentinel-2 User Upload"),
    center_lat: float = Form(12.9716),
    center_lon: float = Form(77.5946)
):
    save_path = Path(INCOMING_DIR) / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = incremental_service.ingest_single_scene(
        filepath=str(save_path),
        acquisition_date=acquisition_date,
        source_name=source_name,
        center_lat=center_lat,
        center_lon=center_lon
    )
    return result
