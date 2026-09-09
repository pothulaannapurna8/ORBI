"""
Results & Temporal Analysis APIs
Provides full inspection records, temporal timelines, evidence package exports, and change history.

Endpoints:
- GET /results/{change_id}: Retrieve detailed change result record with metadata and evidence
- GET /results/{change_id}/temporal: Get chronological timeline for location with earliest change estimation
- GET /results/{change_id}/evidence: Export complete evidence package (images, masks, JSON) as JSON
- GET /results/location/{location_key}: Get all changes for a location ordered by date
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.database.db_manager import db
from backend.database.qdrant_client import qdrant_store
from backend.change_detection.earliest_change import (
    estimate_earliest_change_date,
    compute_temporal_consistency_score
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/results", tags=["Results"])


class ChangeResultResponse(BaseModel):
    """Response model for change result detail."""
    change_id: str
    location_key: str
    before_tile_id: str
    after_tile_id: str
    change_type: str
    change_score: float
    confidence: float
    confidence_level: str
    earliest_change_date: Optional[str]
    evidence_paths: Dict[str, Any]
    registration_shift_px: float
    cloud_mask_quality: float
    model_version: str
    reviews: List[Dict[str, Any]]
    created_at: str


class TemporalTimelineResponse(BaseModel):
    """Response model for temporal timeline."""
    change_id: str
    location_key: str
    earliest_change_date: Optional[str]
    temporal_consistency: float
    timeline: List[Dict[str, Any]]


@router.get("/{change_id}")
async def get_result_detail(change_id: str) -> Dict[str, Any]:
    """
    Retrieve complete change result record with all metadata, evidence paths, and review history.
    
    Args:
        change_id: UUID of the change result
    
    Returns:
        Complete change record including:
        - change_id, location_key, before/after tile IDs
        - change_score (raw model output)
        - confidence (heuristic CCS score)
        - earliest_change_date (from temporal analysis)
        - evidence_paths (image and mask file locations)
        - registration and cloud quality metrics
        - review history (analyst annotations)
    
    Raises:
        HTTPException 404: Change result not found
    """
    try:
        res = db.get_change_result(change_id)
        if not res:
            logger.warning(f"Change result not found: {change_id}")
            raise HTTPException(status_code=404, detail="Change result record not found.")
        
        # Fetch associated reviews
        reviews = db.get_reviews_for_change(change_id)
        res["reviews"] = reviews if reviews else []
        
        logger.debug(f"Retrieved change result: {change_id}")
        return res
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving change result {change_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve change result")


@router.get("/{change_id}/temporal")
async def get_result_temporal_timeline(
    change_id: str,
    min_observations: int = Query(2, ge=1, description="Minimum observations for estimation")
) -> Dict[str, Any]:
    """
    Retrieve temporal timeline for a location and estimate earliest change date.
    
    This endpoint:
    1. Fetches the change result by change_id
    2. Retrieves all tiles for the location_key sorted by date
    3. Builds a timeline of observations with quality metrics
    4. Estimates earliest persistent change date (filtering transient spikes)
    5. Computes temporal consistency score
    
    Args:
        change_id: UUID of the change result
        min_observations: Minimum number of observations required for robust estimation
    
    Returns:
        {
            "change_id": str,
            "location_key": str,
            "earliest_change_date": str (ISO date) or null,
            "temporal_consistency": float [0.0, 1.0],
            "timeline": [
                {
                    "tile_id": str,
                    "date": str (YYYY-MM-DD),
                    "cloud_cover": float,
                    "quality_score": float,
                    "rgb_filepath": str
                },
                ...
            ]
        }
    
    Raises:
        HTTPException 404: Change result not found
    """
    try:
        res = db.get_change_result(change_id)
        if not res:
            raise HTTPException(status_code=404, detail="Change result not found.")
        
        loc_key = res.get("location_key")
        
        # Fetch all tiles for location ordered by date
        tiles = db.get_tiles_by_location(loc_key)
        
        if len(tiles) < min_observations:
            logger.warning(
                f"Insufficient observations for location {loc_key}: "
                f"{len(tiles)} < {min_observations}"
            )
        
        # Build timeline with quality metrics
        timeline = []
        for tile in tiles:
            timeline.append({
                "tile_id": tile.get("tile_id"),
                "date": tile.get("acquisition_datetime", "")[:10],  # ISO date YYYY-MM-DD
                "cloud_cover": float(tile.get("cloud_cover", 0.0)),
                "quality_score": float(tile.get("quality_score", 1.0)),
                "rgb_filepath": tile.get("rgb_filepath", ""),
                "change_score": float(tile.get("change_score", 0.0)) if "change_score" in tile else None
            })
        
        # Estimate earliest change date with persistence check
        if len(timeline) >= 2:
            earliest_date, persistence, _ = estimate_earliest_change_date(
                [
                    {
                        "date": item["date"],
                        "change_score": item.get("change_score", 0.0),
                        "cloud_score": item["cloud_cover"]
                    }
                    for item in timeline
                ]
            )
        else:
            earliest_date = res.get("earliest_change_date")
            persistence = 0.0
        
        # Compute temporal consistency
        if len(timeline) >= 2:
            temporal_consistency = compute_temporal_consistency_score(timeline, "quality_score")
        else:
            temporal_consistency = 1.0
        
        logger.debug(f"Generated temporal timeline for {loc_key}: {len(timeline)} observations")
        
        return {
            "change_id": change_id,
            "location_key": loc_key,
            "earliest_change_date": earliest_date or res.get("earliest_change_date"),
            "temporal_consistency": round(float(temporal_consistency), 2),
            "change_persistence": round(float(persistence), 2),
            "timeline": timeline
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating temporal timeline for {change_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate temporal timeline")


@router.get("/{change_id}/evidence")
async def get_result_evidence_export(change_id: str) -> Dict[str, Any]:
    """
    Export complete evidence package for an analyst review.
    
    This endpoint assembles all evidence components into a JSON-serializable package:
    - Before/after RGB images (as file paths or base64 if small enough)
    - Change detection mask
    - Cloud/quality masks
    - Registration metadata
    - Confidence score breakdown
    - Full change record metadata
    
    This is used for:
    - Analyst manual review workflows
    - External system integrations
    - Audit trails and proof-of-concept documentation
    
    Args:
        change_id: UUID of the change result
    
    Returns:
        {
            "change_id": str,
            "location_key": str,
            "before_tile_id": str,
            "after_tile_id": str,
            "change_type": str,
            "change_score": float,
            "confidence": float,
            "confidence_level": str,
            "earliest_change_date": str,
            "evidence": {
                "before_image_path": str,
                "after_image_path": str,
                "change_mask_path": str,
                "cloud_mask_path": str,
                "registration_metadata": {
                    "shift_y": float,
                    "shift_x": float,
                    "shift_magnitude": float,
                    "coherence": float
                }
            },
            "metadata": {
                "model_version": str,
                "created_at": str,
                "processing_version": str
            },
            "reviews": [...]
        }
    
    Raises:
        HTTPException 404: Change result not found
    """
    try:
        res = db.get_change_result(change_id)
        if not res:
            raise HTTPException(status_code=404, detail="Change result not found.")
        
        evidence_paths = res.get("evidence_paths", {})
        
        # Fetch reviews
        reviews = db.get_reviews_for_change(change_id)
        
        evidence_package = {
            "change_id": res.get("change_id"),
            "location_key": res.get("location_key"),
            "before_tile_id": res.get("before_tile_id"),
            "after_tile_id": res.get("after_tile_id"),
            "change_type": res.get("change_type", "construction"),
            "change_score": float(res.get("change_score", 0.0)),
            "confidence": float(res.get("confidence", 0.0)),
            "confidence_level": _categorize_confidence(float(res.get("confidence", 0.0))),
            "earliest_change_date": res.get("earliest_change_date"),
            "evidence": {
                "before_image_path": evidence_paths.get("before_png", ""),
                "after_image_path": evidence_paths.get("after_png", ""),
                "change_mask_path": evidence_paths.get("change_mask_png", ""),
                "cloud_mask_path": evidence_paths.get("cloud_mask_png", ""),
                "registration_metadata": {
                    "shift_y": float(res.get("registration_shift_px", 0.0)) / np.sqrt(2),
                    "shift_x": float(res.get("registration_shift_px", 0.0)) / np.sqrt(2),
                    "shift_magnitude": float(res.get("registration_shift_px", 0.0)),
                    "coherence": float(res.get("registration_quality", 0.0))
                }
            },
            "metadata": {
                "model_version": res.get("model_version", "unknown"),
                "semantic_retrieval": "CLIP-RSICD v2 (512-d)",
                "spectral_representation": "Clay Foundation v1.5 (768-d)",
                "change_detector": res.get("model_version", "Open-CD SNUNet"),
                "cloud_masking": "s2cloudless",
                "co_registration": "AROSICS Phase Correlation",
                "created_at": res.get("created_at", datetime.now().isoformat()),
                "processing_version": res.get("processing_version", "v1.0.0")
            },
            "ccs_breakdown": res.get("ccs_breakdown", {}),
            "reviews": reviews if reviews else []
        }
        
        logger.info(f"Exported evidence package for change {change_id}")
        return evidence_package
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting evidence for {change_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to export evidence package")


@router.get("/{change_id}/similar")
async def get_similar_sites(
    change_id: str,
    top_k: int = Query(10, ge=1, le=50)
) -> Dict[str, Any]:
    """Find nearby semantic matches using the selected tile's stored embedding."""
    source_tile = db.get_tile(change_id)
    source_change_id = change_id
    if not source_tile:
        source_result = db.get_change_result(change_id)
        if not source_result:
            raise HTTPException(status_code=404, detail="Tile or change result not found.")
        source_tile = db.get_tile(source_result.get("after_tile_id", ""))
        if not source_tile:
            raise HTTPException(status_code=404, detail="Source tile not found.")

    source_tile_id = source_tile["tile_id"]
    stored = qdrant_store.get_semantic_tile(source_tile_id)
    if not stored:
        raise HTTPException(status_code=404, detail="Source semantic embedding not found.")

    candidates = qdrant_store.search_semantic(stored["vector"], top_k=top_k + 1)
    results = []
    for candidate in candidates:
        if candidate["id"] == source_tile_id:
            continue
        payload = candidate.get("payload", {})
        location_changes = db.get_changes_by_location(payload.get("location_key", ""), limit=1)
        latest_change = location_changes[0] if location_changes else {}
        results.append({
            "tile_id": payload.get("tile_id", candidate["id"]),
            "change_id": latest_change.get("change_id"),
            "location_key": payload.get("location_key"),
            "similarity": round(float(candidate.get("score", 0.0)), 4),
            "change_confidence": latest_change.get("confidence", 0.0),
            "confidence_level": _categorize_confidence(float(latest_change.get("confidence", 0.0))),
            "change_type": latest_change.get("change_type", "similar_site"),
            "earliest_change_date": latest_change.get("earliest_change_date"),
            "acquisition_dates": [str(payload.get("acquisition_datetime", ""))[:10]],
            "sensor": payload.get("sensor", "Sentinel-2 L2A"),
            "coordinates": [payload.get("longitude", 0.0), payload.get("latitude", 0.0)],
            "thumbnail_path": payload.get("rgb_filepath", ""),
            "evidence_paths": latest_change.get("evidence_paths", {}),
            "source_change_id": source_change_id
        })
        if len(results) >= top_k:
            break

    return {
        "source_tile_id": source_tile_id,
        "source_location_key": source_tile.get("location_key"),
        "results_count": len(results),
        "results": results
    }


@router.get("/location/{location_key}")
async def get_location_changes(
    location_key: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """
    Retrieve all changes for a specific location, ordered chronologically.
    
    Useful for viewing the change history and detecting trends at a fixed location.
    
    Args:
        location_key: Location grid cell identifier
        limit: Maximum number of results to return (default 20, max 100)
        offset: Result offset for pagination (default 0)
    
    Returns:
        {
            "location_key": str,
            "total_count": int,
            "changes": [
                {
                    "change_id": str,
                    "earliest_change_date": str,
                    "change_type": str,
                    "confidence": float,
                    "created_at": str
                },
                ...
            ]
        }
    """
    try:
        changes = db.get_changes_by_location(location_key, limit=limit, offset=offset)
        total_count = db.count_changes_by_location(location_key)
        
        return {
            "location_key": location_key,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "changes": changes if changes else []
        }
    
    except Exception as e:
        logger.error(f"Error retrieving changes for location {location_key}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve location changes")


def _categorize_confidence(confidence: float) -> str:
    """Helper: categorize CCS into confidence level."""
    if confidence >= 0.70:
        return "high_confidence"
    elif confidence >= 0.40:
        return "review_flagged"
    else:
        return "low_confidence"


import numpy as np

def export_evidence_package(change_id: str):
    """
    Evidence Export package as specified in Section 27.
    """
    res = db.get_change_result(change_id)
    if not res:
        raise HTTPException(status_code=404, detail="Change result not found.")

    reviews = db.get_reviews_for_change(change_id)

    return {
        "export_format": "PS26227_EVIDENCE_PACKAGE_V1",
        "change_id": change_id,
        "location_key": res.get("location_key"),
        "change_type": res.get("change_type"),
        "change_confidence_score": res.get("confidence"),
        "raw_change_score": res.get("change_score"),
        "earliest_change_date": res.get("earliest_change_date"),
        "registration_shift_px": res.get("registration_shift_px"),
        "cloud_mask_quality": res.get("cloud_mask_quality"),
        "model_provenance": {
            "model_version": res.get("model_version"),
            "semantic_retrieval": "CLIP-RSICD v2 (512-d)",
            "spectral_representation": "Clay Foundation v1.5 (768-d)",
            "change_detector": "Open-CD SNUNet-LEVIR-CD",
            "cloud_masking": "s2cloudless",
            "co_registration": "AROSICS Phase Correlation"
        },
        "evidence_paths": res.get("evidence_paths"),
        "review_audit_trail": reviews
    }
