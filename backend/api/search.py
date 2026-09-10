"""
Semantic Retrieval APIs: Text Search, Image Search, and Multimodal Search
Integrates CLIP-RSICD dual-tower embeddings, Qdrant kNN search,
spatial/temporal resolution via PostGIS/SQLite, and Change Confidence reranking.
"""
import io
import logging
import re
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np

from backend.embeddings.clip_rsicd_encoder import clip_encoder
from backend.database.qdrant_client import qdrant_store
from backend.database.db_manager import db
from backend.quality.arosics_register import arosics_aligner
from backend.change_detection.open_cd_wrapper import open_cd_detector
from backend.change_detection.ndwi_water import ndwi_detector
from backend.change_detection.change_confidence import compute_change_confidence_score
from backend.change_detection.earliest_change import estimate_earliest_change_date
from backend.change_detection.landcover_breakdown import compute_landcover_breakdown
from backend.config import RANKING_WEIGHT_SEMANTIC, RANKING_WEIGHT_CHANGE

router = APIRouter(prefix="/search", tags=["Search"])


def display_similarity(raw_similarity: float) -> float:
    """Map cosine similarity from [-1, 1] to an auditable [0, 1] display score."""
    return round((max(-1.0, min(1.0, float(raw_similarity))) + 1.0) / 2.0, 4)
logger = logging.getLogger(__name__)

class TextSearchRequest(BaseModel):
    query: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    sensor: Optional[str] = None
    top_k: Optional[int] = 20
    limit: Optional[int] = None

def filter_candidates_by_date(
    candidates: List[Dict[str, Any]],
    date_from: Optional[str],
    date_to: Optional[str]
) -> List[Dict[str, Any]]:
    """Apply acquisition-date bounds without filtering by confidence score."""
    if not date_from and not date_to:
        return candidates

    filtered = []
    for candidate in candidates:
        acquisition_date = str(candidate.get("payload", {}).get("acquisition_datetime", ""))[:10]
        if not acquisition_date:
            continue
        if date_from and acquisition_date < date_from:
            continue
        if date_to and acquisition_date > date_to:
            continue
        filtered.append(candidate)
    return filtered

def parse_query_rules(query: str) -> Dict[str, Any]:
    """
    Lightweight rule-based query parser:
    Extracts change intent, object classes, and spatial context keywords.
    """
    q_lower = query.lower()
    change_keywords = ["new", "newly", "built", "construction", "appeared", "between", "developed", "expansion", "lost", "change"]
    is_change_query = any(k in q_lower for k in change_keywords)

    water_context = any(k in q_lower for k in ["river", "water", "lake", "reservoir", "coast"])
    building_context = any(k in q_lower for k in ["structure", "building", "urban", "house", "facility", "construction"])

    # Extract potential 4-digit years (e.g. 2024, 2025, 2026)
    years = re.findall(r"\b(20\d\d)\b", query)

    return {
        "is_change_query": is_change_query,
        "water_context": water_context,
        "building_context": building_context,
        "detected_years": sorted(list(set(years)))
    }

def run_temporal_analysis_for_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For candidate locations, fetch temporal history, compute before/after registration,
    change detection, NDWI water typing, earliest change date, and Change Confidence Score (CCS).
    """
    results = []
    seen_locations = set()

    for cand in candidates:
        payload = cand.get("payload", {})
        loc_key = payload.get("location_key")
        if not loc_key or loc_key in seen_locations:
            continue
        seen_locations.add(loc_key)

        raw_similarity = float(cand.get("score", 0.0))
        similarity = display_similarity(raw_similarity)
        location_changes = db.get_changes_by_location(loc_key, limit=1)
        latest_change = location_changes[0] if location_changes else {}
        # Fetch all available dates for this location
        tiles = db.get_tiles_by_location(loc_key)
        if len(tiles) < 2:
            # Single date candidate: surface with baseline confidence
            results.append({
                "tile_id": payload.get("tile_id"),
                "change_id": latest_change.get("change_id"),
                "location_key": loc_key,
                "similarity": similarity,
                "raw_similarity": round(raw_similarity, 4),
                "change_confidence": 0.0,
                "confidence_level": "suppressed",
                "change_type": "single_observation",
                "earliest_change_date": payload.get("acquisition_datetime", "")[:10],
                "acquisition_dates": [payload.get("acquisition_datetime", "")[:10]],
                "sensor": payload.get("sensor", "Sentinel-2 L2A"),
                "coordinates": [payload.get("longitude", 0.0), payload.get("latitude", 0.0)],
                "evidence_paths": {
                    "current_png": payload.get("rgb_filepath", "")
                },
                "provenance": {
                    "model_version": "N/A (Single Date)",
                    "registration_shift_px": 0.0,
                    "cloud_mask_quality": round(float(payload.get("quality_score", 1.0)), 2)
                },
                "reranked_score": round(similarity, 4)
            })
            continue

        ref_tile = tiles[0]
        latest_tile = tiles[-1]

        # Load RGB images for change analysis
        try:
            im_before = np.array(Image.open(ref_tile["rgb_filepath"]).convert("RGB"))
            im_after = np.array(Image.open(latest_tile["rgb_filepath"]).convert("RGB"))

            # 1. AROSICS Sub-pixel Registration
            im_reg, shift_px, reg_quality = arosics_aligner.register(im_before, im_after)

            # 2. Open-CD Binary Change Detection
            change_mask, raw_evidence = open_cd_detector.detect_change(im_before, im_reg)

            landcover_breakdown = compute_landcover_breakdown(
                ref_tile.get("filepath", ""),
                latest_tile.get("filepath", ""),
                change_mask,
            )

            # 3. Water-Extent NDWI Analysis
            # Green (channel 1), NIR (approximated from red-channel contrast when in RGB)
            water_analysis = ndwi_detector.compute_water_extent_change(
                im_before[:, :, 1], im_before[:, :, 0],
                im_reg[:, :, 1], im_reg[:, :, 0],
                red_before=im_before[:, :, 0],
                red_after=im_reg[:, :, 0]
            )
            water_score = float(water_analysis["water_change_score"])
            water_delta = float(water_analysis["water_delta"])
            water_type = "water_expansion" if water_delta >= 0 else "water_retreat"

            # 4. Multi-Temporal Timeline & Earliest Change Date
            timeline = []
            for t in tiles:
                t_img = np.array(Image.open(t["rgb_filepath"]).convert("RGB"))
                _, t_ev = open_cd_detector.detect_change(im_before, t_img)
                timeline.append({
                    "date": t["acquisition_datetime"][:10],
                    "change_score": t_ev,
                    "cloud_score": float(t.get("cloud_cover", 0.0))
                })

            earliest_date, temp_consistency, _ = estimate_earliest_change_date(timeline)

            # 5. Change Confidence Score (CCS)
            cloud_contam = float(latest_tile.get("cloud_cover", 0.0))
            ccs, ccs_category = compute_change_confidence_score(
                change_evidence=raw_evidence,
                cloud_score=cloud_contam,
                registration_quality=reg_quality,
                temporal_consistency=temp_consistency
            )

            # Determine primary change type
            primary_change = "construction"
            if water_score > 0.35 and water_score > raw_evidence:
                primary_change = water_type
            elif ccs < 0.40:
                primary_change = "no_change"

            # 6. Reranking formula: 0.5 * similarity + 0.5 * CCS
            final_rank_score = (RANKING_WEIGHT_SEMANTIC * similarity) + (RANKING_WEIGHT_CHANGE * ccs)

            # Confidence Level
            conf_level = "high_confidence" if ccs >= 0.70 else ("needs_review" if ccs >= 0.40 else "suppressed")

            results.append({
                "tile_id": latest_tile.get("tile_id"),
                "change_id": latest_change.get("change_id"),
                "location_key": loc_key,
                "similarity": round(similarity, 4),
                "raw_similarity": round(raw_similarity, 4),
                "change_confidence": round(ccs, 4),
                "confidence_level": conf_level,
                "change_type": primary_change,
                "earliest_change_date": earliest_date,
                "acquisition_dates": [ref_tile["acquisition_datetime"][:10], latest_tile["acquisition_datetime"][:10]],
                "sensor": latest_tile.get("sensor", "Sentinel-2 L2A"),
                "coordinates": [latest_tile.get("longitude", 0.0), latest_tile.get("latitude", 0.0)],
                "evidence_paths": {
                    "before_png": ref_tile.get("rgb_filepath", ""),
                    "after_png": latest_tile.get("rgb_filepath", ""),
                    "mask_png": "" # dynamically served or pre-rendered
                },
                "provenance": {
                    "model_version": "SNUNet-v1.2",
                    "registration_shift_px": round(shift_px, 2),
                    "cloud_mask_quality": round(1.0 - cloud_contam, 2)
                },
                "ccs_breakdown": {
                    "change_evidence": round(float(raw_evidence), 4),
                    "cloud_score": round(cloud_contam, 4),
                    "registration_quality": round(reg_quality, 4),
                    "temporal_consistency": round(float(temp_consistency), 4)
                },
                "landcover_breakdown": landcover_breakdown,
                "reranked_score": round(final_rank_score, 4)
            })

        except Exception:
            logger.exception(
                "Temporal analysis failed for candidate tile=%s location=%s",
                payload.get("tile_id"),
                loc_key,
            )
            continue

    # Sort results by final reranked score descending
    results.sort(key=lambda x: x["reranked_score"], reverse=True)
    return results

@router.post("/text")
def search_text(req: TextSearchRequest):
    """
    Natural-language text retrieval endpoint.
    Parses intent, generates CLIP text embedding, searches Qdrant,
    resolves candidates through temporal pipeline, and reranks by CCS.
    """
    requested_top_k = req.limit or req.top_k or 20
    if not 1 <= requested_top_k <= 100:
        raise HTTPException(status_code=400, detail="limit/top_k must be between 1 and 100")

    parsed = parse_query_rules(req.query)
    # 1. Encode text via CLIP-RSICD
    query_vec = clip_encoder.encode_text(req.query)

    # 2. Qdrant kNN search
    candidates = qdrant_store.search_semantic(
        query_vector=query_vec,
        top_k=requested_top_k,
        sensor=req.sensor
    )

    candidates = filter_candidates_by_date(
        candidates,
        req.date_start or req.date_from,
        req.date_end or req.date_to
    )

    # 3. Multi-temporal change analysis & reranking
    ranked_results = run_temporal_analysis_for_candidates(candidates)

    return {
        "query": req.query,
        "parsed_metadata": parsed,
        "results_count": len(ranked_results),
        "results": ranked_results
    }

@router.post("/image")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(20),
    sensor: Optional[str] = Form(None),
    date_start: Optional[str] = Form(None),
    date_end: Optional[str] = Form(None)
):
    """
    Satellite image query endpoint.
    Validates uploaded imagery, normalizes to RGB, encodes with CLIP-RSICD,
    and performs semantic retrieval with multi-temporal change resolution.
    """
    try:
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format. Expected PNG, JPEG, or COG/GeoTIFF.")

    # 1. Encode image via CLIP-RSICD
    query_vec = clip_encoder.encode_image(pil_image)

    # 2. Qdrant kNN search
    candidates = qdrant_store.search_semantic(
        query_vector=query_vec,
        top_k=top_k,
        sensor=sensor
    )

    candidates = filter_candidates_by_date(candidates, date_start, date_end)

    # 3. Temporal analysis & reranking
    ranked_results = run_temporal_analysis_for_candidates(candidates)

    return {
        "query_type": "image_search",
        "filename": file.filename,
        "results_count": len(ranked_results),
        "results": ranked_results
    }

@router.post("/multimodal")
async def search_multimodal(
    query: str = Form(...),
    file: UploadFile = File(...),
    top_k: int = Form(20),
    sensor: Optional[str] = Form(None),
    date_start: Optional[str] = Form(None),
    date_end: Optional[str] = Form(None)
):
    """
    Multimodal (Text + Image) retrieval endpoint.
    Separately encodes text and uploaded image in the same 512-d CLIP-RSICD space,
    applies weighted averaging, normalizes, and searches Qdrant.
    """
    try:
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid uploaded image file.")

    # 1. Weighted multimodal embedding blend
    blend_vec = clip_encoder.encode_multimodal(
        text=query,
        image=pil_image,
        text_weight=0.5,
        image_weight=0.5
    )

    # 2. Qdrant kNN search
    candidates = qdrant_store.search_semantic(
        query_vector=blend_vec,
        top_k=top_k,
        sensor=sensor
    )

    candidates = filter_candidates_by_date(candidates, date_start, date_end)

    # 3. Temporal change analysis & reranking
    ranked_results = run_temporal_analysis_for_candidates(candidates)

    return {
        "query_type": "multimodal_search",
        "text_query": query,
        "filename": file.filename,
        "results_count": len(ranked_results),
        "results": ranked_results
    }
