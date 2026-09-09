"""
Tile Metadata API Endpoint
Returns full Section 5 metadata record for a requested tile ID.
"""
from fastapi import APIRouter, HTTPException
from backend.database.db_manager import db

router = APIRouter(prefix="/tiles", tags=["Tiles"])

@router.get("/{tile_id}")
def get_tile_metadata(tile_id: str):
    tile = db.get_tile(tile_id)
    if not tile:
        raise HTTPException(status_code=404, detail="Tile record not found.")
    return tile
