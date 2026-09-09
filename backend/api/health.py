"""
Health Check Endpoint
Exposes service connectivity, vector store state, and loaded models.
"""
from fastapi import APIRouter
from backend.database.qdrant_client import qdrant_store
from backend.database.db_manager import db
from backend.embeddings.clip_rsicd_encoder import clip_encoder

router = APIRouter(tags=["Health"])

@router.get("/health")
def get_health():
    qdrant_status = "up"
    try:
        qdrant_store.client.get_collections()
    except Exception as e:
        qdrant_status = f"degraded ({str(e)})"

    db_status = "postgres_up" if db.use_postgres else "sqlite_local_up"

    return {
        "status": "ok",
        "database": db_status,
        "qdrant": qdrant_status,
        "clip_rsicd_dim": clip_encoder.dim,
        "models_loaded": True,
        "version": "1.0.0"
    }
