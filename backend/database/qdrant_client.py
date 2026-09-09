"""
Qdrant Vector Database Client for PS26227
Manages two distinct, un-concatenated collections:
1. 'semantic_tiles': 512-dimensional CLIP-RSICD embeddings for text & image similarity
2. 'spectral_tiles': 768-dimensional Clay Foundation embeddings for multi-spectral change representation
Seamlessly connects to Docker Qdrant or local disk storage.
"""
import os
import uuid
import logging
import math
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from backend.config import (
    USE_LOCAL_QDRANT, QDRANT_HOST, QDRANT_PORT,
    QDRANT_LOCAL_PATH, QDRANT_SEMANTIC_COLLECTION, QDRANT_SPECTRAL_COLLECTION
)

logger = logging.getLogger(__name__)

class QdrantStore:
    def __init__(self):
        self.semantic_collection = QDRANT_SEMANTIC_COLLECTION
        self.spectral_collection = QDRANT_SPECTRAL_COLLECTION
        self.client = None
        self._init_client()
        self._ensure_collections()

    def _init_client(self):
        if not USE_LOCAL_QDRANT:
            try:
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
                # Verify connectivity
                self.client.get_collections()
                return
            except Exception as exc:
                logger.warning("Remote Qdrant unavailable; using local storage: %s", exc)
        
        # Fallback to local persistent directory mode (no Docker needed)
        os.makedirs(str(QDRANT_LOCAL_PATH), exist_ok=True)
        self.client = QdrantClient(path=str(QDRANT_LOCAL_PATH))

    def _ensure_collections(self):
        # 1. Semantic Collection (512-d CLIP-RSICD)
        collections = [c.name for c in self.client.get_collections().collections]
        if self.semantic_collection not in collections:
            self.client.create_collection(
                collection_name=self.semantic_collection,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )

        # 2. Spectral Collection (768-d Clay v1.5)
        if self.spectral_collection not in collections:
            self.client.create_collection(
                collection_name=self.spectral_collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

        for collection_name in (self.semantic_collection, self.spectral_collection):
            count = self.client.count(collection_name=collection_name, exact=True).count
            if count == 0:
                logger.warning("Qdrant collection '%s' has no points", collection_name)
            else:
                logger.info("Qdrant collection '%s' contains %d points", collection_name, count)

    @staticmethod
    def _validate_vector(vector: List[float], expected_size: int, label: str) -> List[float]:
        values = list(vector)
        if len(values) != expected_size:
            raise ValueError(f"{label} vector has dimension {len(values)}; expected {expected_size}")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"{label} vector contains non-finite values")
        if not any(float(value) != 0.0 for value in values):
            raise ValueError(f"{label} vector is all zeros")
        return values

    def upsert_semantic_tile(self, point_id: str, vector: List[float], payload: Dict[str, Any]):
        vector = self._validate_vector(vector, 512, "semantic")
        self.client.upsert(
            collection_name=self.semantic_collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
        )

    def upsert_spectral_tile(self, point_id: str, vector: List[float], payload: Dict[str, Any]):
        vector = self._validate_vector(vector, 768, "spectral")
        self.client.upsert(
            collection_name=self.spectral_collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
        )

    def search_semantic(
        self,
        query_vector: List[float],
        top_k: int = 20,
        sensor: Optional[str] = None,
        location_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_vector = self._validate_vector(query_vector, 512, "semantic query")
        query_filter = None
        conditions = []
        if sensor:
            conditions.append(FieldCondition(key="sensor", match=MatchValue(value=sensor)))
        if location_key:
            conditions.append(FieldCondition(key="location_key", match=MatchValue(value=location_key)))
        if conditions:
            query_filter = Filter(must=conditions)

        point_count = self.client.count(collection_name=self.semantic_collection, exact=True).count
        if point_count == 0:
            logger.warning("Semantic search requested but collection '%s' is empty", self.semantic_collection)
            return []

        results = self.client.query_points(
            collection_name=self.semantic_collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        ).points

        formatted = []
        for r in results:
            formatted.append({
                "id": str(r.id),
                "score": float(r.score) if r.score is not None else 0.0,
                "payload": r.payload or {}
            })
        return formatted

    def search_spectral(
        self,
        query_vector: List[float],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        results = self.client.query_points(
            collection_name=self.spectral_collection,
            query=query_vector,
            limit=top_k
        ).points
        return [{
            "id": str(r.id),
            "score": float(r.score) if r.score is not None else 0.0,
            "payload": r.payload or {}
        } for r in results]

# Global singleton
qdrant_store = QdrantStore()
