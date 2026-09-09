"""
PS26227 - Earth Observation Semantic Search & Multi-Temporal Change Detection
FastAPI Backend Application Entrypoint
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import TILES_DIR, DATA_DIR, API_HOST, API_PORT
from backend.api.health import router as health_router
from backend.api.search import router as search_router
from backend.api.results import router as results_router
from backend.api.review import router as review_router
from backend.api.ingest import router as ingest_router
from backend.api.tiles import router as tiles_router

app = FastAPI(
    title="PS26227 EO Semantic Search & Multi-Temporal Change API",
    description="Full implementation of SIH PS26227 specification for multi-sensor retrieval and change detection",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve generated tile images, masks, and evidence to MapLibre & React UI
os.makedirs(str(TILES_DIR), exist_ok=True)
app.mount("/static/tiles", StaticFiles(directory=str(TILES_DIR)), name="tiles")
app.mount("/static/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# Register Routers
app.include_router(health_router)
app.include_router(search_router)
app.include_router(results_router)
app.include_router(review_router)
app.include_router(ingest_router)
app.include_router(tiles_router)

@app.get("/")
def root():
    return {
        "project": "PS26227 - Earth Observation Semantic Retrieval & Change Detection",
        "docs_url": "/docs",
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
