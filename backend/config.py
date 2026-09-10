"""
Global System Configuration for PS26227
"""
import os
from pathlib import Path

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TILES_DIR = DATA_DIR / "tiles"
WEIGHTS_DIR = DATA_DIR / "weights"
INCOMING_DIR = DATA_DIR / "incoming"
QDRANT_STORAGE_DIR = Path(os.getenv("QDRANT_STORAGE_DIR", str(DATA_DIR / "qdrant_storage")))

# Create directories if not present
for d in [DATA_DIR, RAW_DIR, TILES_DIR, WEIGHTS_DIR, INCOMING_DIR, QDRANT_STORAGE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Database settings
USE_LOCAL_SQLITE = os.getenv("USE_LOCAL_SQLITE_FALLBACK", "true").lower() == "true"
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "sih_eo_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgrespassword")
SQLITE_DB_PATH = DATA_DIR / "metadata.db"

# Qdrant Vector DB
USE_LOCAL_QDRANT = os.getenv("USE_LOCAL_QDRANT_STORAGE", "true").lower() == "true"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_SEMANTIC_COLLECTION = os.getenv("QDRANT_SEMANTIC_COLLECTION", "semantic_tiles")
QDRANT_SPECTRAL_COLLECTION = os.getenv("QDRANT_SPECTRAL_COLLECTION", "spectral_tiles")
QDRANT_LOCAL_PATH = Path(os.getenv("QDRANT_LOCAL_PATH", str(QDRANT_STORAGE_DIR)))
QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)

# Model settings
CLIP_RSICD_MODEL = os.getenv("CLIP_RSICD_MODEL", "flax-community/clip-rsicd-v2")
CLAY_WEIGHTS_PATH = os.getenv("CLAY_WEIGHTS_PATH", str(WEIGHTS_DIR / "clay_v1_5.pt"))
OPEN_CD_CHECKPOINT = os.getenv("OPEN_CD_CHECKPOINT", str(WEIGHTS_DIR / "open_cd_snunet.pt"))
OPEN_CD_MODEL_TYPE = os.getenv("OPEN_CD_MODEL_TYPE", "snunet")

# Thresholds & Weights
TILE_SIZE_PX = int(os.getenv("TILE_SIZE_PX", 512))
TILE_GSD_METERS = float(os.getenv("TILE_GSD_METERS", 10.0))
CLOUD_PROB_THRESHOLD = float(os.getenv("CLOUD_PROB_THRESHOLD", 0.40))
MAX_REGISTRATION_TOLERANCE_PX = float(os.getenv("MAX_REGISTRATION_TOLERANCE_PX", 5.0))

# Change Confidence Score Weights (must sum to 1.0)
CCS_W1_CHANGE = 0.40
CCS_W2_CLEAR_SKY = 0.20
CCS_W3_REGISTRATION = 0.20
CCS_W4_CONSISTENCY = 0.20

# Ranking Weights
RANKING_WEIGHT_SEMANTIC = float(os.getenv("RANKING_WEIGHT_SEMANTIC", 0.50))
RANKING_WEIGHT_CHANGE = float(os.getenv("RANKING_WEIGHT_CHANGE", 0.50))

# Multimodal blend weights
MULTIMODAL_WEIGHT_TEXT = float(os.getenv("MULTIMODAL_WEIGHT_TEXT", 0.50))
MULTIMODAL_WEIGHT_IMAGE = float(os.getenv("MULTIMODAL_WEIGHT_IMAGE", 0.50))
