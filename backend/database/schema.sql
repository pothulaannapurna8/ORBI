-- ====================================================================
-- PS26227 - Earth Observation Semantic Retrieval & Change Detection
-- PostgreSQL + PostGIS Schema (pgSTAC Compatible)
-- ====================================================================

-- 1. Enable Extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Tile Table
-- Stores geographic, sensor, quality, and vector reference metadata for each 512x512 tile
CREATE TABLE IF NOT EXISTS tile (
    tile_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    geometry             GEOMETRY(POLYGON, 4326) NOT NULL,
    latitude             DOUBLE PRECISION NOT NULL,
    longitude            DOUBLE PRECISION NOT NULL,
    sensor               TEXT NOT NULL DEFAULT 'Sentinel-2 L2A',
    acquisition_datetime TIMESTAMPTZ NOT NULL,
    cloud_cover          REAL DEFAULT 0.0,
    source               TEXT NOT NULL,
    filepath             TEXT NOT NULL,
    rgb_filepath         TEXT NOT NULL,
    embedding_id_semantic TEXT,                               -- Qdrant ID in 'semantic_tiles' collection (512-d)
    embedding_id_spectral TEXT,                               -- Qdrant ID in 'spectral_tiles' collection (768-d)
    quality_score        REAL DEFAULT 1.0,                    -- Combined s2cloudless & registration metric
    processing_version   TEXT NOT NULL DEFAULT 'v1.0.0',
    location_key         TEXT NOT NULL,                       -- Snapped regular grid cell ID (e.g. loc_grid_12_77)
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- Indices for spatial, temporal, and join performance
CREATE INDEX IF NOT EXISTS idx_tile_geom ON tile USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_tile_location_date ON tile (location_key, acquisition_datetime ASC);
CREATE INDEX IF NOT EXISTS idx_tile_semantic_id ON tile (embedding_id_semantic);
CREATE INDEX IF NOT EXISTS idx_tile_spectral_id ON tile (embedding_id_spectral);

-- 3. Multi-Temporal Change Result Table
-- Stores pairwise and longitudinal change evidence, masks, confidence scores, and provenance
CREATE TABLE IF NOT EXISTS change_result (
    change_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location_key          TEXT NOT NULL,
    before_tile_id        UUID REFERENCES tile(tile_id) ON DELETE CASCADE,
    after_tile_id         UUID REFERENCES tile(tile_id) ON DELETE CASCADE,
    earliest_change_date  DATE,
    change_type           TEXT NOT NULL,                      -- 'construction' | 'water_extent' | 'clearance' | 'no_change'
    change_score          REAL NOT NULL,                      -- Raw model output (0.0 to 1.0)
    confidence            REAL NOT NULL,                      -- Heuristic Change Confidence Score (CCS) [0.0 to 1.0]
    evidence_paths        JSONB NOT NULL DEFAULT '{}'::jsonb, -- {"before_png": "...", "after_png": "...", "mask_png": "..."}
    registration_shift_px REAL DEFAULT 0.0,                   -- Diagnostic sub-pixel shift from AROSICS
    cloud_mask_quality    REAL DEFAULT 1.0,                   -- Diagnostic clear-sky fraction from s2cloudless
    registration_quality  REAL DEFAULT 0.0,
    ccs_breakdown         JSONB,
    processing_version    TEXT,
    model_version         TEXT NOT NULL,                      -- Pinned model version (e.g. SNUNet-LEVIR-CD-v1.2)
    created_at            TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_change_location ON change_result (location_key);
CREATE INDEX IF NOT EXISTS idx_change_confidence ON change_result (confidence DESC);
CREATE INDEX IF NOT EXISTS idx_change_type ON change_result (change_type);

-- 4. Review Log Table
-- Audit trail for human analyst confirmation, rejection, and commentary
CREATE TABLE IF NOT EXISTS review_log (
    review_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    change_id    UUID REFERENCES change_result(change_id) ON DELETE CASCADE,
    decision     TEXT NOT NULL CHECK (decision IN ('confirmed', 'rejected')),
    analyst_note TEXT,
    analyst_id   TEXT,
    confidence_override REAL,
    reviewed_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_review_change_id ON review_log (change_id);
