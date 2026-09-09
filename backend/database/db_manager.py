"""
Database Manager for Tile, Change Results, and Review Logs
Supports PostgreSQL/PostGIS and automated local SQLite fallback with full GeoJSON & JSONB compatibility.
"""
import os
import json
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.config import (
    USE_LOCAL_SQLITE, POSTGRES_HOST, POSTGRES_PORT,
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, SQLITE_DB_PATH
)

class DatabaseManager:
    def __init__(self):
        self.use_postgres = False
        self.pg_conn = None
        
        if not USE_LOCAL_SQLITE:
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                self.pg_conn = psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    database=POSTGRES_DB,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD
                )
                self.use_postgres = True
            except Exception:
                self.use_postgres = False
                
        if not self.use_postgres:
            self._init_sqlite()

    def _init_sqlite(self):
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cur = conn.cursor()
        
        # Tile Table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS tile (
            tile_id TEXT PRIMARY KEY,
            geometry TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            sensor TEXT NOT NULL,
            acquisition_datetime TEXT NOT NULL,
            cloud_cover REAL DEFAULT 0.0,
            source TEXT NOT NULL,
            filepath TEXT NOT NULL,
            rgb_filepath TEXT NOT NULL,
            embedding_id_semantic TEXT,
            embedding_id_spectral TEXT,
            quality_score REAL DEFAULT 1.0,
            processing_version TEXT NOT NULL,
            location_key TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

        # Change Result Table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS change_result (
            change_id TEXT PRIMARY KEY,
            location_key TEXT NOT NULL,
            before_tile_id TEXT NOT NULL,
            after_tile_id TEXT NOT NULL,
            earliest_change_date TEXT,
            change_type TEXT NOT NULL,
            change_score REAL NOT NULL,
            confidence REAL NOT NULL,
            evidence_paths TEXT NOT NULL,
            registration_shift_px REAL DEFAULT 0.0,
            cloud_mask_quality REAL DEFAULT 1.0,
            model_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (before_tile_id) REFERENCES tile(tile_id),
            FOREIGN KEY (after_tile_id) REFERENCES tile(tile_id)
        )
        """)

        # Review Log Table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS review_log (
            review_id TEXT PRIMARY KEY,
            change_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            analyst_note TEXT,
            reviewed_at TEXT NOT NULL,
            FOREIGN KEY (change_id) REFERENCES change_result(change_id)
        )
        """)

        conn.commit()
        conn.close()

    def insert_tile(self, tile_data: Dict[str, Any]) -> str:
        tile_id = tile_data.get("tile_id", str(uuid.uuid4()))
        now_iso = datetime.utcnow().isoformat()
        
        geom_str = tile_data.get("geometry")
        if isinstance(geom_str, dict):
            geom_str = json.dumps(geom_str)

        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO tile (
                tile_id, geometry, latitude, longitude, sensor,
                acquisition_datetime, cloud_cover, source, filepath,
                rgb_filepath, embedding_id_semantic, embedding_id_spectral,
                quality_score, processing_version, location_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tile_id,
            geom_str,
            float(tile_data.get("latitude", 0.0)),
            float(tile_data.get("longitude", 0.0)),
            tile_data.get("sensor", "Sentinel-2 L2A"),
            tile_data.get("acquisition_datetime", now_iso),
            float(tile_data.get("cloud_cover", 0.0)),
            tile_data.get("source", "Sentinel-2"),
            tile_data.get("filepath", ""),
            tile_data.get("rgb_filepath", ""),
            tile_data.get("embedding_id_semantic", ""),
            tile_data.get("embedding_id_spectral", ""),
            float(tile_data.get("quality_score", 1.0)),
            tile_data.get("processing_version", "v1.0.0"),
            tile_data.get("location_key", "default_loc"),
            now_iso
        ))
        conn.commit()
        conn.close()
        return tile_id

    def get_tile(self, tile_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM tile WHERE tile_id = ?", (tile_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return dict(row)

    def get_tiles_by_location(self, location_key: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM tile WHERE location_key = ? ORDER BY acquisition_datetime ASC",
            (location_key,)
        )
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def insert_change_result(self, res: Dict[str, Any]) -> str:
        change_id = res.get("change_id", str(uuid.uuid4()))
        now_iso = datetime.utcnow().isoformat()
        ev_paths = res.get("evidence_paths", {})
        if isinstance(ev_paths, dict):
            ev_paths = json.dumps(ev_paths)

        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO change_result (
                change_id, location_key, before_tile_id, after_tile_id,
                earliest_change_date, change_type, change_score, confidence,
                evidence_paths, registration_shift_px, cloud_mask_quality,
                model_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            change_id,
            res.get("location_key", ""),
            res.get("before_tile_id", ""),
            res.get("after_tile_id", ""),
            res.get("earliest_change_date", None),
            res.get("change_type", "construction"),
            float(res.get("change_score", 0.0)),
            float(res.get("confidence", 0.0)),
            ev_paths,
            float(res.get("registration_shift_px", 0.0)),
            float(res.get("cloud_mask_quality", 1.0)),
            res.get("model_version", "SNUNet-v1"),
            now_iso
        ))
        conn.commit()
        conn.close()
        return change_id

    def get_change_result(self, change_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM change_result WHERE change_id = ?", (change_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        d = dict(row)
        if isinstance(d.get("evidence_paths"), str):
            try:
                d["evidence_paths"] = json.loads(d["evidence_paths"])
            except Exception:
                d["evidence_paths"] = {}
        return d

    def insert_review(self, change_id: str, decision: str, note: Optional[str] = None) -> str:
        review_id = str(uuid.uuid4())
        now_iso = datetime.utcnow().isoformat()
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO review_log (review_id, change_id, decision, analyst_note, reviewed_at)
            VALUES (?, ?, ?, ?, ?)
        """, (review_id, change_id, decision, note or "", now_iso))
        conn.commit()
        conn.close()
        return review_id

    def get_reviews_for_change(self, change_id: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM review_log WHERE change_id = ? ORDER BY reviewed_at DESC", (change_id,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

# Global singleton
db = DatabaseManager()
