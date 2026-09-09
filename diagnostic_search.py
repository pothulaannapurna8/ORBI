#!/usr/bin/env python
"""
Comprehensive Diagnostic Script for Search Functionality
Tests each component of the search pipeline in isolation.
"""
import sys
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("=" * 80)
print("PS26227 SEARCH DIAGNOSTIC - Component Analysis")
print("=" * 80)

# ============================================================================
# TEST 1: Embedding Generation
# ============================================================================
print("\n[TEST 1] EMBEDDING GENERATION")
print("-" * 80)

try:
    from backend.embeddings.clip_rsicd_encoder import clip_encoder
    print("✓ CLIP encoder imported successfully")
    
    # Test text encoding
    text_query = "newly built structures near river"
    text_vec = clip_encoder.encode_text(text_query)
    print(f"  Text embedding: {len(text_vec)} dimensions")
    print(f"  Sample values: {text_vec[:5]}")
    print(f"  Non-zero count: {sum(1 for v in text_vec if abs(v) > 1e-6)}")
    
    # Check for all-zeros or NaN
    if all(abs(v) < 1e-6 for v in text_vec):
        print("  ⚠️  WARNING: Text embedding is all zeros!")
    if any(np.isnan(v) for v in text_vec):
        print("  ⚠️  WARNING: Text embedding contains NaN!")
    
    # Test image encoding
    test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    image_vec = clip_encoder.encode_image(test_img)
    print(f"  Image embedding: {len(image_vec)} dimensions")
    print(f"  Non-zero count: {sum(1 for v in image_vec if abs(v) > 1e-6)}")
    
    if all(abs(v) < 1e-6 for v in image_vec):
        print("  ⚠️  WARNING: Image embedding is all zeros!")
    if any(np.isnan(v) for v in image_vec):
        print("  ⚠️  WARNING: Image embedding contains NaN!")
    
    print("✓ Embedding generation working")
except Exception as e:
    print(f"✗ EMBEDDING GENERATION FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 2: Qdrant Collections Status
# ============================================================================
print("\n[TEST 2] QDRANT VECTOR DATABASE STATUS")
print("-" * 80)

try:
    from backend.database.qdrant_client import qdrant_store
    print("✓ Qdrant store initialized")
    
    # Check collections
    collections = qdrant_store.client.get_collections().collections
    collection_names = [c.name for c in collections]
    print(f"  Available collections: {collection_names}")
    
    for coll_name in collection_names:
        try:
            points_info = qdrant_store.client.count(collection_name=coll_name)
            print(f"  - {coll_name}: {points_info.count} points")
            
            if points_info.count == 0:
                print(f"    ⚠️  WARNING: Collection '{coll_name}' is EMPTY! No search results possible.")
        except Exception as e:
            print(f"  - {coll_name}: Error counting points - {e}")
    
    # Try a test search
    print("\n  Testing search capability...")
    test_vec = [0.1] * 512
    try:
        results = qdrant_store.search_semantic(test_vec, top_k=5)
        print(f"  Search returned {len(results)} results")
        if len(results) > 0:
            print(f"  First result ID: {results[0].get('id')}")
            print(f"  First result score: {results[0].get('score')}")
            payload = results[0].get('payload', {})
            print(f"  First result payload keys: {list(payload.keys())}")
            if 'location_key' not in payload:
                print("  ⚠️  WARNING: 'location_key' missing from Qdrant payload!")
    except Exception as e:
        print(f"  ✗ Search test failed: {e}")
    
    print("✓ Qdrant database accessible")
except Exception as e:
    print(f"✗ QDRANT DATABASE FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 3: Database Tile Storage
# ============================================================================
print("\n[TEST 3] TILE DATABASE (SQLite/PostGIS)")
print("-" * 80)

try:
    from backend.database.db_manager import db
    print("✓ Database manager initialized")
    
    # Count tiles in database
    try:
        conn = __import__('sqlite3').connect(str(Path(__file__).parent / 'data' / 'metadata.db'))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM tile")
        tile_count = cur.fetchone()[0]
        conn.close()
        print(f"  Total tiles in database: {tile_count}")
        
        if tile_count == 0:
            print("  ⚠️  CRITICAL: No tiles in database! Search has nothing to process.")
        else:
            # Get sample locations
            conn = __import__('sqlite3').connect(str(Path(__file__).parent / 'data' / 'metadata.db'))
            conn.row_factory = __import__('sqlite3').Row
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT location_key FROM tile LIMIT 5")
            locations = [r['location_key'] for r in cur.fetchall()]
            conn.close()
            print(f"  Sample location_keys: {locations}")
            
            # Check tiles per location
            for loc_key in locations[:2]:
                tiles = db.get_tiles_by_location(loc_key)
                print(f"  - Location '{loc_key}': {len(tiles)} tiles")
    except Exception as e:
        print(f"  Error querying tiles: {e}")
    
    print("✓ Database accessible")
except Exception as e:
    print(f"✗ DATABASE FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: Search Pipeline - Full Integration
# ============================================================================
print("\n[TEST 4] FULL SEARCH PIPELINE")
print("-" * 80)

try:
    # Simulate a search request
    from backend.api.search import run_temporal_analysis_for_candidates
    
    # Use the indexed semantic collection so this test exercises real tile pairs.
    real_candidates = qdrant_store.search_semantic([0.1] * 512, top_k=5)
    print(f"  Testing with {len(real_candidates)} indexed candidates...")
    results = run_temporal_analysis_for_candidates(real_candidates)
    print(f"  Results returned: {len(results)}")
    
    if len(results) > 0:
        result = results[0]
        print(f"  Sample result keys: {list(result.keys())}")
        print(f"  Confidence level: {result.get('confidence_level')}")
        print(f"  Reranked score: {result.get('reranked_score')}")
    else:
        print("  ⚠️  Pipeline returned 0 results from indexed candidates!")
    
    print("✓ Search pipeline completed")
except Exception as e:
    print(f"✗ SEARCH PIPELINE FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 5: Data Ingestion Check
# ============================================================================
print("\n[TEST 5] DATA INGESTION PIPELINE")
print("-" * 80)

try:
    from backend.ingestion.tiler import tiler
    print("✓ Tiler module imported")
    
    # Check if there's sample data
    data_dir = Path(__file__).parent / 'data'
    incoming_dir = data_dir / 'incoming'
    raw_dir = data_dir / 'raw'
    tiles_dir = data_dir / 'tiles'
    
    print(f"  Incoming scenes: {len(list(incoming_dir.glob('*')))}")
    print(f"  Raw data: {len(list(raw_dir.glob('*')))}")
    print(f"  Generated tiles: {len(list(tiles_dir.glob('*.png')))}")
    
    staged_scene_count = len(list(incoming_dir.glob('*'))) + len(list(raw_dir.glob('*')))
    if staged_scene_count == 0:
        print("  ⚠️  No staged scenes in incoming/ or raw/")
    else:
        print(f"  Staged scenes available for ingestion: {staged_scene_count}")
    
    print("✓ Ingestion pipeline accessible")
except Exception as e:
    print(f"✗ INGESTION PIPELINE FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# DIAGNOSTIC SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print("""
If search is returning empty results, the issue is likely one of:

1. EMPTY QDRANT: Collections have 0 points
   → Solution: Run data ingestion pipeline to populate collections

2. EMPTY DATABASE: Tiles table has 0 records
   → Solution: Run data ingestion pipeline to populate database

3. SILENT ENCODING FAILURES: Embeddings are all-zeros or NaN
   → Solution: Check embedding function error handling

4. PAYLOAD MISSING: Qdrant points missing 'location_key' field
   → Solution: Re-index with proper payload structure

5. PIPELINE FILTERING: Results filtered out by confidence thresholds
   → Solution: Adjust CCS weights or check filtering logic

Check the warnings above (⚠️) to identify the specific issue.
""")
print("=" * 80)
