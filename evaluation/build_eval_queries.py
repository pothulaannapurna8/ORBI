"""
Benchmark Query Dataset Generator for Information Retrieval
Defines ground-truth evaluation queries across:
- Object classes (structures, roads, water, vegetation)
- Temporal change queries (new construction, flood expansion, stable)
- Multi-sensor and cross-modal queries
"""
import json
from pathlib import Path

EVAL_QUERIES = [
    {
        "query_id": "Q01",
        "query": "newly built structures and urban expansion",
        "expected_type": "construction",
        "expected_aoi": "urban"
    },
    {
        "query_id": "Q02",
        "query": "river basin water extent increase and flooding",
        "expected_type": "water_expansion",
        "expected_aoi": "river"
    },
    {
        "query_id": "Q03",
        "query": "dense residential buildings and roads",
        "expected_type": "construction",
        "expected_aoi": "urban"
    },
    {
        "query_id": "Q04",
        "query": "natural water bodies lakes and wetlands",
        "expected_type": "water_expansion",
        "expected_aoi": "river"
    },
    {
        "query_id": "Q05",
        "query": "construction between 2024 and 2026",
        "expected_type": "construction",
        "expected_aoi": "urban"
    }
]

def save_benchmark_queries(dest_path: str = "evaluation/queries.json"):
    p = Path(dest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(EVAL_QUERIES, f, indent=2)
    print(f"Saved {len(EVAL_QUERIES)} benchmark queries to {dest_path}")

if __name__ == "__main__":
    save_benchmark_queries()
