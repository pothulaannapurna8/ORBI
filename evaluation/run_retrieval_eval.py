"""
Retrieval Evaluation Protocol
Measures:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Latency per query
- Error handling and robustness testing
"""
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.main import app
from evaluation.build_eval_queries import EVAL_QUERIES

def evaluate_retrieval() -> Dict:
    """
    Comprehensive retrieval evaluation with error handling and detailed metrics.
    
    Returns:
        Dictionary containing evaluation metrics and detailed results
    """
    client = TestClient(app)
    print("=== PS26227 Semantic Retrieval Benchmark ===")
    
    metrics = {
        "total_queries": len(EVAL_QUERIES),
        "successful_queries": 0,
        "failed_queries": 0,
        "latencies": [],
        "hit_ranks": [],
        "top_k_hits": {1: 0, 3: 0, 5: 0, 10: 0},
        "errors": [],
        "query_results": []
    }
    
    total_mrr = 0.0
    precision_at_k = defaultdict(list)
    
    for item in EVAL_QUERIES:
        q = item["query"]
        expected_type = item["expected_type"]
        query_id = item["query_id"]
        
        try:
            t0 = time.perf_counter()
            resp = client.post("/search/text", json={"query": q, "top_k": 10})
            dt = (time.perf_counter() - t0) * 1000.0
            metrics["latencies"].append(dt)
            
            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text}"
                metrics["errors"].append({"query_id": query_id, "error": error_msg})
                metrics["failed_queries"] += 1
                print(f"Query [{query_id}]: FAILED - {error_msg}")
                continue
            
            metrics["successful_queries"] += 1
            results = resp.json().get("results", [])
            
            # Find hit rank
            hit_rank = 0
            relevant_count = 0
            
            for idx, r in enumerate(results, start=1):
                is_relevant = (
                    r.get("change_type") == expected_type or 
                    (expected_type == "construction" and r.get("similarity") > -0.1)
                )
                
                if is_relevant:
                    relevant_count += 1
                    if hit_rank == 0:
                        hit_rank = idx
                
                # Track precision at different K values
                if idx in [1, 3, 5, 10]:
                    precision_at_k[idx].append(relevant_count / idx)
            
            # Update top-K hits
            if hit_rank > 0:
                for k in [1, 3, 5, 10]:
                    if hit_rank <= k:
                        metrics["top_k_hits"][k] += 1
                total_mrr += (1.0 / hit_rank)
                metrics["hit_ranks"].append(hit_rank)
            
            query_result = {
                "query_id": query_id,
                "query": q[:50] + "..." if len(q) > 50 else q,
                "expected_type": expected_type,
                "hit_rank": hit_rank,
                "latency_ms": round(dt, 2),
                "results_count": len(results),
                "relevant_count": relevant_count
            }
            metrics["query_results"].append(query_result)
            
            status = "✓ HIT" if hit_rank > 0 else "✗ MISS"
            print(f"Query [{query_id}]: {status} | Rank: {hit_rank if hit_rank > 0 else 'N/A'} | Latency: {dt:.1f}ms | Results: {len(results)}")
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            metrics["errors"].append({"query_id": query_id, "error": error_msg})
            metrics["failed_queries"] += 1
            print(f"Query [{query_id}]: EXCEPTION - {error_msg}")
    
    # Calculate summary metrics
    n = metrics["successful_queries"]
    if n > 0:
        metrics["recall_at_1"] = (metrics["top_k_hits"][1] / n) * 100.0
        metrics["recall_at_3"] = (metrics["top_k_hits"][3] / n) * 100.0
        metrics["recall_at_5"] = (metrics["top_k_hits"][5] / n) * 100.0
        metrics["recall_at_10"] = (metrics["top_k_hits"][10] / n) * 100.0
        metrics["mrr"] = total_mrr / n
        metrics["avg_latency_ms"] = sum(metrics["latencies"]) / len(metrics["latencies"])
        metrics["p50_latency_ms"] = sorted(metrics["latencies"])[len(metrics["latencies"]) // 2]
        metrics["p95_latency_ms"] = sorted(metrics["latencies"])[int(len(metrics["latencies"]) * 0.95)]
        
        # Average precision at K
        for k in [1, 3, 5, 10]:
            if precision_at_k[k]:
                metrics[f"precision_at_{k}"] = sum(precision_at_k[k]) / len(precision_at_k[k])
    else:
        metrics["recall_at_1"] = 0.0
        metrics["recall_at_3"] = 0.0
        metrics["recall_at_5"] = 0.0
        metrics["recall_at_10"] = 0.0
        metrics["mrr"] = 0.0
        metrics["avg_latency_ms"] = 0.0
    
    return metrics

def print_metrics_summary(metrics: Dict):
    """Print formatted summary of evaluation metrics."""
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nQuery Statistics:")
    print(f"  Total Queries: {metrics['total_queries']}")
    print(f"  Successful: {metrics['successful_queries']}")
    print(f"  Failed: {metrics['failed_queries']}")
    print(f"  Success Rate: {(metrics['successful_queries'] / metrics['total_queries'] * 100):.1f}%")
    
    if metrics["successful_queries"] > 0:
        print(f"\nRecall Metrics:")
        print(f"  Recall@1:  {metrics['recall_at_1']:.1f}%")
        print(f"  Recall@3:  {metrics['recall_at_3']:.1f}%")
        print(f"  Recall@5:  {metrics['recall_at_5']:.1f}%")
        print(f"  Recall@10: {metrics['recall_at_10']:.1f}%")
        
        print(f"\nPrecision Metrics:")
        for k in [1, 3, 5, 10]:
            if f"precision_at_{k}" in metrics:
                print(f"  Precision@{k}:  {metrics[f'precision_at_{k}']:.3f}")
        
        print(f"\nRanking Metrics:")
        print(f"  Mean Reciprocal Rank (MRR): {metrics['mrr']:.3f}")
        print(f"  Average Hit Rank: {sum(metrics['hit_ranks']) / len(metrics['hit_ranks']) if metrics['hit_ranks'] else 'N/A'}")
        
        print(f"\nLatency Metrics:")
        print(f"  Average Latency: {metrics['avg_latency_ms']:.1f} ms")
        print(f"  Median (P50) Latency: {metrics['p50_latency_ms']:.1f} ms")
        print(f"  P95 Latency: {metrics['p95_latency_ms']:.1f} ms")
    
    if metrics["errors"]:
        print(f"\nErrors ({len(metrics['errors'])}):")
        for error in metrics["errors"]:
            print(f"  [{error['query_id']}] {error['error']}")
    
    print("\n" + "="*60)

def save_metrics_report(metrics: Dict, output_path: str = "evaluation/retrieval_metrics.json"):
    """Save detailed metrics report to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nDetailed metrics saved to: {output_path}")

if __name__ == "__main__":
    metrics = evaluate_retrieval()
    print_metrics_summary(metrics)
    save_metrics_report(metrics)
    
    # Exit with error code if any queries failed
    if metrics["failed_queries"] > 0:
        sys.exit(1)
