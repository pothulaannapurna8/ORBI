"""
Multi-Temporal Change Detection & False-Alarm Suppression Benchmark
Evaluates:
- Change Confidence Score calibration
- True Positive change detection rate
- False-alarm suppression rate under cloud contamination
- Earliest-change-date onset accuracy
- Registration quality and sub-pixel alignment
- NDWI water detection accuracy
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.db_manager import db
from backend.quality.arosics_register import arosics_aligner
from backend.change_detection.open_cd_wrapper import open_cd_detector
from backend.change_detection.change_confidence import compute_change_confidence_score, categorize_confidence
from backend.change_detection.earliest_change import estimate_earliest_change_date
from backend.change_detection.ndwi_water import ndwi_detector
from PIL import Image
import numpy as np

def evaluate_change_pipeline() -> Dict:
    """
    Comprehensive change detection evaluation with error handling and detailed metrics.
    
    Returns:
        Dictionary containing evaluation metrics and detailed results
    """
    print("=== PS26227 Multi-Temporal Change & Suppression Benchmark ===")
    
    metrics = {
        "total_locations": 0,
        "successful_evaluations": 0,
        "failed_evaluations": 0,
        "change_detection_results": [],
        "suppression_results": [],
        "registration_results": [],
        "water_detection_results": [],
        "errors": []
    }
    
    # Test locations (urban and river basins)
    test_locations = [
        {"location_key": "loc_cell_p12_950_p77_550", "type": "urban", "expected_change": True},
        {"location_key": "loc_cell_p13_080_p80_270", "type": "river", "expected_change": True}
    ]
    
    for test_loc in test_locations:
        location_key = test_loc["location_key"]
        location_type = test_loc["type"]
        expected_change = test_loc["expected_change"]
        
        metrics["total_locations"] += 1
        
        try:
            tiles = db.get_tiles_by_location(location_key)
            print(f"\n{location_type.upper()} AOI [{location_key}]: Found {len(tiles)} temporal dates")
            
            if len(tiles) < 2:
                print(f"  ⚠ Insufficient tiles for analysis (need >= 2, got {len(tiles)})")
                metrics["errors"].append({
                    "location_key": location_key,
                    "error": f"Insufficient tiles: {len(tiles)} < 2"
                })
                metrics["failed_evaluations"] += 1
                continue
            
            # Load before/after images
            try:
                im_before = np.array(Image.open(tiles[0]["rgb_filepath"]).convert("RGB"))
                im_after = np.array(Image.open(tiles[-1]["rgb_filepath"]).convert("RGB"))
            except Exception as e:
                error_msg = f"Image loading failed: {str(e)}"
                metrics["errors"].append({"location_key": location_key, "error": error_msg})
                metrics["failed_evaluations"] += 1
                print(f"  ✗ {error_msg}")
                continue
            
            # Co-registration evaluation
            try:
                reg, shift_px, reg_q = arosics_aligner.register(im_before, im_after)
                registration_result = {
                    "location_key": location_key,
                    "shift_px": shift_px,
                    "registration_quality": reg_q,
                    "within_tolerance": shift_px < 5.0
                }
                metrics["registration_results"].append(registration_result)
                print(f"  Registration: Shift={shift_px:.2f}px, Quality={reg_q:.2f}")
            except Exception as e:
                print(f"  ⚠ Registration failed: {e}")
                reg, shift_px, reg_q = im_after, 0.0, 0.5
            
            # Open-CD change detection
            try:
                mask, evidence = open_cd_detector.detect_change(im_before, reg)
                change_result = {
                    "location_key": location_key,
                    "change_evidence": evidence,
                    "expected_change": expected_change,
                    "detected_change": evidence > 0.5,
                    "correct_detection": (evidence > 0.5) == expected_change
                }
                metrics["change_detection_results"].append(change_result)
                print(f"  Change Evidence: {evidence:.3f}")
            except Exception as e:
                print(f"  ⚠ Change detection failed: {e}")
                evidence = 0.0
            
            # Build timeline for earliest change estimation
            timeline = []
            for t in tiles:
                try:
                    t_img = np.array(Image.open(t["rgb_filepath"]).convert("RGB"))
                    _, t_ev = open_cd_detector.detect_change(im_before, t_img)
                    timeline.append({
                        "date": t["acquisition_datetime"][:10],
                        "change_score": t_ev,
                        "cloud_score": float(t.get("cloud_cover", 0.0))
                    })
                except Exception as e:
                    print(f"  ⚠ Timeline image processing failed for {t['acquisition_datetime'][:10]}: {e}")
                    continue
            
            if len(timeline) >= 2:
                try:
                    earliest_date, consistency, date_conf = estimate_earliest_change_date(timeline)
                    print(f"  Earliest Change: {earliest_date} (consistency={consistency:.2f})")
                except Exception as e:
                    print(f"  ⚠ Earliest change estimation failed: {e}")
                    earliest_date, consistency, date_conf = "N/A", 0.0, 0.0
            
            # Compute Change Confidence Score
            try:
                ccs = compute_change_confidence_score(
                    change_evidence=evidence,
                    cloud_score=float(tiles[-1].get("cloud_cover", 0.0)),
                    registration_quality=reg_q,
                    temporal_consistency=consistency
                )
                category = categorize_confidence(ccs)
                print(f"  CCS: {ccs:.3f} -> [{category.upper()}]")
            except Exception as e:
                print(f"  ⚠ CCS computation failed: {e}")
                ccs, category = 0.0, "error"
            
            # NDWI water detection (for river basins)
            if location_type == "river":
                try:
                    ndwi_result = ndwi_detector.compute_ndwi_delta(im_before, im_after)
                    water_result = {
                        "location_key": location_key,
                        "ndwi_delta": ndwi_result,
                        "water_expansion": ndwi_result > 0.1
                    }
                    metrics["water_detection_results"].append(water_result)
                    print(f"  NDWI Delta: {ndwi_result:.3f}")
                except Exception as e:
                    print(f"  ⚠ NDWI detection failed: {e}")
            
            # False-alarm suppression test (check cloudy dates)
            cloudy_tiles = [t for t in tiles if t.get("cloud_cover", 0.0) > 0.5]
            if cloudy_tiles:
                for cloudy_tile in cloudy_tiles[:1]:  # Test first cloudy tile
                    try:
                        cloudy_img = np.array(Image.open(cloudy_tile["rgb_filepath"]).convert("RGB"))
                        _, cloudy_ev = open_cd_detector.detect_change(im_before, cloudy_img)
                        cloudy_ccs = compute_change_confidence_score(
                            change_evidence=cloudy_ev,
                            cloud_score=float(cloudy_tile.get("cloud_cover", 0.0)),
                            registration_quality=1.0,
                            temporal_consistency=0.5
                        )
                        suppression_result = {
                            "location_key": location_key,
                            "date": cloudy_tile["acquisition_datetime"][:10],
                            "cloud_cover": cloudy_tile.get("cloud_cover", 0.0),
                            "naive_evidence": cloudy_ev,
                            "corrected_ccs": cloudy_ccs,
                            "suppressed": cloudy_ccs < 0.40
                        }
                        metrics["suppression_results"].append(suppression_result)
                        print(f"  Cloudy Date {cloudy_tile['acquisition_datetime'][:10]}: CCS={cloudy_ccs:.3f} [{'SUPPRESSED' if cloudy_ccs < 0.40 else 'FLAGGED'}]")
                    except Exception as e:
                        print(f"  ⚠ Suppression test failed: {e}")
            
            metrics["successful_evaluations"] += 1
            
        except Exception as e:
            error_msg = f"Location evaluation failed: {str(e)}"
            metrics["errors"].append({"location_key": location_key, "error": error_msg})
            metrics["failed_evaluations"] += 1
            print(f"  ✗ {error_msg}")
    
    return metrics

def print_metrics_summary(metrics: Dict):
    """Print formatted summary of evaluation metrics."""
    print("\n" + "="*60)
    print("CHANGE DETECTION EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nEvaluation Statistics:")
    print(f"  Total Locations: {metrics['total_locations']}")
    print(f"  Successful: {metrics['successful_evaluations']}")
    print(f"  Failed: {metrics['failed_evaluations']}")
    print(f"  Success Rate: {(metrics['successful_evaluations'] / metrics['total_locations'] * 100):.1f}%")
    
    if metrics["change_detection_results"]:
        print(f"\nChange Detection Performance:")
        correct = sum(1 for r in metrics["change_detection_results"] if r["correct_detection"])
        total = len(metrics["change_detection_results"])
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
        avg_evidence = sum(r["change_evidence"] for r in metrics["change_detection_results"]) / total
        print(f"  Average Change Evidence: {avg_evidence:.3f}")
    
    if metrics["registration_results"]:
        print(f"\nRegistration Quality:")
        avg_shift = sum(r["shift_px"] for r in metrics["registration_results"]) / len(metrics["registration_results"])
        avg_quality = sum(r["registration_quality"] for r in metrics["registration_results"]) / len(metrics["registration_results"])
        within_tolerance = sum(1 for r in metrics["registration_results"] if r["within_tolerance"])
        print(f"  Average Shift: {avg_shift:.2f}px")
        print(f"  Average Quality: {avg_quality:.3f}")
        print(f"  Within Tolerance (<5px): {within_tolerance}/{len(metrics['registration_results'])}")
    
    if metrics["suppression_results"]:
        print(f"\nFalse-Alarm Suppression:")
        suppressed = sum(1 for r in metrics["suppression_results"] if r["suppressed"])
        total = len(metrics["suppression_results"])
        print(f"  Suppression Rate: {suppressed}/{total} ({suppressed/total*100:.1f}%)")
        avg_cloud_cover = sum(r["cloud_cover"] for r in metrics["suppression_results"]) / total
        print(f"  Average Cloud Cover: {avg_cloud_cover:.2f}")
    
    if metrics["water_detection_results"]:
        print(f"\nWater Detection (NDWI):")
        water_expansions = sum(1 for r in metrics["water_detection_results"] if r["water_expansion"])
        print(f"  Water Expansions Detected: {water_expansions}/{len(metrics['water_detection_results'])}")
        avg_ndwi = sum(r["ndwi_delta"] for r in metrics["water_detection_results"]) / len(metrics["water_detection_results"])
        print(f"  Average NDWI Delta: {avg_ndwi:.3f}")
    
    if metrics["errors"]:
        print(f"\nErrors ({len(metrics['errors'])}):")
        for error in metrics["errors"]:
            print(f"  [{error.get('location_key', 'UNKNOWN')}] {error['error']}")
    
    print("\n" + "="*60)

def save_metrics_report(metrics: Dict, output_path: str = "evaluation/change_detection_metrics.json"):
    """Save detailed metrics report to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nDetailed metrics saved to: {output_path}")

if __name__ == "__main__":
    metrics = evaluate_change_pipeline()
    print_metrics_summary(metrics)
    save_metrics_report(metrics)
    
    # Exit with error code if any evaluations failed
    if metrics["failed_evaluations"] > 0:
        sys.exit(1)
