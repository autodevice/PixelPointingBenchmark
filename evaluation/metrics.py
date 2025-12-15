import math
from typing import Any, Dict, List, Tuple
import numpy as np


def calculate_distance(
    predicted: Tuple[int, int], expected: Tuple[int, int]
) -> float:
    """Calculate Euclidean distance between predicted and expected coordinates."""
    return math.sqrt(
        (predicted[0] - expected[0]) ** 2 + (predicted[1] - expected[1]) ** 2
    )


def calculate_accuracy_metrics(
    results: List[Dict[str, Any]], image_size: Tuple[int, int]
) -> Dict[str, Any]:
    """Calculate accuracy metrics from results."""
    distances = []
    successful_extractions = 0
    total_tests = len(results)

    for result in results:
        if result["predicted_coords"] is not None:
            successful_extractions += 1
            distance = result["distance"]
            distances.append(distance)

    if not distances:
        return {
            "total_tests": total_tests,
            "successful_extractions": 0,
            "extraction_rate": 0.0,
            "mean_distance": None,
            "median_distance": None,
            "max_distance": None,
            "min_distance": None,
            "accuracy_within_10px": 0.0,
            "accuracy_within_5_percent": 0.0,
        }

    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)

    screen_diagonal = math.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
    threshold_5_percent = screen_diagonal * 0.05

    accuracy_10px = sum(1 for d in distances if d <= 10) / len(distances) * 100
    accuracy_5_percent = (
        sum(1 for d in distances if d <= threshold_5_percent) / len(distances) * 100
    )

    return {
        "total_tests": total_tests,
        "successful_extractions": successful_extractions,
        "extraction_rate": (successful_extractions / total_tests) * 100,
        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "max_distance": max_distance,
        "min_distance": min_distance,
        "accuracy_within_10px": accuracy_10px,
        "accuracy_within_5_percent": accuracy_5_percent,
    }


def calculate_statistics_across_passes(
    all_passes: List[List[Dict[str, Any]]], image_size: Tuple[int, int]
) -> Dict[str, Any]:
    """Calculate statistics across multiple passes, including standard deviation."""
    if not all_passes:
        return {}
    
    all_distances = []
    pass_metrics = []
    
    for pass_results in all_passes:
        distances = [
            r["distance"] for r in pass_results 
            if r.get("predicted_coords") is not None and r.get("distance") is not None
        ]
        if distances:
            all_distances.extend(distances)
            metrics = calculate_accuracy_metrics(pass_results, image_size)
            pass_metrics.append(metrics)
    
    if not all_distances:
        return {}
    
    mean_distances = [m["mean_distance"] for m in pass_metrics if m.get("mean_distance") is not None]
    
    stats = {
        "num_passes": len(all_passes),
        "total_runs": len(all_distances),
        "overall_mean_distance": np.mean(all_distances),
        "overall_std_distance": np.std(all_distances),
        "overall_median_distance": np.median(all_distances),
        "overall_min_distance": np.min(all_distances),
        "overall_max_distance": np.max(all_distances),
    }
    
    if mean_distances:
        stats["mean_distance_across_passes"] = np.mean(mean_distances)
        stats["std_distance_across_passes"] = np.std(mean_distances)
    
    screen_diagonal = math.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
    threshold_5_percent = screen_diagonal * 0.05
    
    stats["overall_accuracy_within_10px"] = sum(1 for d in all_distances if d <= 10) / len(all_distances) * 100
    stats["overall_accuracy_within_5_percent"] = sum(1 for d in all_distances if d <= threshold_5_percent) / len(all_distances) * 100
    
    return stats

