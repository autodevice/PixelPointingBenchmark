"""Evaluation module for pixel pointing benchmark."""

from evaluation.vlm_evaluator import VLMEvaluator
from evaluation.metrics import (
    calculate_distance,
    calculate_accuracy_metrics,
    calculate_statistics_across_passes,
)
from evaluation.results_manager import ResultsManager
from evaluation.runner import run_evaluation, MODELS

__all__ = [
    "VLMEvaluator",
    "calculate_distance",
    "calculate_accuracy_metrics",
    "calculate_statistics_across_passes",
    "ResultsManager",
    "run_evaluation",
    "MODELS",
]

