"""Task-specific evaluation code, isolated from selector fitting."""

from .detection import cem_score_map, evaluate_detection_scores

__all__ = ["cem_score_map", "evaluate_detection_scores"]

