"""Task-agnostic mathematical core."""

from .api import FTCCSResult, fit_and_select_coordinates
from .coordinate_information import CoordinateInformation, coordinate_information
from .coordinate_selector import CoordinateSelection, select_coordinates
from .task_geometry import TaskGeometry, learn_task_geometry
from .task_signal_model import TaskSignalModel, build_task_signal_model

__all__ = [
    "CoordinateInformation",
    "CoordinateSelection",
    "FTCCSResult",
    "TaskGeometry",
    "TaskSignalModel",
    "build_task_signal_model",
    "coordinate_information",
    "fit_and_select_coordinates",
    "learn_task_geometry",
    "select_coordinates",
]

