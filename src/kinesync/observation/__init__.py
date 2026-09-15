"""Differentiable observation backends for state correction."""

from .camera import TorchCamera
from .projected_centers import ProjectedCentersBackend
from .soft_occupancy import SoftOccupancyBackend

__all__ = ["ProjectedCentersBackend", "SoftOccupancyBackend", "TorchCamera"]
from .real_route_a import (
    RealObservationLossWeights,
    RealObservationTarget,
    RealRouteAObservationBackend,
    prepare_real_targets,
)

__all__ = [
    "RealObservationLossWeights",
    "RealObservationTarget",
    "RealRouteAObservationBackend",
    "prepare_real_targets",
]
