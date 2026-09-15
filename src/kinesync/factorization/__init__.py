"""Multi-frame physical factorization for KineSync-GS."""

from .backend import FactorFrame, RouteAFactorizationBackend, select_evenly
from .frozen_backend import FrozenCameraObservationBackend
from .observability import BlockObservability, analyze_block_gradients
from .optimizer import PhysicalFactorOptimizer
from .schema import FactorizationConfig, FactorizationResult

__all__ = [
    "BlockObservability",
    "FactorFrame",
    "FactorizationConfig",
    "FactorizationResult",
    "FrozenCameraObservationBackend",
    "PhysicalFactorOptimizer",
    "RouteAFactorizationBackend",
    "analyze_block_gradients",
    "select_evenly",
]
