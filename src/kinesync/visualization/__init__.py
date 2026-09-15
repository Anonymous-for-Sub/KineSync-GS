"""Diagnostic image and video generation.

Keep optional rendering dependencies lazy so pure provenance validators can run
on a CPU-only review environment.
"""


def write_observation_audit(*args, **kwargs):
    from .observation_audit import write_observation_audit as implementation
    return implementation(*args, **kwargs)


def write_observation_audit_video(*args, **kwargs):
    from .observation_audit import write_observation_audit_video as implementation
    return implementation(*args, **kwargs)


def write_temporal_comparison(*args, **kwargs):
    from .temporal import write_temporal_comparison as implementation
    return implementation(*args, **kwargs)


def write_temporal_video(*args, **kwargs):
    from .temporal import write_temporal_video as implementation
    return implementation(*args, **kwargs)

__all__ = [
    "write_observation_audit",
    "write_observation_audit_video",
    "write_temporal_comparison",
    "write_temporal_video",
]
