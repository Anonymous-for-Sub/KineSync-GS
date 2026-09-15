"""Development-only risk calibration and reference-free atomic state commits."""

from dataclasses import asdict, dataclass
from itertools import product
import math

import numpy as np

from kinesync.guard.decision import decide_update
from kinesync.guard.schema import GuardThresholds, UpdateEvidence


@dataclass(frozen=True)
class RiskRecord:
    state_id: str
    split: str
    evidence: UpdateEvidence
    measured_mae: float
    candidate_mae: float


def _grid(values, lower, upper=None):
    values = np.asarray(values, dtype=float)
    anchors = sorted(set([lower, *np.quantile(values, [0, .25, .5, .75, 1])]))
    if upper is not None:
        anchors.append(upper)
    anchors = sorted(set(max(lower, min(x, upper) if upper is not None else x) for x in anchors))
    return sorted(set(anchors + [(a + b) / 2 for a, b in zip(anchors, anchors[1:])]))


def calibrate(records):
    """Minimize harmful accepts, maximize useful accepts, then separation margin.

    MAE labels are used here only. Runtime commit accepts neither labels nor truth.
    Neutral updates do not contribute to risk or benefit; ties are deterministic.
    """
    if not records or any(r.split != 'development' for r in records):
        raise ValueError('Only nonempty development records may calibrate thresholds')
    if len({r.state_id for r in records}) != len(records):
        raise ValueError('duplicate development state')
    if any(not math.isfinite(r.measured_mae + r.candidate_mae) for r in records):
        raise ValueError('Nonfinite calibration labels')
    evidence = [r.evidence for r in records]
    if any(not e.finite for e in evidence):
        raise ValueError('Calibration evidence must be finite')
    features = np.array([[e.visual_gain_ratio, e.gradient_cosine,
                          e.correction_cosine, e.relative_correction_disagreement]
                         for e in evidence])
    harmful = np.array([r.candidate_mae > r.measured_mae + 1e-8 for r in records])
    helpful = np.array([r.candidate_mae < r.measured_mae - 1e-8 for r in records])
    valid = np.array([e.view_count >= 2 for e in evidence])
    grids = [_grid(features[:, 0], 0), _grid(features[:, 1], -1, 1),
             _grid(features[:, 2], -1, 1), _grid(features[:, 3], 0)]
    best = None
    evaluated = 0
    for values in product(*grids):
        margins = features - np.asarray(values)
        margins[:, 3] *= -1
        margin = margins.min(axis=1)
        accepted = (margin >= 0) & valid
        harm = int(np.sum(accepted & harmful))
        benefit = int(np.sum(accepted & helpful))
        classified = np.concatenate([margin[helpful], -margin[harmful]])
        separation = float(classified.min()) if len(classified) else 0.
        rank = (harm, -benefit, -separation, *values)
        evaluated += 1
        if best is None or rank < best[0]:
            best = (rank, values, harm, benefit, separation)
    _, values, harm, benefit, separation = best
    return {'thresholds': asdict(GuardThresholds(*values)),
            'harmful_accepted': harm, 'beneficial_accepted': benefit,
            'harmful_total': int(harmful.sum()), 'beneficial_total': int(helpful.sum()),
            'minimum_classification_margin': separation, 'grid_candidates': evaluated,
            'development_state_ids': sorted(r.state_id for r in records),
            'ranking': ['harmful_accept_count_min', 'beneficial_accept_count_max',
                        'minimum_classification_margin_max', 'lexicographic_thresholds'],
            'label_tolerance_mae_rad': 1e-8}


def commit(measured, candidate, evidence, thresholds):
    """Commit an entire bounded candidate or retain the entire measured state."""
    if len(measured) != len(candidate) or not measured:
        raise ValueError('State vectors must have equal nonzero length')
    if not all(math.isfinite(float(x)) for x in measured):
        raise ValueError('Measured state must be finite')
    decision = decide_update(evidence, GuardThresholds(**thresholds))
    if not all(math.isfinite(float(x)) for x in candidate):
        return list(measured), False
    return list(candidate if decision.accepted else measured), decision.accepted
