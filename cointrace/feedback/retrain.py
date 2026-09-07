"""
Phase 9 (cont.) - turns accumulated analyst feedback into updated fusion
weights, so corrections made in the dashboard actually change future
rankings instead of just being logged. Everything here is local
pandas/sklearn - no network access - so it runs fully offline.
"""
from __future__ import annotations

import json

import pandas as pd

from cointrace import config
from cointrace.detection.fusion import fit_logistic_weights
from cointrace.feedback.store import latest_labels_only, load_feedback


def enough_feedback_to_retrain(min_n: int | None = None) -> bool:
    """Logistic regression needs a handful of examples of BOTH classes to
    mean anything - one confirmed illicit entity alone can't teach the
    system to distrust false positives generally."""
    min_n = config.MIN_FEEDBACK_FOR_RETRAIN if min_n is None else min_n
    labels = latest_labels_only(load_feedback())
    if labels.empty:
        return False
    return len(labels) >= min_n and labels.nunique() >= 2


def retrain_from_feedback(score_frames: dict[str, pd.Series]) -> dict[str, float] | None:
    """Fits fresh fusion weights against every entity an analyst has
    labeled so far and saves them to config.LEARNED_WEIGHTS_JSON so the
    NEXT pipeline run picks them up automatically (see
    scripts/run_pipeline.py). Returns the new weights, or None if there
    isn't enough feedback yet."""
    if not enough_feedback_to_retrain():
        return None

    labels = latest_labels_only(load_feedback())
    model = fit_logistic_weights(score_frames, labels)

    score_cols = list(pd.DataFrame(score_frames).columns)
    coefs = dict(zip(score_cols, model.coef_[0]))
    # Clip to non-negative and renormalize to sum to 1, so the learned
    # weights are drop-in compatible with config.FUSION_WEIGHTS' shape.
    coefs = {k: max(v, 0.0) for k, v in coefs.items()}
    total = sum(coefs.values()) or 1.0
    weights = {k: v / total for k, v in coefs.items()}

    payload = {
        "weights": weights,
        "bias": float(model.intercept_[0]),
        "n_feedback_examples": int(len(labels)),
    }
    config.PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.LEARNED_WEIGHTS_JSON.write_text(json.dumps(payload, indent=2))
    return weights


def load_learned_weights() -> dict[str, float] | None:
    """Weights learned from feedback on a previous run, if any - used by
    run_pipeline.py in place of the static config.FUSION_WEIGHTS."""
    if not config.LEARNED_WEIGHTS_JSON.exists():
        return None
    payload = json.loads(config.LEARNED_WEIGHTS_JSON.read_text())
    return payload.get("weights")
