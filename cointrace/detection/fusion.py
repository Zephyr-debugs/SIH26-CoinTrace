"""
Phase 5 - Score fusion.

Combines the (already-normalized-to-[0,1]) scores from each detector into
one weighted risk_score per entity, per config.FUSION_WEIGHTS. Kept as a
plain weighted sum by default (transparent, easy to defend/explain in a
write-up); if labeled data is ever available, `fit_logistic_weights` can
learn the blend instead - but unsupervised weighted-sum is the default so
the system works with zero labels, matching the proposal's design intent.
"""
from __future__ import annotations

import pandas as pd

from cointrace import config


def fuse_scores(score_frames: dict[str, pd.Series], weights: dict[str, float] = None) -> pd.DataFrame:
    """score_frames: {'isolation_forest_score': Series, 'autoencoder_score': Series,
    'motif_score': Series, ...}. Any subset of config.FUSION_WEIGHTS' keys is fine -
    missing detectors just get their weight redistributed proportionally
    across whichever ones ARE present, so the system still works if e.g.
    torch isn't installed and the autoencoder was skipped."""
    weights = weights or config.FUSION_WEIGHTS
    combined = pd.DataFrame(score_frames).fillna(0.0)

    present_weights = {k: w for k, w in weights.items() if k in combined.columns}
    total_weight = sum(present_weights.values()) or 1.0
    normalized_weights = {k: w / total_weight for k, w in present_weights.items()}

    combined["risk_score"] = sum(
        combined[col] * w for col, w in normalized_weights.items()
    )
    combined = combined.sort_values("risk_score", ascending=False)
    combined["rank"] = range(1, len(combined) + 1)
    combined["is_alert"] = combined["risk_score"] >= config.RISK_ALERT_THRESHOLD
    return combined


def fit_logistic_weights(score_frames: dict[str, pd.Series], labels: pd.Series):
    """Optional: if you DO have some labeled data (even a small hand-labeled
    sample), learn the blend weights via logistic regression instead of
    using the fixed config.FUSION_WEIGHTS. labels: Series of 0/1 aligned
    to the same entity_id index. Returns a fitted sklearn model."""
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame(score_frames).fillna(0.0)
    common_idx = X.index.intersection(labels.index)
    X, y = X.loc[common_idx], labels.loc[common_idx]

    # class_weight="balanced" matters here specifically because analyst
    # feedback is almost always lopsided (far more confirmed-benign
    # clicks than confirmed-illicit ones), so an unweighted fit would
    # just learn to ignore the minority class.
    model = LogisticRegression(class_weight="balanced")
    model.fit(X, y)
    return model
