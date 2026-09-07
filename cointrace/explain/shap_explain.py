"""
Phase 6 - SHAP feature attribution.

Explains the Isolation Forest's contribution to an entity's risk score
(the model that most directly benefits from SHAP; the autoencoder's
reconstruction error and the rule-based motif score are already
self-explanatory - the motif_reasons string IS their explanation).

Deliberately scoped: only computed for the top-N ranked entities
(config.TOP_N_FOR_EAGER_EXPLANATION), per the proposal's performance
guidance (Section 3.iii) - SHAP on a full 3,000+ entity population is
wasted compute when only the top alerts get looked at.
"""
from __future__ import annotations

import pandas as pd

from cointrace import config

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def explain_top_entities(
    model,
    feat_df: pd.DataFrame,
    ranked_entity_ids: list[str],
    top_n: int = config.TOP_N_FOR_EAGER_EXPLANATION,
) -> dict[str, dict[str, float]]:
    """model: a fitted IsolationForest (or any sklearn model shap.Explainer
    supports). Returns {entity_id: {feature_name: shap_value, ...}} for
    the top_n ranked entities only."""
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "shap is required for explainability. Install it with: pip install shap"
        )

    top_ids = ranked_entity_ids[:top_n]
    subset = feat_df.loc[top_ids]

    explainer = shap.Explainer(model, feat_df)
    shap_values = explainer(subset)

    result = {}
    for i, entity_id in enumerate(top_ids):
        result[entity_id] = dict(zip(feat_df.columns, shap_values.values[i]))
    return result


def top_contributing_features(shap_row: dict[str, float], n: int = 3) -> list[tuple[str, float]]:
    """Given one entity's {feature: shap_value} dict, return the n
    features with the largest |shap_value| - i.e. what drove the score."""
    return sorted(shap_row.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
