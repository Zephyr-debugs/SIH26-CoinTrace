"""
Phase 5 - Isolation Forest point-anomaly detector.

Unsupervised: fits directly on the entity feature matrix, no labels used.
Output is normalized to [0, 1] where higher = more anomalous, so it can
be blended with the other detectors in fusion.py on a common scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from cointrace import config


def run_isolation_forest(
    feat_df: pd.DataFrame,
    contamination: float = config.ISOLATION_FOREST_CONTAMINATION,
    random_state: int = config.RANDOM_SEED,
) -> pd.Series:
    """Returns a Series (index=entity_id) of isolation_forest_score in [0,1]."""
    X = StandardScaler().fit_transform(feat_df.values)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    # score_samples: higher = more normal. Flip and min-max normalize so
    # higher = more anomalous, matching the other detectors' convention.
    raw = -model.score_samples(X)
    normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    return pd.Series(normalized, index=feat_df.index, name="isolation_forest_score")
