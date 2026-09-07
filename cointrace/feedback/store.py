"""
Phase 9 - Analyst feedback store.

This is the "learn from its mistakes and don't repeat them" mechanism.
Worth being precise about the name: this is NOT one-shot learning in the
academic sense (that term means learning a brand-new class from a single
labeled example, e.g. a Siamese network recognizing a new face after
seeing it once). What's implemented here is human-in-the-loop feedback /
online retraining, which is the correct and much more reliable tool for
"the system was wrong about this entity, don't be wrong about entities
like it again":

  1. An analyst reviews a ranked entity in the dashboard (or CLI) and
     marks it "confirmed illicit" or "false positive".
  2. That correction is appended to a local CSV - data/pipeline_output/feedback.csv.
  3. On the next `python scripts/run_pipeline.py` run, cointrace.feedback.retrain
     re-fits the fusion weights against every correction on file, so the
     detectors that were actually right about that entity count for more
     and the ones that were wrong count for less.

Every function here only reads/writes a local file - no network calls -
so it works fully offline, same as the rest of the pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from cointrace import config

FEEDBACK_COLUMNS = ["entity_id", "label", "note", "timestamp"]


def _empty_feedback_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def load_feedback() -> pd.DataFrame:
    """Every analyst correction on file (full audit log, oldest first).
    label is 1 for confirmed illicit, 0 for confirmed benign/false positive."""
    if not config.FEEDBACK_CSV.exists():
        return _empty_feedback_df()
    df = pd.read_csv(config.FEEDBACK_CSV)
    for col in FEEDBACK_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[FEEDBACK_COLUMNS]


def record_feedback(entity_id: str, is_illicit: bool, note: str = "") -> pd.DataFrame:
    """Append one correction for entity_id. If the same entity is labeled
    again later (an analyst revising an earlier call), the new row is
    just appended - the full history stays for audit purposes, and
    latest_labels_only() below is what training actually reads."""
    config.PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_feedback()
    new_row = pd.DataFrame([{
        "entity_id": entity_id,
        "label": int(bool(is_illicit)),
        "note": note,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(config.FEEDBACK_CSV, index=False)
    return df


def latest_labels_only(df: pd.DataFrame | None = None) -> pd.Series:
    """Collapses the (possibly repeated, timestamped) feedback log down to
    one label per entity_id - the most recent correction - indexed by
    entity_id. This is what retraining is actually fit on."""
    df = load_feedback() if df is None else df
    if df.empty:
        return pd.Series(dtype=int)
    df = df.sort_values("timestamp")
    return (
        df.drop_duplicates("entity_id", keep="last")
        .set_index("entity_id")["label"]
        .astype(int)
    )
