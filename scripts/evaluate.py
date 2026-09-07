"""
Phase 8 - Validation against the synthetic ground truth.

Compares data/pipeline_output/ranked_entities.csv against
data/synthetic/ground_truth.csv and reports precision/recall/F1 overall
and per typology, plus precision-at-K for a few K values - this is the
quantitative evidence for the write-up/demo (see proposal Section 1.iii,
"ground-truth-validated synthetic data").

Usage:
    python scripts/evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from cointrace import config
from cointrace.graph.clustering import build_entity_clusters


def build_entity_ground_truth() -> pd.DataFrame:
    """Ground truth is keyed by TRUE synthetic entity + its original
    wallet list. We need it keyed by CLUSTERED entity_id instead, since
    that's what the pipeline actually scores. A clustered entity is
    labeled illicit if ANY of its wallets came from an illicit true
    entity (conservative: mirrors how an investigator would treat it)."""
    df = pd.read_csv(config.TRANSACTIONS_CSV)
    gt = pd.read_csv(config.GROUND_TRUTH_CSV)
    wallet_to_entity = build_entity_clusters(df)

    wallet_label = {}
    wallet_typology = {}
    for row in gt.itertuples():
        for w in row.wallets.split(";"):
            wallet_label[w] = row.is_illicit
            wallet_typology[w] = row.typology

    rows = {}
    for w, clustered_id in wallet_to_entity.items():
        is_illicit = wallet_label.get(w, False)
        typology = wallet_typology.get(w, "none")
        if clustered_id not in rows or is_illicit:
            # if ANY wallet in the cluster is illicit, the cluster is illicit
            prev = rows.get(clustered_id, {"is_illicit": False, "typology": "none"})
            rows[clustered_id] = {
                "is_illicit": prev["is_illicit"] or is_illicit,
                "typology": typology if is_illicit else prev["typology"],
            }
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    ranked_path = config.PIPELINE_OUTPUT_DIR / "ranked_entities.csv"
    if not ranked_path.exists():
        print("No pipeline output found - run scripts/run_pipeline.py first.")
        return

    ranked = pd.read_csv(ranked_path, index_col=0)
    entity_gt = build_entity_ground_truth()

    merged = ranked.join(entity_gt, how="left")
    merged["is_illicit"] = merged["is_illicit"].fillna(False)

    y_true = merged["is_illicit"]
    y_pred = merged["is_alert"]

    tp = int(((y_true) & (y_pred)).sum())
    fp = int(((~y_true) & (y_pred)).sum())
    fn = int(((y_true) & (~y_pred)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"At threshold {config.RISK_ALERT_THRESHOLD}:")
    print(f"  precision: {precision:.3f}   recall: {recall:.3f}   f1: {f1:.3f}")
    print(f"  ({tp} true positives, {fp} false positives, {fn} false negatives)")
    print()

    print("Precision at top-K:")
    for k in (10, 25, 50, 100):
        topk = merged.head(k)
        p_at_k = topk["is_illicit"].mean()
        print(f"  P@{k}: {p_at_k:.3f}")
    print()

    print("Recall by typology (among flagged illicit true entities):")
    illicit = merged[merged["is_illicit"]]
    print(illicit.groupby("typology")["is_alert"].mean().to_string())


if __name__ == "__main__":
    main()
