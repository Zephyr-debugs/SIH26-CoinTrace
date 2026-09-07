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

import json
import sys
from datetime import datetime, timezone
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

    precision_at_k = {}
    print("Precision at top-K:")
    for k in (10, 25, 50, 100):
        topk = merged.head(k)
        p_at_k = float(topk["is_illicit"].mean()) if len(topk) else 0.0
        precision_at_k[k] = p_at_k
        print(f"  P@{k}: {p_at_k:.3f}")
    print()

    print("Recall by typology (among flagged illicit true entities):")
    illicit = merged[merged["is_illicit"]]
    recall_by_typology = illicit.groupby("typology")["is_alert"].mean()
    print(recall_by_typology.to_string())

    # --- Phase 8 (cont.): persist this run as a report file, so
    # precision/recall have a permanent artifact instead of only living
    # in terminal scrollback. Written every run as of/at the timestamp
    # below, so re-running after a feedback-driven reweight (Phase 9)
    # naturally produces a fresh before/after comparison.
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {
        "generated_at": generated_at,
        "risk_alert_threshold": config.RISK_ALERT_THRESHOLD,
        "n_entities_scored": int(len(merged)),
        "n_entities_true_illicit": int(y_true.sum()),
        "n_entities_flagged": int(y_pred.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "precision_at_k": {str(k): round(v, 4) for k, v in precision_at_k.items()},
        "recall_by_typology": {
            str(k): round(float(v), 4) for k, v in recall_by_typology.items()
        },
        "fusion_weights_used": (
            "learned_from_feedback" if config.LEARNED_WEIGHTS_JSON.exists()
            else "default_config_weights"
        ),
    }

    config.PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.PIPELINE_OUTPUT_DIR / "evaluation_report.json"
    json_path.write_text(json.dumps(report, indent=2))

    md_lines = [
        "# CoinTrace evaluation report",
        "",
        f"Generated: {generated_at}",
        f"Alert threshold: {config.RISK_ALERT_THRESHOLD}",
        f"Fusion weights used: {report['fusion_weights_used']}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {precision:.3f} |",
        f"| Recall | {recall:.3f} |",
        f"| F1 | {f1:.3f} |",
        f"| True positives | {tp} |",
        f"| False positives | {fp} |",
        f"| False negatives | {fn} |",
        "",
        "## Precision at top-K",
        "",
        "| K | Precision@K |",
        "|---|---|",
    ]
    md_lines += [f"| {k} | {v:.3f} |" for k, v in precision_at_k.items()]
    md_lines += [
        "",
        "## Recall by typology",
        "",
        "(share of true illicit entities of that typology that were flagged as alerts)",
        "",
        "| Typology | Recall |",
        "|---|---|",
    ]
    md_lines += [f"| {t} | {v:.3f} |" for t, v in recall_by_typology.items()]
    md_path = config.PIPELINE_OUTPUT_DIR / "evaluation_report.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"\nSaved report -> {json_path}")
    print(f"Saved report -> {md_path}")


if __name__ == "__main__":
    main()
