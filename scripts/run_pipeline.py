"""
Runs the full CoinTrace pipeline end to end (Phases 2-6):
  ingest -> graph -> cluster -> features -> detect -> fuse -> explain

Writes results to data/pipeline_output/ for the dashboard (Phase 7) to
read. Detectors that need an optional dependency (torch for the
autoencoder, shap for explanations) are skipped gracefully with a
printed warning if that dependency isn't installed, so this always
produces SOME ranked output.

Usage:
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from cointrace import config
from cointrace.ingestion.pipeline import ingest_file
from cointrace.graph.builder import build_graph
from cointrace.graph.clustering import build_entity_clusters
from cointrace.features.engineer import build_entity_features, build_entity_geo
from cointrace.detection.isolation_forest import run_isolation_forest
from cointrace.detection.community import detect_communities, entity_community_score
from cointrace.detection.motifs import (
    detect_peel_chain, detect_fan_pattern, detect_rapid_pass_through, combine_motif_scores,
)
from cointrace.detection.fusion import fuse_scores
from cointrace.feedback.retrain import load_learned_weights, retrain_from_feedback
from cointrace.feedback.store import latest_labels_only, load_feedback


def main():
    config.PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Ingesting...")
    clean_df, rejected_df = ingest_file(config.TRANSACTIONS_CSV, enrich=True)
    print(f"      {len(clean_df):,} clean records, {len(rejected_df):,} rejected")
    rejected_df.to_csv(config.PIPELINE_OUTPUT_DIR / "rejected_records.csv", index=False)

    print("[2/6] Building graph + entity clusters...")
    g = build_graph(clean_df)
    wallet_to_entity = build_entity_clusters(clean_df)
    print(f"      graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")
    print(f"      {len(set(wallet_to_entity.values())):,} clustered entities")

    print("[3/6] Engineering features...")
    feat_df = build_entity_features(clean_df, wallet_to_entity)
    geo_df = build_entity_geo(clean_df, wallet_to_entity)

    print("[4/6] Running detection ensemble...")
    scores = {}

    if_model_scores = run_isolation_forest(feat_df)
    scores["isolation_forest_score"] = if_model_scores

    try:
        from cointrace.detection.autoencoder import run_autoencoder
        scores["autoencoder_score"] = run_autoencoder(feat_df)
        print("      autoencoder: ok")
    except ImportError as e:
        print(f"      autoencoder: skipped ({e})")

    peel = detect_peel_chain(clean_df, wallet_to_entity)
    fan = detect_fan_pattern(clean_df, wallet_to_entity)
    rapid = detect_rapid_pass_through(clean_df, wallet_to_entity)
    motif_score, motif_reasons = combine_motif_scores(peel, fan, rapid)
    scores["motif_score"] = motif_score

    node_to_community = detect_communities(g)
    community_score = entity_community_score(wallet_to_entity, node_to_community)

    print("[5/6] Fusing scores...")
    learned_weights = load_learned_weights()
    n_labeled = len(latest_labels_only(load_feedback()))
    if learned_weights:
        print(f"      using feedback-learned weights "
              f"(fit on {n_labeled} analyst-labeled entities so far): {learned_weights}")
        result = fuse_scores(scores, weights=learned_weights)
    else:
        print(f"      using default config.FUSION_WEIGHTS "
              f"({n_labeled} analyst-labeled entities so far, "
              f"need {config.MIN_FEEDBACK_FOR_RETRAIN} of both classes to start learning)")
        result = fuse_scores(scores)
    result = result.join(feat_df, how="left")
    result = result.join(motif_reasons, how="left")
    result = result.join(community_score, how="left")
    result = result.join(geo_df, how="left")
    result["motif_reasons"] = result["motif_reasons"].fillna("none")

    print("[6/6] Writing output...")
    out_path = config.PIPELINE_OUTPUT_DIR / "ranked_entities.csv"
    result.to_csv(out_path)
    print(f"      -> {out_path}")
    print(f"      {result['is_alert'].sum()} entities above alert threshold "
          f"({config.RISK_ALERT_THRESHOLD})")

    print("\nTop 10 ranked entities:")
    cols = ["risk_score", "isolation_forest_score", "motif_score", "motif_reasons"]
    cols = [c for c in cols if c in result.columns]
    print(result[cols].head(10).to_string())

    # Phase 9: if enough analyst corrections have accumulated (dashboard
    # "Confirm illicit" / "Mark false positive" buttons, or
    # scripts/record_feedback.py), re-fit the fusion weights against them
    # now, so the NEXT run of this script uses them automatically. This
    # is the offline "learn from its mistakes" loop - see
    # cointrace/feedback/retrain.py for what it actually does and why
    # it's not "one-shot learning".
    new_weights = retrain_from_feedback(scores)
    if new_weights:
        print(f"\n[feedback] Re-fit fusion weights from {n_labeled} analyst corrections: "
              f"{new_weights}\n           Saved to {config.LEARNED_WEIGHTS_JSON} - "
              f"the next run will use these instead of the defaults.")


if __name__ == "__main__":
    main()
