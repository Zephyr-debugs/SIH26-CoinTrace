"""
Phase 6 - Evidence subgraph extraction + plain-language explanations.

For a flagged entity, pulls the literal N-hop neighborhood around its
wallet(s) out of the full graph - this is what an investigator (or a
court) can inspect directly, rather than trusting an opaque score. Also
generates the "flagged: ... - consistent with a peel-chain typology"
style sentence the proposal describes (Section 1.i), built from the
motif reasons + top SHAP features rather than a canned template per
typology.
"""
from __future__ import annotations

import networkx as nx

from cointrace import config
from cointrace.graph.builder import wallet_node


def extract_evidence_subgraph(
    g: nx.MultiDiGraph,
    wallets: list[str],
    hops: int = config.EVIDENCE_SUBGRAPH_HOPS,
) -> nx.MultiDiGraph:
    """Returns the induced subgraph within `hops` of any of this entity's
    wallet nodes - the exact nodes/edges an investigator can inspect to
    verify why the entity was flagged."""
    seeds = [wallet_node(w) for w in wallets if wallet_node(w) in g]
    if not seeds:
        return nx.MultiDiGraph()

    undirected = g.to_undirected(as_view=True)
    nearby_nodes: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(undirected.neighbors(node))
        nearby_nodes.update(next_frontier)
        frontier = next_frontier

    return g.subgraph(nearby_nodes).copy()


def plain_language_explanation(
    entity_id: str,
    motif_reasons: str,
    distinct_ip_count: int,
    tx_count: int,
    round_trip_time_mean_seconds: float,
    top_shap_features: list[tuple[str, float]] | None = None,
) -> str:
    """Builds the human-readable explanation string, e.g.:
    "flagged: 6 distinct source IPs across 9 transactions, funds passed
    through in under 3 minutes on average - consistent with a peel-chain /
    rapid-pass-through typology."
    """
    parts = [f"{distinct_ip_count} distinct broadcast IP(s) across {tx_count} transactions"]

    if round_trip_time_mean_seconds < 3600:
        minutes = round_trip_time_mean_seconds / 60
        parts.append(f"funds moved onward in ~{minutes:.1f} minutes on average")

    typology_note = ""
    if motif_reasons and motif_reasons != "none":
        typology_note = f" — consistent with a {motif_reasons.replace('_', ' ')} typology"

    explanation = f"Flagged: {', '.join(parts)}{typology_note}."

    if top_shap_features:
        feat_str = ", ".join(f"{name} ({val:+.2f})" for name, val in top_shap_features)
        explanation += f" Top contributing features: {feat_str}."

    return explanation
