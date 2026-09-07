"""
Phase 5 - Graph community detection (Louvain).

Uses NetworkX's built-in `louvain_communities` (native since networkx 3.0,
so no separate `python-louvain` dependency is required). Communities
themselves aren't a risk score - they're used to build a
"community_isolation" feature: entities in very small, disconnected
communities relative to the rest of the graph are structurally unusual,
which the fusion layer can weigh in.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from cointrace.graph.builder import wallet_node


def detect_communities(g: nx.MultiDiGraph, seed: int = 42) -> dict[str, int]:
    """Runs Louvain on the undirected, simple-graph projection of the
    wallet<->transaction structure (Louvain needs an undirected simple
    graph). Returns {node_id: community_index}."""
    undirected_simple = nx.Graph(g)  # collapses multi-edges, drops direction
    communities = nx.algorithms.community.louvain_communities(
        undirected_simple, seed=seed
    )
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx
    return node_to_community


def entity_community_score(
    wallet_to_entity: dict[str, str],
    node_to_community: dict[str, int],
) -> pd.Series:
    """A simple structural-isolation proxy per entity: 1 / community_size
    of the community its (first) wallet belongs to, normalized to [0,1]
    across entities. Small/unusual communities score higher."""
    # community sizes (counting wallet nodes only, so tx/IP/ASN nodes
    # don't skew what should be a wallet-clustering signal)
    from collections import Counter
    wallet_communities = {
        w: node_to_community.get(wallet_node(w)) for w in wallet_to_entity
    }
    community_sizes = Counter(c for c in wallet_communities.values() if c is not None)

    entity_scores = {}
    for wallet, entity_id in wallet_to_entity.items():
        comm = wallet_communities.get(wallet)
        size = community_sizes.get(comm, 1)
        entity_scores.setdefault(entity_id, []).append(1.0 / size)

    # an entity may span multiple wallets/communities post-clustering;
    # take the max isolation score seen across its wallets
    reduced = {eid: max(scores) for eid, scores in entity_scores.items()}
    s = pd.Series(reduced, name="community_isolation_score")
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s
