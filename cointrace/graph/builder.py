"""
Phase 3 - Multi-layer graph construction.

Builds one heterogeneous NetworkX MultiDiGraph with four node types:
  Wallet:<addr>        Transaction:<tx_id>       IP:<addr>       ASN:<asn>

Edge types:
  wallet -> tx   ("sends")            tx -> wallet   ("receives")
  tx -> IP       ("broadcast_from")   IP -> ASN      ("belongs_to")

This graph is intentionally kept behind a thin builder function so the
backend (NetworkX now, Neo4j later) can be swapped without touching
anything downstream - callers only ever get back a NetworkX graph object
and query it with the helper functions in this module.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def wallet_node(addr: str) -> str:
    return f"Wallet:{addr}"


def tx_node(tx_id: str) -> str:
    return f"Transaction:{tx_id}"


def ip_node(ip: str) -> str:
    return f"IP:{ip}"


def asn_node(asn: str) -> str:
    return f"ASN:{asn}"


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """df must have columns: tx_id, timestamp, src_wallet, dst_wallet,
    amount_btc, broadcast_ip, asn (the canonical ingestion schema)."""
    g = nx.MultiDiGraph()

    for row in df.itertuples(index=False):
        w_src, w_dst = wallet_node(row.src_wallet), wallet_node(row.dst_wallet)
        t_node = tx_node(row.tx_id)
        i_node = ip_node(row.broadcast_ip)
        a_node = asn_node(row.asn)

        g.add_node(w_src, kind="Wallet", addr=row.src_wallet)
        g.add_node(w_dst, kind="Wallet", addr=row.dst_wallet)
        g.add_node(t_node, kind="Transaction", tx_id=row.tx_id,
                   timestamp=row.timestamp, amount_btc=row.amount_btc)
        g.add_node(i_node, kind="IP", addr=row.broadcast_ip)
        g.add_node(a_node, kind="ASN", asn=row.asn)

        g.add_edge(w_src, t_node, kind="sends", amount_btc=row.amount_btc,
                   timestamp=row.timestamp)
        g.add_edge(t_node, w_dst, kind="receives", amount_btc=row.amount_btc,
                   timestamp=row.timestamp)
        g.add_edge(t_node, i_node, kind="broadcast_from", timestamp=row.timestamp)
        g.add_edge(i_node, a_node, kind="belongs_to")

    return g


def nodes_of_kind(g: nx.MultiDiGraph, kind: str) -> list[str]:
    return [n for n, d in g.nodes(data=True) if d.get("kind") == kind]


def wallet_transactions(g: nx.MultiDiGraph, wallet_addr: str) -> list[str]:
    """All transaction node ids touching this wallet (sent OR received)."""
    w = wallet_node(wallet_addr)
    out_tx = [v for _, v, d in g.out_edges(w, data=True) if d.get("kind") == "sends"]
    in_tx = [u for u, _, d in g.in_edges(w, data=True) if d.get("kind") == "receives"]
    return list(set(out_tx + in_tx))
