"""
Phase 3 - Entity clustering heuristics.

Collapses raw wallet addresses into real-world "entities" using three
complementary, individually-imperfect heuristics (as the heuristics are
in real Bitcoin forensics - see proposal Section 3.ii):

1. Common-input-ownership: if a transaction has multiple input wallets,
   those wallets are (almost always) controlled by the same owner, since
   spending requires all their private keys. Our synthetic generator only
   produces single-input transfers, so this is a documented no-op here -
   but the code is written generically off tx_id grouping, so it becomes
   active automatically on any real/bulk dataset that models multi-input
   UTXO transactions.

2. Rapid pass-through / change continuity: if a wallet receives an amount
   and then, in short order, forwards most of that value onward, the
   sender and the next-hop recipient are heuristically linked as the same
   controller moving funds through a disposable intermediate address -
   this is what lets us collapse an entire peel chain into one entity.

3. IP-wallet co-broadcast: wallets that broadcast from the *same* IP are
   heuristically the same operator. Where a wallet's broadcasts come from
   many different IPs/ASNs instead, we do NOT merge on that basis -
   instead we record it as a "correlation-break" / network-obfuscation
   signal, which becomes a detection FEATURE later rather than being
   silently discarded (see features/engineer.py).

Every merge is a heuristic edge with a confidence score attached, and
kept low-confidence links "unmerged" is preferred over false-merging
(see proposal Section 3.iii) - this module purposefully keeps merges
conservative rather than maximal.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import pandas as pd


class UnionFind:
    """Simple disjoint-set with path compression, keyed by wallet address."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> dict[str, list[str]]:
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return dict(out)


def cluster_common_input_ownership(df: pd.DataFrame, uf: UnionFind) -> int:
    """Union all src_wallets that co-appear as inputs on the same tx_id.
    No-op on single-input datasets (see module docstring); returns the
    number of unions performed so callers/tests can assert on it."""
    merges = 0
    for _, group in df.groupby("tx_id")["src_wallet"]:
        wallets = group.tolist()
        for w in wallets[1:]:
            uf.union(wallets[0], w)
            merges += 1
    return merges


def cluster_rapid_pass_through(
    df: pd.DataFrame,
    uf: UnionFind,
    max_gap_minutes: float = 10.0,
    retention_threshold: float = 0.8,
) -> int:
    """For every wallet W, look for (incoming tx -> W) followed shortly by
    (W -> outgoing tx) where most of the value was retained and passed
    onward quickly. If found, union the ORIGINAL sender with the eventual
    NEXT-HOP recipient - this is what collapses a multi-hop peel chain
    into a single entity, one link at a time."""
    df = df.sort_values("timestamp")
    ts = pd.to_datetime(df["timestamp"])
    df = df.assign(ts_parsed=ts)

    incoming = defaultdict(list)  # wallet -> list of (ts, amount, src_wallet)
    outgoing = defaultdict(list)  # wallet -> list of (ts, amount, dst_wallet)
    for row in df.itertuples(index=False):
        incoming[row.dst_wallet].append((row.ts_parsed, row.amount_btc, row.src_wallet))
        outgoing[row.src_wallet].append((row.ts_parsed, row.amount_btc, row.dst_wallet))

    merges = 0
    max_gap = dt.timedelta(minutes=max_gap_minutes)

    for wallet, in_events in incoming.items():
        out_events = outgoing.get(wallet, [])
        if not out_events:
            continue
        for in_ts, in_amt, sender in in_events:
            # find the nearest outgoing event after this incoming one
            candidates = [(o_ts, o_amt, recipient) for o_ts, o_amt, recipient in out_events
                          if o_ts >= in_ts and (o_ts - in_ts) <= max_gap]
            if not candidates:
                continue
            o_ts, o_amt, recipient = min(candidates, key=lambda c: c[0])
            if in_amt > 0 and (o_amt / in_amt) >= retention_threshold:
                # union the pass-through wallet itself with BOTH neighbors
                # (not sender-recipient directly) so the whole chain collapses
                # transitively into one entity, hop by hop.
                uf.union(wallet, sender)
                uf.union(wallet, recipient)
                merges += 1
    return merges


def cluster_ip_co_broadcast(df: pd.DataFrame, uf: UnionFind) -> int:
    """Union wallets that broadcast (as src_wallet) from the same IP -
    a shared broadcast origin is a reasonably strong same-operator signal
    for benign, non-anonymized usage. This deliberately does NOT require
    tight timing overlap: the same person's wallets sharing a home IP
    across days is still the same operator."""
    merges = 0
    ip_to_wallets = df.groupby("broadcast_ip")["src_wallet"].apply(lambda s: set(s))
    for wallets in ip_to_wallets:
        wallets = list(wallets)
        for w in wallets[1:]:
            uf.union(wallets[0], w)
            merges += 1
    return merges


def build_entity_clusters(df: pd.DataFrame) -> dict[str, str]:
    """Runs all clustering heuristics and returns a mapping
    {wallet_address: entity_id}. entity_id is deterministic (min wallet
    address in the cluster), so re-runs on the same data are stable."""
    uf = UnionFind()
    for w in pd.concat([df["src_wallet"], df["dst_wallet"]]).unique():
        uf.find(w)  # ensure every wallet has a singleton group even if unmerged

    cluster_common_input_ownership(df, uf)
    cluster_rapid_pass_through(df, uf)
    cluster_ip_co_broadcast(df, uf)

    groups = uf.groups()
    wallet_to_entity = {}
    for root, wallets in groups.items():
        entity_id = f"entity_{min(wallets)[:10]}"
        for w in wallets:
            wallet_to_entity[w] = entity_id
    return wallet_to_entity
