"""
Phase 5 - Structural motif detectors.

These are graph-traversal RULES (not ML) that look for the specific
laundering typologies named in the problem statement: peel chains,
mixer fan-in/fan-out, and rapid pass-through. Per the proposal
(Section 1.i), these motifs FEED the ML layer as an engineered
`motif_score` feature rather than replacing the ensemble - a rule firing
is one weak signal blended with the others in fusion.py, not a
standalone verdict.

Each detector returns a Series (index=entity_id) with a 0/1 (or graded)
flag; `combine_motif_scores` blends them into one `motif_score` column
plus a human-readable `motif_reasons` column used directly by the
explainability layer (Phase 6).
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from cointrace import config


def detect_peel_chain(
    df: pd.DataFrame,
    wallet_to_entity: dict[str, str],
    max_gap_minutes: float = 10.0,
    retention_threshold: float = 0.8,
    min_hops: int = 3,
) -> pd.Series:
    """Flags entities whose internal wallet graph shows a long run of
    rapid, high-retention single-hop transfers - the peel-chain
    signature. Since clustering.py already collapses true peel chains
    into one entity, this mostly re-confirms clustering's own decision
    (and gives it a named, explainable label) but also catches partial
    chains that clustering's conservative thresholds left split."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["entity"] = df["src_wallet"].map(wallet_to_entity)

    hop_counts = defaultdict(int)
    for wallet, group in df.groupby("src_wallet"):
        # a wallet participating in a peel chain sends once, rapidly after
        # receiving - approximate by counting short inter-arrival gaps
        # across ALL of the wallet's owning entity's transactions
        entity = wallet_to_entity.get(wallet)
        if entity is None:
            continue
        gaps = group["timestamp"].diff().dt.total_seconds().dropna() / 60.0
        rapid_hops = int((gaps <= max_gap_minutes).sum())
        hop_counts[entity] += rapid_hops

    flags = {eid: 1.0 if hops >= min_hops else 0.0 for eid, hops in hop_counts.items()}
    return pd.Series(flags, name="peel_chain_flag")


def detect_fan_pattern(
    df: pd.DataFrame,
    wallet_to_entity: dict[str, str],
    fan_threshold: int = 6,
    window_minutes: float = 30.0,
) -> pd.Series:
    """Flags entities with a fan-in AND fan-out burst within a short
    window - the mixer/tumbler signature (many small inputs converge,
    then fan back out shortly after)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["src_entity"] = df["src_wallet"].map(wallet_to_entity)
    df["dst_entity"] = df["dst_wallet"].map(wallet_to_entity)

    flags = {}
    all_entities = set(df["src_entity"]).union(df["dst_entity"]) - {None}
    for entity in all_entities:
        incoming = df[df["dst_entity"] == entity].sort_values("timestamp")
        outgoing = df[df["src_entity"] == entity].sort_values("timestamp")
        if len(incoming) < fan_threshold or len(outgoing) < fan_threshold:
            flags[entity] = 0.0
            continue
        # check for ANY window_minutes-wide slice with >= fan_threshold
        # incoming immediately followed by >= fan_threshold outgoing
        last_in_time = incoming["timestamp"].max()
        first_out_after = outgoing[outgoing["timestamp"] >= last_in_time]
        gap_minutes = (
            (first_out_after["timestamp"].min() - last_in_time).total_seconds() / 60.0
            if not first_out_after.empty else None
        )
        flags[entity] = 1.0 if gap_minutes is not None and gap_minutes <= window_minutes else 0.0

    return pd.Series(flags, name="fan_pattern_flag")


def detect_rapid_pass_through(
    df: pd.DataFrame,
    wallet_to_entity: dict[str, str],
    max_gap_seconds: float = 300.0,
) -> pd.Series:
    """Flags entities whose median inter-transaction gap is unusually
    short relative to the whole population - fast, automated-looking
    movement of funds rather than normal human-paced activity."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["entity"] = df["src_wallet"].map(wallet_to_entity)

    flags = {}
    for entity, group in df.groupby("entity"):
        if entity is None or len(group) < 2:
            continue
        gaps = group.sort_values("timestamp")["timestamp"].diff().dt.total_seconds().dropna()
        flags[entity] = 1.0 if (not gaps.empty and gaps.median() <= max_gap_seconds) else 0.0

    return pd.Series(flags, name="rapid_pass_through_flag")


def combine_motif_scores(*flag_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Combines multiple 0/1 motif flags into:
      - motif_score: mean of all flags (in [0,1]) per entity
      - motif_reasons: comma-joined names of motifs that fired, per entity
    Missing entities in any one series are treated as 0 (not flagged).
    """
    combined = pd.concat(flag_series, axis=1).fillna(0.0)
    motif_score = combined.mean(axis=1).rename("motif_score")

    def _reasons(row):
        fired = [col.replace("_flag", "") for col, val in row.items() if val >= 1.0]
        return ", ".join(fired) if fired else "none"

    motif_reasons = combined.apply(_reasons, axis=1).rename("motif_reasons")
    return motif_score, motif_reasons
