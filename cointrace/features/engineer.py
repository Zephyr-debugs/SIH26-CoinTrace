"""
Phase 4 - Feature engineering.

Takes the canonical transaction DataFrame + a {wallet: entity_id} mapping
(from graph/clustering.py) and produces one feature row per entity,
matching config.FEATURE_COLUMNS. This is what feeds the detection
ensemble in Phase 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cointrace import config


def _safe_std(x: pd.Series) -> float:
    return float(x.std()) if len(x) > 1 else 0.0


def build_entity_features(df: pd.DataFrame, wallet_to_entity: dict[str, str]) -> pd.DataFrame:
    """df: canonical transaction rows (tx_id, timestamp, src_wallet,
    dst_wallet, amount_btc, broadcast_ip, asn, ...).
    Returns a DataFrame indexed by entity_id with config.FEATURE_COLUMNS.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["src_entity"] = df["src_wallet"].map(wallet_to_entity)
    df["dst_entity"] = df["dst_wallet"].map(wallet_to_entity)

    # An entity "touches" a transaction whether it's the sender or receiver -
    # build one long frame of (entity, tx, role, amount, ip, asn, ts) so every
    # feature can be computed with simple groupby aggregations.
    as_sender = df.rename(columns={"src_entity": "entity"})[
        ["entity", "tx_id", "amount_btc", "broadcast_ip", "asn", "timestamp"]
    ].assign(role="sender")
    as_receiver = df.rename(columns={"dst_entity": "entity"})[
        ["entity", "tx_id", "amount_btc", "broadcast_ip", "asn", "timestamp"]
    ].assign(role="receiver")
    touches = pd.concat([as_sender, as_receiver], ignore_index=True).dropna(subset=["entity"])

    rows = []
    for entity_id, g in touches.groupby("entity"):
        g = g.sort_values("timestamp")
        amounts = g["amount_btc"]
        inter_times = g["timestamp"].diff().dt.total_seconds().dropna()

        fan_in = int((g["role"] == "receiver").sum())
        fan_out = int((g["role"] == "sender").sum())
        distinct_ip = g["broadcast_ip"].nunique()
        distinct_asn = g["asn"].nunique()
        tx_count = g["tx_id"].nunique()

        # network-obfuscation score: how spread out are broadcasts relative
        # to activity volume? 0 = always same IP (consistent fingerprint),
        # closer to 1 = a different IP almost every time (correlation-break).
        obfuscation = (distinct_ip - 1) / max(tx_count - 1, 1)
        obfuscation = float(min(max(obfuscation, 0.0), 1.0))

        amount_mean = float(amounts.mean())
        amount_std = _safe_std(amounts)
        amount_cv = amount_std / amount_mean if amount_mean > 0 else 0.0

        # round-trip time: for entities that both receive and send, how
        # quickly (on average) does received value get forwarded onward?
        # Rough proxy: overall span of activity / number of hops.
        span_seconds = (g["timestamp"].max() - g["timestamp"].min()).total_seconds()
        round_trip = span_seconds / max(tx_count, 1)

        rows.append(
            {
                "entity_id": entity_id,
                "tx_count": tx_count,
                "fan_in": fan_in,
                "fan_out": fan_out,
                "amount_mean": amount_mean,
                "amount_std": amount_std,
                "amount_cv": amount_cv,
                "inter_tx_time_mean": float(inter_times.mean()) if len(inter_times) else 0.0,
                "inter_tx_time_std": _safe_std(inter_times),
                "distinct_ip_count": distinct_ip,
                "distinct_asn_count": distinct_asn,
                "network_obfuscation_score": obfuscation,
                "round_trip_time_mean": round_trip,
            }
        )

    feat_df = pd.DataFrame(rows).set_index("entity_id")
    # ensure column order matches config.FEATURE_COLUMNS exactly
    feat_df = feat_df[config.FEATURE_COLUMNS]
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat_df


def build_entity_geo(df: pd.DataFrame, wallet_to_entity: dict[str, str]) -> pd.DataFrame:
    """One row per entity_id with a representative broadcast location
    (mode country/city/lat/lon across all its transactions) plus a count
    of distinct countries seen - a wide geographic footprint is itself a
    signal, not just a display nicety.

    Requires df to have been ingested with enrich=True (i.e. GeoLite2
    .mmdb files present). If the geo columns aren't there - enrichment
    was off, or no .mmdb files were found - returns an empty frame with
    the right shape so callers can join it in either way.
    """
    geo_cols = {"country", "city", "latitude", "longitude"}
    if not geo_cols.issubset(df.columns):
        return pd.DataFrame(
            columns=["country", "city", "latitude", "longitude", "distinct_country_count"]
        ).rename_axis("entity_id")

    df = df.copy()
    df["src_entity"] = df["src_wallet"].map(wallet_to_entity)
    df["dst_entity"] = df["dst_wallet"].map(wallet_to_entity)
    as_sender = df.rename(columns={"src_entity": "entity"})[
        ["entity", "country", "city", "latitude", "longitude"]
    ]
    as_receiver = df.rename(columns={"dst_entity": "entity"})[
        ["entity", "country", "city", "latitude", "longitude"]
    ]
    touches = pd.concat([as_sender, as_receiver], ignore_index=True).dropna(subset=["entity"])

    rows = []
    for entity_id, g in touches.groupby("entity"):
        distinct_countries = int(g["country"].dropna().nunique())
        located = g.dropna(subset=["latitude", "longitude"])
        if located.empty:
            rows.append({"entity_id": entity_id, "country": None, "city": None,
                         "latitude": None, "longitude": None,
                         "distinct_country_count": distinct_countries})
            continue

        # representative point: most common (country, city) pair for this
        # entity, then the most common lat/lon within that pair. dropna=False
        # because MaxMind sometimes returns lat/lon with no city name (rural/
        # low-precision lookups) - those still count as a valid location.
        top_country, top_city = located.groupby(["country", "city"], dropna=False).size().idxmax()
        top_mask = (located["country"] == top_country if pd.notna(top_country)
                    else located["country"].isna())
        top_mask &= (located["city"] == top_city if pd.notna(top_city)
                     else located["city"].isna())
        top_point = located[top_mask]
        rows.append({
            "entity_id": entity_id,
            "country": top_country,
            "city": top_city,
            "latitude": float(top_point["latitude"].mode().iloc[0]),
            "longitude": float(top_point["longitude"].mode().iloc[0]),
            "distinct_country_count": distinct_countries,
        })

    return pd.DataFrame(rows).set_index("entity_id")
