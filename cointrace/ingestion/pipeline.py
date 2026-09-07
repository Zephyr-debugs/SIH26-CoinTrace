"""
Phase 2 - Ties adapters + schema validation + GeoIP enrichment together
into one function: raw file path in, clean canonical DataFrame out
(plus a rejects DataFrame for the quarantine log).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cointrace.ingestion.adapters import load_any
from cointrace.ingestion.geoip import GeoEnricher
from cointrace.ingestion.schema import validate_records


def ingest_file(path: str | Path, enrich: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clean_df, rejected_df)."""
    raw_records = load_any(path)
    clean, rejected = validate_records(raw_records)

    clean_df = pd.DataFrame(clean)
    rejected_df = pd.DataFrame(rejected)

    if enrich and not clean_df.empty:
        enricher = GeoEnricher()
        enriched = clean_df["broadcast_ip"].apply(enricher.enrich).apply(pd.Series)
        clean_df = pd.concat([clean_df.reset_index(drop=True), enriched], axis=1)
        enricher.close()

    return clean_df, rejected_df


def ingest_files(paths: list[str | Path], enrich: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest multiple files (mixed formats OK) and concatenate results."""
    clean_frames, rejected_frames = [], []
    for p in paths:
        c, r = ingest_file(p, enrich=enrich)
        clean_frames.append(c)
        rejected_frames.append(r)
    clean_df = pd.concat(clean_frames, ignore_index=True) if clean_frames else pd.DataFrame()
    rejected_df = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
    return clean_df, rejected_df
