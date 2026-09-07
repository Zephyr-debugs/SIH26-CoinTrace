"""
Phase 2 - Format adapters.

Each adapter's only job is: "turn this file into a list[dict] using the
CANONICAL_FIELDS key names". Validation/coercion happens later in
schema.py - adapters should stay dumb and format-specific.

Supported formats: CSV, JSON (list-of-objects or {"transactions": [...]}),
XML (a repeated element per transaction).
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from cointrace.ingestion.schema import CANONICAL_FIELDS


def load_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # accept either a bare list or a wrapper key like "transactions"/"records"/"data"
        for key in ("transactions", "records", "data"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError(
                "JSON object provided but no 'transactions'/'records'/'data' key found"
            )
    if not isinstance(data, list):
        raise ValueError("JSON input must resolve to a list of transaction objects")
    return data


def load_xml(path: str | Path, record_tag: str = "transaction") -> list[dict]:
    """Expects a structure like:
        <transactions>
          <transaction>
            <tx_id>...</tx_id>
            <timestamp>...</timestamp>
            ...
          </transaction>
        </transactions>
    """
    tree = ET.parse(path)
    root = tree.getroot()
    records = []
    for el in root.findall(f".//{record_tag}"):
        rec = {}
        for field in CANONICAL_FIELDS:
            child = el.find(field)
            rec[field] = child.text.strip() if child is not None and child.text else None
        # also grab any attributes in case the XML uses attributes instead of children
        rec.update({k: v for k, v in el.attrib.items() if k in CANONICAL_FIELDS})
        records.append(rec)
    return records


_LOADERS = {
    ".csv": load_csv,
    ".json": load_json,
    ".xml": load_xml,
}


def load_any(path: str | Path) -> list[dict]:
    """Dispatch to the right loader based on file extension."""
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Supported: {', '.join(_LOADERS)}"
        )
    return loader(path)
