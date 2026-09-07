"""
Phase 2 - Canonical schema + validation.

Every format adapter (CSV/JSON/XML) converts its input into a list of
dicts matching CANONICAL_FIELDS, then `validate_records` splits that
list into (clean_records, rejected_records) so malformed rows are
quarantined and logged rather than silently dropped or silently crashing
the pipeline.
"""
from __future__ import annotations

import datetime as dt

CANONICAL_FIELDS = [
    "tx_id", "timestamp", "src_wallet", "dst_wallet",
    "amount_btc", "broadcast_ip", "asn",
]


class ValidationError(ValueError):
    pass


def _require(record: dict, field: str):
    if field not in record or record[field] in (None, ""):
        raise ValidationError(f"missing required field '{field}'")


def validate_record(record: dict) -> dict:
    """Validate + coerce a single raw record into the canonical schema.
    Raises ValidationError on anything unrecoverable."""
    for f in CANONICAL_FIELDS:
        _require(record, f)

    out = dict(record)

    # timestamp -> ISO 8601 string, but confirm it actually parses
    try:
        ts = out["timestamp"]
        if isinstance(ts, (int, float)):
            parsed = dt.datetime.fromtimestamp(ts)
        else:
            parsed = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        out["timestamp"] = parsed.isoformat()
    except Exception as e:
        raise ValidationError(f"unparseable timestamp {record.get('timestamp')!r}: {e}")

    # amount must be a positive float
    try:
        amount = float(out["amount_btc"])
    except (TypeError, ValueError):
        raise ValidationError(f"unparseable amount_btc {record.get('amount_btc')!r}")
    if amount <= 0:
        raise ValidationError(f"amount_btc must be positive, got {amount}")
    out["amount_btc"] = amount

    # wallets/tx_id must be non-trivial strings
    for f in ("tx_id", "src_wallet", "dst_wallet"):
        out[f] = str(out[f]).strip()
        if len(out[f]) < 4:
            raise ValidationError(f"'{f}' looks too short to be valid: {out[f]!r}")

    if out["src_wallet"] == out["dst_wallet"]:
        raise ValidationError("src_wallet and dst_wallet are identical (self-transfer)")

    out["broadcast_ip"] = str(out["broadcast_ip"]).strip()
    out["asn"] = str(out["asn"]).strip()

    return out


def validate_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (clean, rejected). Each rejected entry keeps the original
    record plus a 'reject_reason' field, so the quarantine log is useful."""
    clean, rejected = [], []
    for r in records:
        try:
            clean.append(validate_record(r))
        except ValidationError as e:
            rejected.append({**r, "reject_reason": str(e)})
    return clean, rejected
