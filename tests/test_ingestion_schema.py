import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cointrace.ingestion.schema import validate_record, validate_records, ValidationError

VALID = {
    "tx_id": "a" * 64,
    "timestamp": "2026-01-01T00:00:00",
    "src_wallet": "1AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "dst_wallet": "1BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "amount_btc": "0.5",
    "broadcast_ip": "8.8.8.8",
    "asn": "AS15169",
}


def test_valid_record_passes():
    out = validate_record(VALID)
    assert out["amount_btc"] == 0.5
    assert out["src_wallet"] == VALID["src_wallet"]


def test_missing_field_rejected():
    bad = dict(VALID)
    del bad["amount_btc"]
    try:
        validate_record(bad)
        assert False, "should have raised"
    except ValidationError:
        pass


def test_zero_amount_rejected():
    bad = dict(VALID, amount_btc="0")
    try:
        validate_record(bad)
        assert False, "should have raised"
    except ValidationError:
        pass


def test_self_transfer_rejected():
    bad = dict(VALID, dst_wallet=VALID["src_wallet"])
    try:
        validate_record(bad)
        assert False, "should have raised"
    except ValidationError:
        pass


def test_batch_validation_splits_clean_and_rejected():
    records = [VALID, dict(VALID, amount_btc="not_a_number")]
    clean, rejected = validate_records(records)
    assert len(clean) == 1
    assert len(rejected) == 1
    assert "reject_reason" in rejected[0]


if __name__ == "__main__":
    test_valid_record_passes()
    test_missing_field_rejected()
    test_zero_amount_rejected()
    test_self_transfer_rejected()
    test_batch_validation_splits_clean_and_rejected()
    print("All ingestion schema tests passed.")
