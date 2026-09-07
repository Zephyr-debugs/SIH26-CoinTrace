import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from cointrace.graph.clustering import build_entity_clusters, UnionFind


def _row(tx_id, ts, src, dst, amount, ip, asn="AS1"):
    return {
        "tx_id": tx_id, "timestamp": ts, "src_wallet": src, "dst_wallet": dst,
        "amount_btc": amount, "broadcast_ip": ip, "asn": asn,
    }


def test_union_find_basic():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("a") != uf.find("d")


def test_peel_chain_collapses_to_one_entity():
    # W0 -> W1 -> W2 -> W3, each hop retaining ~90%, all within seconds
    rows = [
        _row("t1", "2026-01-01T00:00:00", "W0", "W1", 10.0, "1.1.1.1"),
        _row("t2", "2026-01-01T00:00:30", "W1", "W2", 9.0, "2.2.2.2"),
        _row("t3", "2026-01-01T00:01:00", "W2", "W3", 8.1, "3.3.3.3"),
    ]
    df = pd.DataFrame(rows)
    wallet_to_entity = build_entity_clusters(df)
    entities = {wallet_to_entity[w] for w in ["W0", "W1", "W2", "W3"]}
    assert len(entities) == 1, f"expected 1 entity, got {entities}"


def test_benign_multiwallet_same_ip_collapses():
    rows = [
        _row("t1", "2026-01-01T00:00:00", "A1", "X", 1.0, "9.9.9.9"),
        _row("t2", "2026-01-05T00:00:00", "A2", "Y", 2.0, "9.9.9.9"),
    ]
    df = pd.DataFrame(rows)
    wallet_to_entity = build_entity_clusters(df)
    assert wallet_to_entity["A1"] == wallet_to_entity["A2"]


def test_unrelated_wallets_stay_separate():
    rows = [
        _row("t1", "2026-01-01T00:00:00", "A", "B", 1.0, "1.1.1.1"),
        _row("t2", "2026-01-02T00:00:00", "C", "D", 1.0, "2.2.2.2"),
    ]
    df = pd.DataFrame(rows)
    wallet_to_entity = build_entity_clusters(df)
    assert wallet_to_entity["A"] != wallet_to_entity["C"]


if __name__ == "__main__":
    test_union_find_basic()
    test_peel_chain_collapses_to_one_entity()
    test_benign_multiwallet_same_ip_collapses()
    test_unrelated_wallets_stay_separate()
    print("All clustering tests passed.")
