"""
Phase 1 - Synthetic dataset generation.

Produces a population of wallets/transactions/IPs with injected, LABELED
laundering typologies (peel chains, mixer fan-in/out, structuring,
IP-hopping) plus a held-out ground-truth file. The ground truth is used
ONLY for evaluation (Phase 8) - never fed into the unsupervised detectors.

Output schema (one row per transaction, this is deliberately "flat" so it
can be round-tripped through CSV/JSON/XML in the ingestion layer):

    tx_id, timestamp, src_wallet, dst_wallet, amount_btc,
    broadcast_ip, asn, entity_id (hidden - stripped before ingestion)

`entity_id` and the ground-truth typology label are written to a SEPARATE
ground_truth.csv, keyed by entity_id, so the "raw data" the ingestion layer
sees never contains the answer key.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import random
import string
from dataclasses import dataclass, field

import pandas as pd

from cointrace import config


def _rand_wallet(rng: random.Random) -> str:
    return "1" + "".join(rng.choices(string.ascii_letters + string.digits, k=33))


def _rand_txid(rng: random.Random) -> str:
    return "".join(rng.choices("0123456789abcdef", k=64))


def _rand_ip(rng: random.Random, asn_pool: list[tuple[str, str]]) -> tuple[str, str]:
    """Return (ip, asn) drawn from a small pool of fake ASN blocks, so
    IP-diversity / ASN-diversity features have realistic structure."""
    base_ip, asn = rng.choice(asn_pool)
    net = ipaddress.ip_network(base_ip, strict=False)
    host = rng.randrange(1, net.num_addresses - 1)
    return str(net[host]), asn


@dataclass
class GeneratorState:
    rng: random.Random
    start_time: dt.datetime
    asn_pool: list[tuple[str, str]]
    rows: list[dict] = field(default_factory=list)
    ground_truth: list[dict] = field(default_factory=list)
    entity_counter: int = 0

    def new_entity_id(self, prefix: str) -> str:
        self.entity_counter += 1
        return f"{prefix}_{self.entity_counter:04d}"


def _emit_tx(state, ts, src, dst, amount, ip, asn, tx_id=None):
    state.rows.append(
        {
            "tx_id": tx_id or _rand_txid(state.rng),
            "timestamp": ts.isoformat(),
            "src_wallet": src,
            "dst_wallet": dst,
            "amount_btc": round(amount, 8),
            "broadcast_ip": ip,
            "asn": asn,
        }
    )


def _generate_benign_entities(state: GeneratorState):
    for _ in range(config.N_BENIGN_ENTITIES):
        entity_id = state.new_entity_id("benign")
        n_wallets = state.rng.randint(*config.N_WALLETS_PER_BENIGN_ENTITY)
        wallets = [_rand_wallet(state.rng) for _ in range(n_wallets)]
        # benign entities mostly use one consistent IP/ASN (their home network)
        ip, asn = _rand_ip(state.rng, state.asn_pool)

        n_tx = state.rng.randint(*config.N_TX_PER_BENIGN_ENTITY)
        t = state.start_time + dt.timedelta(days=state.rng.randint(0, 25))
        for _ in range(n_tx):
            t += dt.timedelta(hours=state.rng.uniform(2, 48))
            src = state.rng.choice(wallets)
            dst = _rand_wallet(state.rng)  # pays some external party
            amount = state.rng.uniform(0.001, 2.0)
            _emit_tx(state, t, src, dst, amount, ip, asn)

        state.ground_truth.append(
            {"entity_id": entity_id, "wallets": ";".join(wallets),
             "is_illicit": False, "typology": "none"}
        )


def _generate_peel_chains(state: GeneratorState):
    """A peel chain: a large input gets 'peeled' - most value forwarded on,
    a small change amount returned - repeated across a chain of hops,
    each hop from a different broadcast IP (laundering pattern)."""
    for _ in range(config.N_PEEL_CHAINS):
        entity_id = state.new_entity_id("peel")
        chain_len = state.rng.randint(5, 10)
        wallets = [_rand_wallet(state.rng) for _ in range(chain_len + 1)]
        t = state.start_time + dt.timedelta(days=state.rng.randint(0, 25))
        # `holding` is what the CURRENT wallet in the chain actually has to
        # move onward - each hop forwards most of what it just received
        # (high retention), which is what makes a peel chain a peel chain.
        holding = state.rng.uniform(5, 20)

        for i in range(chain_len):
            t += dt.timedelta(minutes=state.rng.uniform(1, 8))  # rapid pass-through
            ip, asn = _rand_ip(state.rng, state.asn_pool)  # different IP each hop
            send_amount = holding * state.rng.uniform(0.85, 0.95)
            _emit_tx(state, t, wallets[i], wallets[i + 1], send_amount, ip, asn)
            holding = send_amount  # next wallet now holds (and will forward) this

        state.ground_truth.append(
            {"entity_id": entity_id, "wallets": ";".join(wallets),
             "is_illicit": True, "typology": "peel_chain"}
        )


def _generate_mixer_fan_patterns(state: GeneratorState):
    """Fan-in: many small inputs converge on one wallet in a short window.
    Fan-out: that wallet immediately fans back out to many new wallets.
    Classic mixer/tumbler signature."""
    for _ in range(config.N_MIXER_FAN_PATTERNS):
        entity_id = state.new_entity_id("mixer")
        hub = _rand_wallet(state.rng)
        n_in = state.rng.randint(6, 15)
        n_out = state.rng.randint(6, 15)
        wallets = [hub]
        t = state.start_time + dt.timedelta(days=state.rng.randint(0, 25))

        # fan-in: many distinct source IPs converging quickly
        for _ in range(n_in):
            src = _rand_wallet(state.rng)
            wallets.append(src)
            ip, asn = _rand_ip(state.rng, state.asn_pool)
            t += dt.timedelta(minutes=state.rng.uniform(0.5, 5))
            _emit_tx(state, t, src, hub, state.rng.uniform(0.01, 0.3), ip, asn)

        # short pause, then fan-out
        t += dt.timedelta(minutes=state.rng.uniform(5, 20))
        for _ in range(n_out):
            dst = _rand_wallet(state.rng)
            wallets.append(dst)
            ip, asn = _rand_ip(state.rng, state.asn_pool)
            t += dt.timedelta(minutes=state.rng.uniform(0.5, 5))
            _emit_tx(state, t, hub, dst, state.rng.uniform(0.01, 0.3), ip, asn)

        state.ground_truth.append(
            {"entity_id": entity_id, "wallets": ";".join(wallets),
             "is_illicit": True, "typology": "mixer_fan_in_out"}
        )


def _generate_structuring(state: GeneratorState):
    """Structuring / smurting: one controller splits a large sum into many
    sub-threshold transactions across many wallets, staggered over time to
    avoid any single large/obvious transaction."""
    for _ in range(config.N_STRUCTURING_RINGS):
        entity_id = state.new_entity_id("structuring")
        controller = _rand_wallet(state.rng)
        n_wallets = state.rng.randint(8, 20)
        wallets = [controller] + [_rand_wallet(state.rng) for _ in range(n_wallets)]
        ip, asn = _rand_ip(state.rng, state.asn_pool)
        t = state.start_time + dt.timedelta(days=state.rng.randint(0, 25))

        for w in wallets[1:]:
            t += dt.timedelta(hours=state.rng.uniform(1, 6))  # staggered, not rapid
            amount = state.rng.uniform(0.05, 0.09)  # deliberately small/similar
            _emit_tx(state, t, controller, w, amount, ip, asn)

        state.ground_truth.append(
            {"entity_id": entity_id, "wallets": ";".join(wallets),
             "is_illicit": True, "typology": "structuring"}
        )


def _generate_ip_hopping(state: GeneratorState):
    """Same wallet, but every single broadcast comes from a different
    ASN/IP block - a strong network-obfuscation signal (Tor/VPN rotation)
    independent of any on-chain pattern."""
    for _ in range(config.N_IP_HOPPING_ENTITIES):
        entity_id = state.new_entity_id("iphop")
        wallet = _rand_wallet(state.rng)
        n_tx = state.rng.randint(8, 15)
        t = state.start_time + dt.timedelta(days=state.rng.randint(0, 25))

        for _ in range(n_tx):
            t += dt.timedelta(hours=state.rng.uniform(1, 10))
            ip, asn = _rand_ip(state.rng, state.asn_pool)  # re-rolled every tx
            dst = _rand_wallet(state.rng)
            _emit_tx(state, t, wallet, dst, state.rng.uniform(0.01, 1.0), ip, asn)

        state.ground_truth.append(
            {"entity_id": entity_id, "wallets": wallet,
             "is_illicit": True, "typology": "ip_hopping"}
        )


def generate(seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the full synthetic dataset. Returns (transactions_df, ground_truth_df)."""
    rng = random.Random(seed)

    # A handful of fake ASN blocks so IP/ASN diversity features are meaningful
    asn_pool = [
        (f"{a}.{b}.0.0/16", f"AS{1000 + i}")
        for i, (a, b) in enumerate([(41, 12), (77, 88), (103, 5), (185, 220),
                                     (192, 33), (203, 7), (45, 90), (91, 200)])
    ]

    state = GeneratorState(
        rng=rng,
        start_time=dt.datetime(2026, 1, 1),
        asn_pool=asn_pool,
    )

    _generate_benign_entities(state)
    _generate_peel_chains(state)
    _generate_mixer_fan_patterns(state)
    _generate_structuring(state)
    _generate_ip_hopping(state)

    tx_df = pd.DataFrame(state.rows).sort_values("timestamp").reset_index(drop=True)
    gt_df = pd.DataFrame(state.ground_truth)
    return tx_df, gt_df


def generate_and_save(seed: int = config.RANDOM_SEED) -> None:
    config.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    tx_df, gt_df = generate(seed=seed)
    tx_df.to_csv(config.TRANSACTIONS_CSV, index=False)
    gt_df.to_csv(config.GROUND_TRUTH_CSV, index=False)
    print(f"Wrote {len(tx_df):,} transactions -> {config.TRANSACTIONS_CSV}")
    print(f"Wrote {len(gt_df):,} entities (ground truth) -> {config.GROUND_TRUTH_CSV}")
    print(gt_df["typology"].value_counts().to_string())


if __name__ == "__main__":
    generate_and_save()
