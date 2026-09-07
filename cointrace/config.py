"""
Central place for paths and tunable constants, so no module hardcodes
a path or a magic number that another module needs to agree on.
"""
from pathlib import Path

# ---- Paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
GEOIP_DIR = DATA_DIR / "geoip"
PIPELINE_OUTPUT_DIR = DATA_DIR / "pipeline_output"

GEOIP_CITY_DB = GEOIP_DIR / "GeoLite2-City.mmdb"
GEOIP_ASN_DB = GEOIP_DIR / "GeoLite2-ASN.mmdb"

TRANSACTIONS_CSV = SYNTHETIC_DIR / "transactions.csv"
GROUND_TRUTH_CSV = SYNTHETIC_DIR / "ground_truth.csv"

# ---- Synthetic data generation ----------------------------------------
RANDOM_SEED = 42

N_BENIGN_ENTITIES = 400
N_WALLETS_PER_BENIGN_ENTITY = (1, 3)      # inclusive range
N_TX_PER_BENIGN_ENTITY = (2, 12)

N_PEEL_CHAINS = 15
N_MIXER_FAN_PATTERNS = 15
N_STRUCTURING_RINGS = 10
N_IP_HOPPING_ENTITIES = 10

# ---- Graph / clustering -------------------------------------------------
# Max time gap (seconds) for an IP broadcast to be considered
# "timing-correlated" with a wallet's transaction.
IP_WALLET_TIMING_WINDOW_SECONDS = 30

# ---- Feature engineering -------------------------------------------------
FEATURE_COLUMNS = [
    "tx_count",
    "fan_in",
    "fan_out",
    "amount_mean",
    "amount_std",
    "amount_cv",                 # coefficient of variation
    "inter_tx_time_mean",
    "inter_tx_time_std",
    "distinct_ip_count",
    "distinct_asn_count",
    "network_obfuscation_score",  # derived from correlation-break signal
    "round_trip_time_mean",
]

# ---- Detection ensemble --------------------------------------------------
ISOLATION_FOREST_CONTAMINATION = 0.05
AUTOENCODER_HIDDEN_DIM = 8
AUTOENCODER_EPOCHS = 30
AUTOENCODER_LR = 1e-3

# Weighted blend of normalized [0,1] detector scores into one risk score.
# Keys must match the score column names produced by each detector.
FUSION_WEIGHTS = {
    "isolation_forest_score": 0.35,
    "autoencoder_score": 0.35,
    "motif_score": 0.30,
}

RISK_ALERT_THRESHOLD = 0.6   # entities above this are surfaced as "alerts" by default

# ---- Phase 9: analyst feedback / retraining -------------------------------
FEEDBACK_CSV = PIPELINE_OUTPUT_DIR / "feedback.csv"
LEARNED_WEIGHTS_JSON = PIPELINE_OUTPUT_DIR / "learned_fusion_weights.json"
# Minimum distinct labeled entities (with both classes present) before
# fusion weights are re-fit from feedback instead of using the static
# FUSION_WEIGHTS above.
MIN_FEEDBACK_FOR_RETRAIN = 8

# ---- Explainability -------------------------------------------------------
EVIDENCE_SUBGRAPH_HOPS = 2
TOP_N_FOR_EAGER_EXPLANATION = 50   # only precompute SHAP/evidence for top-N ranked entities
