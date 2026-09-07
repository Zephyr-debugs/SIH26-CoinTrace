# CoinTrace

AI-powered, offline, self-hosted monitoring & analysis of Bitcoin transaction traffic.
Fuses network-layer telemetry (IP/ASN/timing) with blockchain-layer data (wallets,
transactions) into a unified graph, runs an unsupervised anomaly-detection ensemble
over it, and surfaces ranked, explainable leads through a Streamlit dashboard.

See `ROADMAP.md` for the phased build plan this scaffold follows.

## Quickstart (PyCharm or terminal)

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic labeled data
python scripts/generate_data.py

# 4. Run the full pipeline (ingestion -> graph -> features -> detection -> explain)
python scripts/run_pipeline.py

# 5. Launch the dashboard
streamlit run cointrace/dashboard/app.py
```

Or just run `./run.sh` (Linux/Mac) once the venv is set up, which does steps 3-5.

## Project layout

```
cointrace/
  cointrace/                 # the actual Python package
    config.py                # central paths & tunable constants
    synth/generator.py        # Phase 1: synthetic typology-injected dataset
    ingestion/                 # Phase 2: parse + validate + GeoIP enrich
      adapters.py
      schema.py
      geoip.py
    graph/                     # Phase 3: multi-layer graph + entity clustering
      builder.py
      clustering.py
    features/engineer.py       # Phase 4: per-entity feature engineering
    detection/                 # Phase 5: anomaly-detection ensemble
      isolation_forest.py
      autoencoder.py
      community.py
      motifs.py
      fusion.py
    explain/                   # Phase 6: SHAP + evidence subgraphs
      shap_explain.py
      evidence.py
    dashboard/app.py           # Phase 7: Streamlit UI
  scripts/
    generate_data.py           # CLI: runs Phase 1
    run_pipeline.py             # CLI: runs Phases 2-6 end to end
  tests/                       # unit tests per module
  data/                        # gitignored: synthetic data + GeoLite2 .mmdb files
  requirements.txt
  run.sh
```

## GeoIP setup (optional but recommended)

Download the free MaxMind GeoLite2 City and ASN databases (requires a free
MaxMind account) and place them here:

```
data/geoip/GeoLite2-City.mmdb
data/geoip/GeoLite2-ASN.mmdb
```

https://dev.maxmind.com/geoip/geolite2-free-geolocation-data

If these files are absent, the ingestion layer still runs — GeoIP fields are
just left blank rather than failing, so you can build/test everything else
without downloading anything first.

## Notes on heavier dependencies

- **PyTorch** (autoencoder) and **SHAP** (explainability) are the two heaviest
  installs. Everything through the graph + Isolation Forest + Louvain
  detectors works without them. If you want to build incrementally, comment
  them out of `requirements.txt` at first and add them back for Phases 5-6.
- **Neo4j** is not required anywhere in this scaffold — it's a stretch-goal
  swap-in for the graph backend, mentioned in the roadmap but not wired up.
