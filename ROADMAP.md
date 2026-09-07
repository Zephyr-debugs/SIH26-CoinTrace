# CoinTrace — Development Roadmap & Phases

*AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic (SIH Proposal)*

This roadmap translates the proposal's methodology into a buildable, sequenced plan — suitable for a hackathon sprint (SIH) or a slower, learn-as-you-go build on a personal laptop.

---

## Phase 0 — Project Setup (Day 0–1)

**Goal:** A working, version-controlled Python project skeleton, nothing functional yet.

- Initialize Git repo, `.gitignore`, `README.md`
- Create virtual environment (`venv` or `conda`)
- Set up `pyproject.toml` / `requirements.txt` with pinned versions
- Decide project structure, e.g.:
  ```
  cointrace/
    ingestion/
    graph/
    features/
    detection/
    explain/
    dashboard/
    data/            (synthetic + GeoLite2 db, gitignored)
    tests/
    run.sh
  ```
- Download MaxMind GeoLite2 City + ASN `.mmdb` files once (needs internet; used fully offline afterward)
- **Deliverable:** empty-but-runnable project, `streamlit run dashboard/app.py` shows a placeholder page

---

## Phase 1 — Synthetic Dataset Generation (Day 1–3)

**Goal:** Labeled, typology-injected ground-truth dataset to build and evaluate everything else against.

- Generate a base population of wallets, transactions, IPs, ASNs (random but plausible)
- Inject known laundering typologies with hidden ground-truth labels:
  - Peel chains
  - Mixer fan-in / fan-out
  - Structuring (smurfing)
  - Rapid IP-hopping
- Output as CSV/JSON/XML (to also exercise the ingestion layer's format-agnosticism)
- Keep a held-out `ground_truth.csv` (entity_id → is_illicit, typology) — used only for evaluation, never for training
- **Deliverable:** `data/synthetic/` with realistic bulk data + hidden ground truth

---

## Phase 2 — Ingestion & Normalization Layer (Day 3–5)

**Goal:** Turn raw CSV/JSON/XML into one canonical, validated schema.

- Format adapters behind a common interface (`csv`, `json`, `xml.etree.ElementTree`)
- Schema validation; malformed records go to a reject/quarantine log, not silently dropped
- GeoIP/ASN enrichment via local `.mmdb` lookups (geoip2)
- **Deliverable:** `ingestion/` module — raw files in, canonical enriched records out; unit tests on malformed-input handling

---

## Phase 3 — Graph Construction & Entity Clustering (Day 5–8)

**Goal:** Build the multi-layer graph and collapse addresses into real-world entities.

- NetworkX heterogeneous graph with 4 node types: `Wallet`, `Transaction`, `IP`, `ASN`
- Clustering heuristics:
  - Common-input-ownership
  - Change-address detection
  - IP–wallet timing correlation (and explicit "correlation-break" edges/features)
- Build the graph layer behind a thin interface so NetworkX can later be swapped for Neo4j
- **Deliverable:** `graph/` module producing an entity-resolved graph object from canonical records

---

## Phase 4 — Feature Engineering (Day 8–9)

**Goal:** Per-entity feature vectors for the ML layer.

- Tx frequency, fan-in/fan-out counts, amount variance
- IP/ASN diversity, inter-tx timing statistics, round-trip time
- Network-obfuscation score (from correlation-break signal)
- **Deliverable:** `features/` module — entity graph in, feature matrix out

---

## Phase 5 — AI/ML Detection Ensemble (Day 9–13)

**Goal:** A single calibrated risk score per entity, from complementary weak signals.

- Isolation Forest on tabular features (scikit-learn)
- Autoencoder on behavioral sequences (PyTorch, CPU-trainable)
- Louvain community detection on the graph (`python-louvain`)
- Motif detectors as graph traversal routines (peel chain, fan-in/out, rapid pass-through)
- Score fusion: normalize each detector's output to [0,1], combine via weighted sum (or simple logistic blend if labels are used only for tuning weights)
- **Deliverable:** `detection/` module — feature matrix + graph in, ranked risk score per entity out

---

## Phase 6 — Explainability Layer (Day 13–15)

**Goal:** Every alert is traceable, not a black-box number.

- SHAP attribution against Isolation Forest / autoencoder inputs
- 2-hop evidence subgraph extraction per flagged entity
- Motif-to-plain-language-explanation template mapping
- Cache SHAP explainer objects; compute subgraphs lazily (top-N or on-click only)
- **Deliverable:** `explain/` module — entity + score in, SHAP chart data + evidence subgraph + text explanation out

---

## Phase 7 — Dashboard (Day 15–18)

**Goal:** The investigator-facing surface.

- Streamlit app: ranked, filterable alert table (by score/typology)
- Click-through to interactive link-analysis graph (pyvis / streamlit-agraph)
- SHAP waterfall chart per selected alert
- **Deliverable:** `dashboard/app.py` — full pipeline output browsable end-to-end via `streamlit run`

---

## Phase 8 — Validation (Day 18–20)

**Goal:** Quantified, defensible performance numbers for the write-up/demo.

- Run full pipeline on synthetic data, compare against held-out ground truth
- Precision / recall / F1 per typology and overall
- Sensitivity analysis: vary generator parameters, check generalization (not just overfitting to one synthetic pattern)
- **Deliverable:** `tests/evaluation_report.md` (or notebook) with metrics and a couple of illustrative case walkthroughs

---

## Phase 9 — Packaging & Polish (Day 20–21+)

**Goal:** Reproducible, demo-ready, and (if relevant) presentation-ready.

- `run.sh` for one-command startup; optional Dockerfile for offline reproducibility
- README with setup instructions, architecture diagram, sample screenshots
- Trim dependencies, pin versions, confirm it runs clean in a fresh venv
- (Optional, stretch) Dockerized Neo4j swap-in to demonstrate the pluggable graph backend

---

## Suggested Build Order Priorities (if time-constrained)

If time runs short, the priority order for a compelling demo is:
1. Synthetic data + ingestion (Phase 1–2) — nothing works without this
2. Graph + one working detector (Isolation Forest) + basic scoring (Phase 3, part of 5)
3. A minimal dashboard showing ranked alerts (part of Phase 7)
4. Then layer back in: autoencoder, Louvain, motif detectors, SHAP, evidence subgraphs, validation metrics

This ensures there's always an end-to-end working demo, even if individual layers are still shallow.

---

## Stretch Goals (post-hackathon / production path)

- Swap NetworkX → Neo4j Community Edition (Docker, offline) for scale
- Human-in-the-loop feedback loop: investigator labels feed back into score-fusion weights
- Distributed graph backend (Spark/Cassandra) for full-chain-scale deployment, mirroring GraphSense's scale-out approach
