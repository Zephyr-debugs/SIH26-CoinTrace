#!/usr/bin/env bash
# One-command startup: generate data (if missing) -> run pipeline -> launch dashboard.
set -e

cd "$(dirname "$0")"

if [ ! -f "data/synthetic/transactions.csv" ]; then
    echo "No synthetic data found - generating..."
    python scripts/generate_data.py
fi

echo "Running pipeline..."
python scripts/run_pipeline.py

echo "Launching dashboard..."
streamlit run cointrace/dashboard/app.py
