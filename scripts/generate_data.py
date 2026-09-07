"""
Generates the synthetic, typology-injected dataset (Phase 1).

Usage:
    python scripts/generate_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cointrace.synth.generator import generate_and_save

if __name__ == "__main__":
    generate_and_save()
