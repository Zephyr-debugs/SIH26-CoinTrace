"""
Phase 9 - record an analyst's correction on a ranked entity from the
command line (the dashboard has the same thing as buttons - see
cointrace/dashboard/app.py). Fully offline; just appends one row to a
local CSV that scripts/run_pipeline.py reads on its next run.

Usage:
    python scripts/record_feedback.py <entity_id> illicit ["optional note"]
    python scripts/record_feedback.py <entity_id> benign  ["optional note"]

<entity_id> is whatever's in the left-hand index column of
data/pipeline_output/ranked_entities.csv (also shown in the dashboard's
"Inspect an entity" dropdown).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cointrace import config
from cointrace.feedback.retrain import enough_feedback_to_retrain
from cointrace.feedback.store import latest_labels_only, load_feedback, record_feedback


def main():
    if len(sys.argv) < 3 or sys.argv[2] not in ("illicit", "benign"):
        print(__doc__)
        sys.exit(1)

    entity_id = sys.argv[1]
    is_illicit = sys.argv[2] == "illicit"
    note = sys.argv[3] if len(sys.argv) > 3 else ""

    record_feedback(entity_id, is_illicit, note)
    n = len(latest_labels_only(load_feedback()))
    print(f"Recorded: {entity_id} -> {'illicit' if is_illicit else 'benign'}")
    print(f"{n} distinct entities labeled so far.")

    if enough_feedback_to_retrain():
        print("Enough feedback on file - the fusion weights will be re-fit "
              "automatically next time you run `python scripts/run_pipeline.py`.")
    else:
        print(f"Need at least {config.MIN_FEEDBACK_FOR_RETRAIN} labeled entities, "
              "with both illicit and benign examples present, before weights "
              "get re-fit.")


if __name__ == "__main__":
    main()
