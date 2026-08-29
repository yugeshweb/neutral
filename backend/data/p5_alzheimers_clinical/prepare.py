"""Adapter: derive a binary dementia target from OASIS-1 cross-sectional CDR.

`load_csv_dataset()` requires the target column to have exactly two distinct
values (reuse-order rule 4 adapter boundary — this is a label-definition
transform, not a capability the engine should grow). CDR (Clinical Dementia
Rating) has 4 levels (0, 0.5, 1, 2) and is missing for subjects never
clinically assessed. The standard OASIS-based-classification practice (used
throughout the literature building on this dataset) is: drop unassessed
subjects, binarize CDR > 0 as "dementia". CDR itself is kept as a column so
the profile can declare it a leakage_column (visible for provenance, excluded
from features) rather than silently deleting information.

Source: Kaggle jboysen/mri-and-alzheimers (CC0-1.0), itself a republish of the
OASIS-1 cross-sectional release (oasis-brains.org).
"""

import csv
from pathlib import Path

SRC = Path(__file__).parent / "oasis_cross-sectional.csv"
DST = Path(__file__).parent / "oasis_cross_sectional_labeled.csv"


def main() -> None:
    with SRC.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    labeled = []
    for row in rows:
        cdr = row["CDR"].strip()
        if not cdr:
            continue
        row = dict(row)
        row["dementia"] = "1" if float(cdr) > 0 else "0"
        labeled.append(row)

    fieldnames = list(rows[0].keys()) + ["dementia"]
    with DST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)

    n_positive = sum(1 for r in labeled if r["dementia"] == "1")
    print(f"wrote {DST}: {len(labeled)} rows (dropped {len(rows) - len(labeled)} without CDR), "
          f"{n_positive} positive ({100 * n_positive / len(labeled):.1f}%)")


if __name__ == "__main__":
    main()
