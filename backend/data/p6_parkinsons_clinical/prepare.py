"""Adapter: derive a subject id column for the UCI Parkinson's voice dataset.

195 rows are repeated voice recordings from only 32 subjects (e.g.
"phon_R01_S01_1".."phon_R01_S01_6" are all subject S01) — `load_csv_dataset`
has no way to derive a group column from a substring of another column, so a
`subject` column is added here (adapter/feature-extraction boundary, reuse-
order rule 4) so the engine's existing GroupShuffleSplit can prevent the same
subject's recordings from crossing the train/test split (FR-021).

Source: Kaggle elnazalikarami/uci-ml-parkinsons-dataset (CC BY 4.0), itself a
republish of the UCI Machine Learning Repository "Parkinsons" dataset
(Little et al., 2007).
"""

import csv
import re
from pathlib import Path

SRC = Path(__file__).parent / "parkinsons.data"
DST = Path(__file__).parent / "parkinsons_with_subject.csv"


def main() -> None:
    with SRC.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        row["subject"] = re.sub(r"_\d+$", "", row["name"])

    fieldnames = list(rows[0].keys())
    with DST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_subjects = len({row["subject"] for row in rows})
    print(f"wrote {DST}: {len(rows)} rows, {n_subjects} unique subjects")


if __name__ == "__main__":
    main()
