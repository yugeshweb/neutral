"""Materialize the Breast Cancer Wisconsin (Diagnostic) dataset as a canonical CSV.

Unlike the other data/ scripts, this one needs no externally-supplied raw
file and no license review: WDBC ships inside scikit-learn itself
(`sklearn.datasets.load_breast_cancer`, BSD-licensed distribution, public
UCI/Wolberg-et-al. source data) as a small bundled data file, so it's
reproducible from a bare `pip install scikit-learn` with nothing to
download and nothing to verify per FR-035.

Column names are left exactly as sklearn provides them (`"mean radius"`,
`"worst concave points"`, etc. — 30 total) and the label is written as
`"malignant"`/`"benign"`, matching `experiment.load_breast_cancer_dataset()`
exactly, so a model trained via that function's in-memory path and a model
trained from this CSV via `load_csv_dataset()` see byte-identical column
names and label strings. That parity is deliberate: see
`standardize.get_disease_schema("breast-cancer")`.
"""

import csv
from pathlib import Path

from sklearn.datasets import load_breast_cancer

DST = Path(__file__).parent / "wdbc.csv"


def main() -> None:
    source = load_breast_cancer()
    feature_names = [str(name) for name in source.feature_names]
    fieldnames = ["id", "diagnosis", *feature_names]

    with DST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, target) in enumerate(zip(source.data, source.target), start=1):
            record = {"id": f"row-{index}", "diagnosis": "malignant" if target == 0 else "benign"}
            record.update(zip(feature_names, row))
            writer.writerow(record)

    n_malignant = int((source.target == 0).sum())
    print(f"wrote {DST}: {len(source.target)} rows, {n_malignant} malignant, {30} features")


if __name__ == "__main__":
    main()
