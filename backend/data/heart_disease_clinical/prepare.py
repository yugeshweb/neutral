"""Adapter: binarize the UCI Cleveland heart-disease severity label.

Not run automatically - no raw source file is bundled or downloaded here
(same policy as p1_stroke_clinical/p3_glioma_upenn: third-party dataset
payloads need a license check per FR-035 before anything gets committed;
see specs/001-neurological-conditions/research/*-reuse-record.md for the
review those went through). Point SRC at a locally-placed copy of the
Cleveland processed file (the classic 14-column
`age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num`
layout - UCI Heart Disease Data Set, Cleveland Clinic Foundation subset,
Detrano et al. 1989) and run this once to produce the canonical CSV
`heart_disease_clinical.json`'s profile expects.

Why binarize: the raw `num` column is 0-4 (angiographic disease severity,
0 = no vessel >50% narrowed), but `load_csv_dataset()` requires the target
column to have exactly two distinct values (FR: binary early-detection
task). `num > 0` -> `heart_disease = 1` is the standard convention used
throughout the heart-disease-classification literature for this dataset.
`num` itself is dropped from the output (kept as a `leakage_columns` entry
in the profile as a belt-and-suspenders guard, in case a caller points the
profile at the raw file directly instead of this script's output).

Known gap, stated plainly: troponin (a blood biomarker) is NOT in this
1988-era cohort - routine high-sensitivity troponin testing postdates it.
`clinical_context.gap` in the profile says so; do not backfill a troponin
column with a placeholder value.
"""

import csv
from pathlib import Path

SRC = Path(__file__).parent / "processed.cleveland.data"
DST = Path(__file__).parent / "heart_disease_uci.csv"

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def main() -> None:
    if not SRC.exists():
        raise SystemExit(
            f"{SRC} not found. Place the UCI Cleveland processed file there "
            "(comma-separated, no header, 14 columns in RAW_COLUMNS order, "
            "'?' for missing ca/thal) and rerun."
        )

    with SRC.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = [dict(zip(RAW_COLUMNS, row)) for row in reader if row]

    fieldnames = ["id"] + [c for c in RAW_COLUMNS if c != "num"] + ["heart_disease"]
    with DST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            severity = row["num"].strip()
            record = {c: row[c] for c in RAW_COLUMNS if c != "num"}
            record["heart_disease"] = "1" if severity not in ("0", "") else "0"
            record["id"] = f"row-{index}"
            writer.writerow(record)

    n_positive = sum(1 for row in rows if row["num"].strip() not in ("0", ""))
    print(f"wrote {DST}: {len(rows)} rows, {n_positive} with heart_disease=1")


if __name__ == "__main__":
    main()
