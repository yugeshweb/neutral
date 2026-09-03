"""Manual verification script for the standardizer, meant to be read while
running it. Every check either prints "PASS: ..." or raises/prints
"FAIL: ...". Nothing here is randomized - the same input files always
produce the same printed numbers, so this output IS the expected result:
save it, rerun the script later (e.g. after a git pull), and diff the two.

Covers the platform's 4 target diseases: heart disease, breast cancer,
Alzheimer's, and brain tumor.

Run from backend/:
    .venv/Scripts/python tests/manual_check/verify_standardizer.py   (Windows)
    .venv/bin/python tests/manual_check/verify_standardizer.py       (macOS/Linux)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qhealth_qml import standardize as std  # noqa: E402

HERE = Path(__file__).parent
FAILURES: list[str] = []


def check(label: str, condition: bool) -> None:
    if condition:
        print(f"PASS: {label}")
    else:
        print(f"FAIL: {label}")
        FAILURES.append(label)


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def fresh_store(name: str) -> std.SchemaStore:
    store = std.SchemaStore(HERE / f"_schema_cache_{name}")
    if store.directory.exists():
        for f in store.directory.glob("*.json"):
            f.unlink()
    return store


# ---------------------------------------------------------------------------
# 0. Registry
# ---------------------------------------------------------------------------

section("0. Disease registry")
diseases = std.list_supported_diseases()
for d in diseases:
    print(f"  {d['disease_id']:14s} status={d['status']:24s} required_fields={len(d['required_fields'])}")
check(
    "registry exposes all 7 target diseases (2 platform demos + 5 of the 6 neuro-conditions - P2 ICH has no dataset)",
    {d["disease_id"] for d in diseases}
    == {"heart-disease", "breast-cancer", "stroke", "brain-tumor", "seizure", "alzheimers", "parkinsons"},
)


# ---------------------------------------------------------------------------
# 1. Heart disease: labeled upload
# ---------------------------------------------------------------------------

section("1. Heart disease - labeled (training/benchmark) upload")
store = fresh_store("heart")
heart_csv = HERE / "heart_disease_sample.csv"
X_heart, y_heart = std.standardize(heart_csv, "heart-disease", schema_store=store)
fitted_heart = store.load("heart-disease")

print(f"X_heart.shape = {X_heart.shape}   (expected: (10, 13) - all 13 UCI fields are numeric codes)")
print(f"y_heart = {y_heart.tolist()}   (expected: [1,1,1,1,1,0,0,0,0,0])")
print(f"feature_names = {fitted_heart.feature_names}")
print(f"row 0 (patient id=1) X = {X_heart[0].tolist()}")

check("X_heart has 10 rows, 13 columns (no one-hot expansion - all fields numeric)", X_heart.shape == (10, 13))
check("y_heart has 5 positives and 5 negatives", y_heart.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
check(
    "the '?' for patient id=2's ca became NaN, not a crash",
    np.isnan(X_heart[1][fitted_heart.feature_names.index("ca")]),
)


# ---------------------------------------------------------------------------
# 2. Heart disease: predict upload reproduces the fitted layout
# ---------------------------------------------------------------------------

section("2. Heart disease - unlabeled (predict) upload, same two patients, no 'heart_disease' column")
predict_csv_text = (
    "id,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n"
    "1,63,1,4,145,233,1,2,150,0,2.3,3,0,6\n"
    "2,67,1,4,160,286,0,2,108,1,1.5,2,?,3\n"
)
X_predict, y_predict = std.standardize(predict_csv_text, "heart-disease", schema_store=store)
print(f"X_predict.shape = {X_predict.shape}   (expected: (2, 13) - same width as training)")
print(f"y_predict = {y_predict}   (expected: None)")

check("predict has no label", y_predict is None)
check("predict X has the same column width as training X", X_predict.shape[1] == X_heart.shape[1])
check(
    "patient id=1's predict-time row is BYTE-IDENTICAL to its training-time row",
    np.allclose(X_predict[0], X_heart[0], equal_nan=True),
)
check(
    "patient id=2's predict-time row (with the same '?' for ca) matches too",
    np.allclose(X_predict[1], X_heart[1], equal_nan=True),
)


# ---------------------------------------------------------------------------
# 3. Breast cancer: labeled upload, REAL WDBC data (bundled with scikit-learn)
# ---------------------------------------------------------------------------

section("3. Breast cancer - labeled upload (real WDBC biopsy data, 8 malignant + 8 benign)")
store_bc = fresh_store("breast_cancer")
bc_csv = HERE / "breast_cancer_sample.csv"
X_bc, y_bc = std.standardize(bc_csv, "breast-cancer", schema_store=store_bc)
fitted_bc = store_bc.load("breast-cancer")

print(f"X_bc.shape = {X_bc.shape}   (expected: (16, 30) - all 30 WDBC measurements, no categoricals)")
print(f"y_bc = {y_bc.tolist()}   (expected: eight 1s then eight 0s - malignant rows written first)")
print(f"row 0 (first malignant sample) mean radius = {X_bc[0][fitted_bc.feature_names.index('mean radius')]}")

check("X_bc has 16 rows, 30 columns", X_bc.shape == (16, 30))
check("y_bc is 8 malignant (1) then 8 benign (0)", y_bc.tolist() == [1] * 8 + [0] * 8)


# ---------------------------------------------------------------------------
# 4. Alzheimer's: labeled upload (unchanged from before)
# ---------------------------------------------------------------------------

section("4. Alzheimer's - labeled (training/benchmark) upload")
store_alz = fresh_store("alzheimers")
alz_csv = HERE / "alzheimers_sample.csv"
X_alz, y_alz = std.standardize(alz_csv, "alzheimers", schema_store=store_alz)
fitted_alz = store_alz.load("alzheimers")
print(f"X_alz.shape = {X_alz.shape}   (expected: (10, 9) - 7 numeric fields + M/F one-hot expanded to 2)")
print(f"y_alz = {y_alz.tolist()}   (expected: [1,1,1,1,1,0,0,0,0,0])")

check("X_alz has 10 rows, 9 columns", X_alz.shape == (10, 9))
check("y_alz has 5 positives and 5 negatives", y_alz.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# 5. Brain tumor: labeled upload (glioma MGMT-methylation proxy, synthetic
#    radiomics-shaped features - no fixed required_fields)
# ---------------------------------------------------------------------------

section("5. Brain tumor - labeled upload (synthetic radiomics-shaped features)")
store_bt = fresh_store("brain_tumor")
bt_csv = HERE / "brain_tumor_sample.csv"
X_bt, y_bt = std.standardize(bt_csv, "brain-tumor", schema_store=store_bt)
print(f"X_bt.shape = {X_bt.shape}   (expected: (10, 4) - 4 feature columns, subject_id/patient_id excluded)")
print(f"y_bt = {y_bt.tolist()}   (expected: [1,1,1,1,1,0,0,0,0,0])")

check("X_bt has 10 rows, 4 columns", X_bt.shape == (10, 4))
check("y_bt has 5 positives and 5 negatives", y_bt.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

schema_bt = std.get_disease_schema("brain-tumor")
check(
    "brain-tumor schema honestly states its narrow scope (MGMT-methylation, not general tumor detection)",
    "MGMT" in schema_bt.notes and "general tumor detection" in schema_bt.notes,
)


# ---------------------------------------------------------------------------
# 6. Error paths - each must raise the documented typed exception
# ---------------------------------------------------------------------------

section("6. Error handling")

try:
    std.standardize(heart_csv, "made-up-disease", schema_store=store)
    check("unregistered disease_id raises UnknownDiseaseError", False)
except std.UnknownDiseaseError:
    check("unregistered disease_id raises UnknownDiseaseError", True)

try:
    std.standardize("id,age,heart_disease\n1,63,1\n2,67,1\n3,52,0\n4,44,0\n", "heart-disease", schema_store=store)
    check("missing required columns raises SchemaMismatchError", False)
except std.SchemaMismatchError as exc:
    check(
        f"missing required columns raises SchemaMismatchError (missing includes 'thal': {'thal' in exc.missing_fields})",
        "thal" in exc.missing_fields,
    )

never_fitted = fresh_store("never_fitted")
try:
    std.standardize(predict_csv_text, "heart-disease", schema_store=never_fitted)
    check("predict before any training upload raises SchemaNotFittedError", False)
except std.SchemaNotFittedError:
    check("predict before any training upload raises SchemaNotFittedError", True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section("Summary")
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"All checks PASSED ({len(diseases)} diseases registered).")
