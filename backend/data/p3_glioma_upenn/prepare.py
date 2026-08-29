"""Merge UPenn-GBM CaPTk radiomic features (structural mpMRI, auto-segmented tumor
regions) with the clinical MGMT-methylation label into one tabular CSV for the
qhealth_qml tabular engine.

Source: TCIA UPENN-GBM collection (CC BY 4.0, verified via NBIA series metadata
LicenseName field, 2026-08-29). Radiomic features are CaPTk's own pre-computed
per-region summary statistics — this script does no image processing itself.
"""
import csv
import glob

MODALITIES = ["FLAIR", "T1", "T1GD", "T2"]
REGIONS = ["ED", "ET", "NC"]  # edema, enhancing tumor, necrotic core


def load_radiomic_file(path: str) -> dict[str, dict[str, str]]:
    with open(path) as f:
        reader = csv.DictReader(f)
        feature_cols = [c for c in reader.fieldnames if c != "SubjectID"]
        return {row["SubjectID"]: {c: row[c] for c in feature_cols} for row in reader}


def main() -> None:
    tables = []
    subject_sets = []
    for modality in MODALITIES:
        for region in REGIONS:
            path = f"extracted/Radiomic_Features_CaPTk_automaticsegm_{modality}_{region}.csv"
            table = load_radiomic_file(path)
            tables.append((f"{modality}_{region}", table))
            subject_sets.append(set(table.keys()))

    common_subjects = set.intersection(*subject_sets)

    clinical = {}
    with open("clinical_info_v2.1.csv") as f:
        for row in csv.DictReader(f):
            clinical[row["ID"]] = row

    labeled_subjects = sorted(
        s for s in common_subjects
        if s in clinical and clinical[s]["MGMT"] in ("Methylated", "Unmethylated")
    )

    fieldnames = ["subject_id", "patient_id", "mgmt_methylated"]
    for prefix, table in tables:
        fieldnames.extend(table[labeled_subjects[0]].keys())

    with open("glioma_mgmt_features.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for subject_id in labeled_subjects:
            row = {
                "subject_id": subject_id,
                "patient_id": subject_id.split("_")[0],
                "mgmt_methylated": "1" if clinical[subject_id]["MGMT"] == "Methylated" else "0",
            }
            for prefix, table in tables:
                row.update(table[subject_id])
            writer.writerow(row)

    print(f"wrote {len(labeled_subjects)} rows, {len(fieldnames)} columns")


if __name__ == "__main__":
    main()
