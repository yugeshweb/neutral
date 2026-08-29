# P3 Glioma Characterization (mpMRI) — Research and Reuse Record

**Feature**: `001-neurological-conditions`
**Date**: 2026-08-29
**Status**: One sub-task attempted end-to-end (MGMT promoter-methylation status from mpMRI
radiomics) with a genuine, non-DUA public dataset (TCIA UPenn-GBM, CC BY 4.0). Production-rigor
result: **negative** — neither classical nor quantum candidates exceed chance after a full
architecture sweep. Both new models are registered `availability: "not available"` as a documented,
honest negative result, not a usable coverage-closing model. The condition's core task
(tumor segmentation, grade/subtype characterization) remains BLOCKED — see §2.

## 1. Existing repository implementation inspected

No existing glioma-specific code. The registry already carries a stub model
(`glioma-characterization-mpmri`, `availability: "not available"`) so the catalog honestly reports
the coverage gap per FR-002/US-1 scenario 3 rather than hiding it.

## 2. Candidate datasets considered and why each was rejected

- **BraTS (any year, official source)** — the spec's named source for this condition. Official
  distribution is via the Synapse platform and requires creating a Synapse account and accepting
  the challenge's data-use terms before download — a per-human registration step, not something an
  automated agent can complete on the user's behalf. **Rejected** (same category of blocker as
  ADNI/PPMI: human-completed access agreement).
- **Kaggle re-uploads of BraTS** (`awsaf49/brats20-dataset-training-validation`,
  `dschettler8845/brats-2021-task1`, and similar mirrors found via search) — checked directly rather
  than assumed usable:
  - `dschettler8845/brats-2021-task1`'s page explicitly references "Synapse" (verified via direct
    fetch, 2026-08-29) — a re-upload of Synapse-gated challenge data.
  - `awsaf49/brats20-dataset-training-validation`'s Kaggle metadata self-declares a "CC0" license,
    but BraTS is a registration-gated academic challenge dataset; a third-party uploader taking data
    from a gated source does not have the standing to grant a CC0 waiver over it, regardless of what
    license tag they attach on Kaggle. Treating a self-declared tag on a re-upload as authoritative
    would be exactly the license-laundering this project's reuse-record standard (FR-034–040) exists
    to prevent — the same reasoning already applied to the P2 CT-ICH Kaggle mirrors.
  - **Both rejected** on this basis, not on file-size or format grounds.
- **TCGA-GBM / TCGA-LGG (via TCIA)** — the collection names cited in the spec no longer resolve in
  TCIA's current NBIA collection list (verified via `getCollectionValues`, 2026-08-29); they appear
  to have been retired/restructured. Searching NBIA's live collection list for glioma-relevant
  entries instead surfaced **`UPENN-GBM`** (630 patients, 3680 MR series) — pursued below.

### 2a. UPENN-GBM — pursued and completed (one sub-task)

- **Access**: TCIA's public NBIA REST API (`services.cancerimagingarchive.net/nbia-api`) serves
  `UPENN-GBM` with **no account, API key, or DUA** — confirmed by directly querying
  `getPatient`/`getSeries` anonymously. **License verified at the DICOM-record level** (not just a
  webpage claim): every series' `LicenseName` field returned by the API reads "Creative Commons
  Attribution 4.0 International License" (`LicenseURI: https://creativecommons.org/licenses/by/4.0/`).
  This is the same verification standard applied to every other condition's data source this session.
- **Why not full BraTS-style segmentation**: UPenn-GBM is single-collection GBM-only (no paired
  low-grade cohort for a grade-classification task), and building a from-scratch tumor segmentation
  model was out of scope for the tabular-engine-reuse pattern this platform uses everywhere else.
  Instead, the collection publishes its own **pre-computed CaPTk radiomic features** over CaPTk's
  automatic tumor-region segmentation (edema/enhancing-tumor/necrotic-core), for each of
  FLAIR/T1/T1GD/T2 — 12 CSVs, 144 features each, downloaded directly from
  `cancerimagingarchive.net/wp-content/uploads/radiomic_features_CaPTk.zip` (same CC BY 4.0 terms).
  A companion `UPENN-GBM_clinical_info_v2.1.csv` carries MGMT promoter methylation status
  (Methylated/Unmethylated/Indeterminate/Not Available) — a well-known, clinically meaningful,
  reasonably balanced (121 methylated / 170 unmethylated known-status cases) molecular
  characterization target used throughout the radiogenomics literature (e.g. the RSNA-MICCAI BraTS
  2021 MGMT challenge). This reframes the achievable sub-task as **molecular characterization from
  imaging-derived radiomics**, not segmentation or grade/subtype classification — a real slice of
  "glioma characterization" per the condition's `expected_output`, but not the whole task.
- **Pipeline**: `backend/data/p3_glioma_upenn/prepare.py` joins the 12 structural-MRI
  `automaticsegm` radiomic CSVs (FLAIR/T1/T1GD/T2 x ED/ET/NC) on `SubjectID`, joins the clinical
  MGMT column, drops rows without a definite Methylated/Unmethylated status → 256 labeled cases,
  1728 radiomic features, `patient_id` group column for leakage-safe splitting. This reuses the
  `qhealth_qml` tabular engine exactly as P1/P4/P5/P6 do — no new imaging/deep-learning code was
  written; CaPTk did the only image processing involved, and it is credited as reused, not
  reproduced.
- **Result — production-rigor, honestly negative**: `execution.benchmark_model()` on the full
  256-case cohort (204 train / 52 test, group-based holdout, `max_train=0`/`max_test=0` to avoid the
  engine's default 80/40 sub-sample cap) gave classical (rbf_svc) balanced accuracy **0.474** and
  QSVC **0.451** — both *below* chance. A follow-up architecture sweep (ANOVA vs PCA reduction x
  4/6/8 qubit-equivalent components, all 4 classical/quantum model families, `repeats=5`) found
  **every one of 24 configurations in the [0.44, 0.55] balanced-accuracy band** — statistically
  indistinguishable from a coin flip, with the one nominally-highest mean (PCA/8-qubit HistGB,
  0.55) inside one standard deviation of chance. This mirrors the P1 ansatz-sweep pattern: a single
  promising-looking cell (the initial 120-row subsample run showed 0.558/0.505) did not survive
  full-data, repeated, multi-configuration scrutiny.
- **Registered honestly, not exposed as usable**: `glioma-mgmt-radiomics-tabular` (quantum) and
  `glioma-mgmt-radiomics-tabular-classical` were both registered with real artifacts, real
  evaluation records, and a full reuse manifest, but `availability: "not available"` — an at-chance
  classifier is not a coverage-closing model regardless of how much benchmarking rigor went into
  showing it. The registry surfaces this as a documented, evidenced attempt (per the fixed
  `catalog()` behavior that shows "not available" models rather than hiding them), not as a live
  assessment path.
- **Plausible reasons this specific attempt came back negative** (for a future attempt, not
  pursued further here to avoid open-ended busywork on an already-answered question): (a) CaPTk's
  automatic segmentation is not independently quality-checked in this pass — upstream segmentation
  errors propagate uncorrected into every downstream radiomic feature; (b) 1728 largely-correlated
  texture/intensity features reduced to 4-8 components by ANOVA/PCA may destroy the specific
  cross-feature interactions that make MGMT prediction from radiomics work in papers that use
  end-to-end deep learning on the raw volumes instead; (c) 256 cases (204 train) is a small sample
  for a molecular-subtype task that published deep-learning approaches typically tackle with
  hundreds to thousands of cases and heavier augmentation.

## 3. Conclusion

BraTS itself and its Kaggle mirrors are blocked for the same reasons as P2's CT-ICH mirrors —
registration-gated source data does not become freely reusable by being re-uploaded elsewhere.
TCGA-GBM/TCGA-LGG under those exact names no longer resolve in TCIA. **UPenn-GBM was found, is
genuinely non-DUA, and was pursued to completion for one achievable sub-task** (MGMT-methylation
characterization from pre-computed radiomics) — the result is a real, production-rigor,
honestly-reported negative finding, not an unattempted lead. Tumor segmentation and grade/subtype
characterization — the condition's primary `expected_output` — remain out of reach without either
(a) a human completing BraTS/Synapse registration, or (b) a dedicated from-scratch imaging
segmentation effort (nnU-Net/MONAI, as sketched in the original stub's reuse manifest), which is a
different scale of engineering than the tabular-engine-reuse pattern used everywhere else in this
project and was not undertaken here.

**Unblock path for the remaining gap**: either (a) a human completes BraTS/Synapse registration and
supplies the data for a proper segmentation/grade model, or (b) a follow-up pass is explicitly
scoped and resourced for a from-scratch imaging segmentation pipeline (not a quick reuse of the
existing tabular engine).
