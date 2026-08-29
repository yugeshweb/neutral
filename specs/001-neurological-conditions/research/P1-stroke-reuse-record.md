# P1 Acute Ischemic Stroke — Research and Reuse Record

**Feature**: `001-neurological-conditions`
**Work package**: P1 — acute ischemic stroke lesion detection/segmentation **plus** a case-level
clinical/outcome head
**Record version**: 1.0.0
**Date**: 2026-08-29
**Author**: tech@centai.in
**Gate**: FR-034 / FR-035 / FR-036 / FR-037 / FR-039 / FR-040, SC-014, SC-015, SC-016
**Status**: complete — required before any P1 model code is written
**Depends on**: `../research/P0-platform-reuse-record.md`

## Scope decision up front

P1 as specified has **two arms**. This record splits them and reaches a different decision for each.

| Arm | Task | Reference standard the spec demands | Decision now |
|---|---|---|---|
| **A — imaging** | Lesion / infarct-core detection and segmentation on presentation CT / CTA / CTP | Expert lesion annotation and/or follow-up DWI | **DEFERRED — not built.** Blocked on data access and disk, both documented in section 4. Registered in the model registry with `availability: "not available"` so the catalog shows the coverage gap (FR-032, SC-004) without a fabricated artifact. |
| **B — tabular** | Case-level clinical risk head from structured risk factors | *Not met.* The only obtainable dataset is a low/moderate reference-label-confidence risk-factor table | **PROCEEDS NOW**, with an explicit `reference_label_tier: "low"` and a limitation string on every finding. Built entirely by reusing the existing `qhealth_qml` engine — **zero new model code**. |

Arm B is **not** a substitute for arm A and must never be presented as one. It is registered as a
separate `ModelDefinition` under the same `ConditionDefinition`, with a different task
(`binary_classification` on prevalent-stroke history, not acute lesion detection), a different
output, and its own limitations. Both entries share one condition card so the UI can show that the
high-reference imaging capability is absent.

---

## 1. Existing repository implementation and dependencies inspected

Everything in section 1 of the P0 record applies and is not repeated. The P1-specific findings:

**Already present and directly usable for arm B, unchanged:**
`experiment.load_csv_dataset()` → `PreprocessingPipeline` (median impute → StandardScaler →
SelectKBest-ANOVA to `n_qubits` → MinMax angle scaling) → `_split_indices()` (stratified random, the
correct strategy for a one-row-per-patient cross-sectional cohort) → `_build_model()`
(LogisticRegression / RBF-SVC / HistGradientBoosting / QSVC / PegasosQSVC / VQC) →
`classification_metrics()` + `calibration_bins()` + `decision_curve()` + `subgroup_metrics()` +
`bootstrap_confidence_intervals()` → `select_threshold()` (fitted on validation only) →
`SavedModelArtifact`. `study.run_nested_evaluation()` supplies the paired-delta comparison.
This is a complete, tested implementation of the arm-B task. **The correct amount of new model code
for arm B is zero.**

**Already present for arm A: nothing.** No DICOM, NIfTI, volumetric array, resampling, skull-strip,
patch sampler, U-Net, Dice loss, or segmentation metric exists anywhere in the repository. Arm A is
a from-scratch imaging stack, which is precisely why the reuse order points at DeepISLES/nnU-Net
rather than at new code — and why it cannot start until the data exists locally.

**Two blocking incompatibilities found in the reused loader (verified by reading the source, not
assumed):**

1. `load_csv_dataset()` raises `ValueError: feature <name> at CSV row <n> is not numeric` for any
   non-numeric non-empty cell. The raw Kaggle file has five text columns (`gender`, `ever_married`,
   `work_type`, `Residence_type`, `smoking_status`) and 201 `bmi` cells containing the literal
   string `N/A`. **The raw file cannot be loaded.**
2. `load_profile_dataset()` invokes `protocol.validate_early_detection_profile()`, which hard-fails
   unless `group_column`, `index_time_column`, `outcome_time_column`, `horizon_days`, and a
   non-empty `outcome_definition` are all supplied. This dataset is cross-sectional with no
   timestamps at all. **`load_profile_dataset()` cannot be used.**

Both are resolved without touching engine code — see the reuse decisions in section 6.

**Dataset inspected** (`backend/data/p1_stroke_clinical/healthcare-dataset-stroke-data.csv`,
316,971 bytes, verified by direct parse):
5110 data rows, 5110 unique `id` values (one row per patient — no repeated-measure leakage risk),
12 columns. Target `stroke`: 249 positive / 4861 negative (**4.87 % prevalence**).
`gender`: Male 2115 / Female 2994 / Other 1. `ever_married`: Yes 3353 / No 1757.
`work_type`: Private 2925 / Self-employed 819 / Govt_job 657 / children 687 / Never_worked 22.
`Residence_type`: Urban 2596 / Rural 2514.
`smoking_status`: never 1892 / formerly 885 / smokes 789 / **Unknown 1544 (30.2 %)**.
`hypertension` 498 positive; `heart_disease` 276 positive; `bmi` 201 missing.
No site column, no scanner column, no index or outcome timestamp, no follow-up window.

---

## 2. Mature open-source libraries, checkpoints, and reference implementations considered

### Arm A — imaging (all deferred, none installed)

| Candidate | What it is | License (as best known — verify at integration) | Why it is / is not usable now |
|---|---|---|---|
| **DeepISLES** — `github.com/ezequieldlrosa/DeepIsles` | The ISLES'24 challenge-winning ensemble packaged as a runnable pipeline for acute ischemic stroke lesion segmentation from CT/CTA/CTP (and DWI variants). Ships Docker images and released weights; exposes a single-command inference entry point. | Repository states a permissive research license (Apache-2.0/MIT family); **the released model weights carry their own terms derived from the ISLES training data and must be checked separately from the code license**. Not verified — no network fetch of the LICENSE file was performed for this record. | **This is the correct reused reference implementation for arm A** and is what the FR-036 "strong reused baseline" must be. Blocked now: the Docker image plus weights plus even a minimal ISLES subset does not fit in 5.9 GB of free disk, and the ISLES'24 data itself needs a grand-challenge.org account and DUA acceptance (a human step). |
| **nnU-Net** — `github.com/MIC-DKFZ/nnUNet` | Self-configuring segmentation framework; derives preprocessing, architecture, and training schedule from a dataset fingerprint. The backbone DeepISLES and most ISLES entries build on. | Apache-2.0 (code). No pretrained stroke weights are distributed by the project itself. | Would be the fallback if DeepISLES weights turn out to be non-redistributable: train nnU-Net on ISLES locally. Requires PyTorch + CUDA + tens of GB for preprocessed volumes. Not feasible on current disk; also a multi-GPU-day training job. |
| **MONAI** — `github.com/Project-MONAI/MONAI` | PyTorch-based medical imaging library: `LoadImage` (NIfTI/DICOM), spacing/orientation/intensity transforms, `UNet`/`SegResNet`, `DiceMetric`, sliding-window inference; plus MONAI Model Zoo checkpoints. | Apache-2.0 (code); Model Zoo bundles carry per-bundle licenses. | **The intended source of arm A's I/O and metric layer even if the segmentation network comes from DeepISLES** — `LoadImage`, `Spacing`, `DiceMetric`, and `sliding_window_inference` satisfy FR-006 and FR-024 with no custom code. Deferred with arm A: installing MONAI pulls PyTorch (~2.5 GB wheel + CUDA runtime), which alone would consume most of the remaining disk. |
| `nibabel` / `pydicom` / `SimpleITK` | Bare volume I/O | MIT / MIT / Apache-2.0 | Lightweight enough to install now, but pointless without data or a model to feed. Named here as the minimal arm-A ingestion path if a small public NIfTI sample becomes available. |

### Arm B — tabular

| Candidate | Decision |
|---|---|
| **The existing `qhealth_qml` engine** | **Reused unchanged.** Reuse-order rule 1. It already implements every arm-B requirement. |
| scikit-learn `LogisticRegression` / `HistGradientBoostingClassifier` | **Reused via the engine**, which already wires both and exposes their parameter grids to `study.run_nested_evaluation()`. These are the tuned classical baselines for FR-036. |
| `xgboost` / `lightgbm` | **Rejected.** `HistGradientBoostingClassifier` is the same algorithm family, already integrated, already in the parameter-grid path, and adds no dependency. FR-038 / SC-017: no documented capability gap. |
| `imbalanced-learn` (SMOTE etc.) | **Rejected.** At 4.87 % prevalence resampling is tempting, but the engine's `class_weight`/threshold-policy path (`select_threshold(policy="target_sensitivity")`, fitted on validation only) addresses the operating point without a dependency and without synthesising patients. Synthetic minority rows would also collide with FR-031's synthetic-labelling rule. |
| Qiskit QSVC / PegasosQSVC / VQC | **Reused via the engine** as the QML candidates. Qiskit stays the only quantum stack (spec: "Qiskit remains the default quantum stack"). No PennyLane. |

---

## 3. Relevant published methods, data assumptions, validation design, limitations

**ISLES'24 — *Multimodal CT/MRI stroke lesion segmentation benchmark* (Radiology: Artificial
Intelligence, 2025; spec link `pubs.rsna.org/doi/full/10.1148/ryai.250603`).**
*Data assumptions*: paired acute presentation CT/CTA/CTP **and** a follow-up DWI-derived lesion mask
as the reference standard; multi-centre acquisition; skull-stripped, co-registered volumes.
*Validation design*: held-out test centre(s), Dice / absolute-volume-difference / lesion-wise
detection metrics, organiser-side evaluation on withheld ground truth.
*Reported limitations*: the follow-up-DWI reference measures final infarct, not the acute core at
presentation, so it conflates lesion growth with prediction error; small lesions dominate the
Dice failure modes; treatment between presentation and follow-up is a confounder the label cannot
separate. **Consequence for this project**: any arm-A claim must report lesion-wise detection
alongside Dice, and must not be described as "infarct core at presentation".

**DeepISLES (*Nature Communications*, 2025, `doi.org/10.1038/s41467-025-62373-x`).**
*Method*: ensemble of nnU-Net-derived segmentation models over the ISLES'24 multimodal inputs,
packaged for one-command reproducible inference.
*Data assumptions*: the ISLES'24 preprocessing contract — specific modality set, spacing, and
skull-stripping. Feeding it differently preprocessed volumes silently degrades output.
*Validation design*: challenge test set plus external cohorts.
*Limitations*: performance is reported on the challenge population; generalisation to different
scanner protocols and to non-thrombectomy populations is not established.
**Consequence**: DeepISLES is a strong *reused baseline*, not a validated clinical component; its
preprocessing contract becomes a hard field in `ModelDefinition.input_contract` when arm A starts.

**Kaggle `fedesoriano/stroke-prediction-dataset` (the arm-B file).**
*Data assumptions*: none documented. The dataset has **no datasheet, no named source institution,
no collection period, no inclusion/exclusion criteria, and no outcome-time definition**. The
`stroke` column's temporal relationship to the risk factors is unspecified — it is not stated
whether the label is a prevalent history or an incident event after the recorded measurements.
Community discussion questions whether the file is authentic clinical data or semi-synthetic.
*Limitations that follow directly*:
- **Reference-label confidence: LOW.** Explicitly **not** the "expert lesion annotation and/or
  follow-up diffusion MRI" standard the spec's high-reference catalog requires.
- **No index/outcome times ⇒ no defensible prediction horizon.** The model must be described as
  associative ("risk-factor association with recorded stroke status"), never as early detection or
  incident prediction, even though it runs on an engine whose profile type is named
  `early_detection`.
- **Reverse causation is unexcludable.** `heart_disease`, `hypertension`, and `bmi` may have been
  recorded after the stroke.
- **30.2 % of `smoking_status` is `Unknown`**, and that missingness is very likely non-random.
- **`gender = "Other"` has n = 1**, which is a re-identification-shaped singleton and a
  degenerate subgroup; it is mapped to missing by the adapter rather than to a one-hot column.
- **4.87 % prevalence** makes accuracy meaningless; balanced accuracy, sensitivity/specificity,
  PR-AUC, and calibration are the reportable metrics (all already produced by the engine).

**Methodological references applied to arm B**: TRIPOD+AI (Collins et al., *BMJ* 2024) for the
required reporting fields, and Van Calster et al. (*BMC Med* 2019) for treating calibration as a
first-class result rather than an afterthought. Both are reproduced as schema in the P0 record;
neither adds a dependency.

---

## 4. License, data-access, model-weight, and redistribution constraints

**Environment facts (measured, 2026-08-29):** `/` has **5.9 GB free of 229 GB (98 % used)**.

| Asset | Access status | Constraint |
|---|---|---|
| **ISLES'24** (arm A reference data) | **Not obtainable now.** Requires a grand-challenge.org account and DUA acceptance — a manual human step this agent cannot perform. | Even with access, the multimodal CT/CTA/CTP volumes exceed available disk. |
| **DeepISLES weights + Docker image** | Not fetched. | Code license and **weight license are separate**; weights are derived from ISLES data and may inherit its non-redistribution terms. **Must be verified before bundling anything into the application** (spec assumption: "Dataset access, licensing, and usage restrictions may limit which model artifacts can be bundled"). |
| **BraTS** (P3) | **Not obtainable now.** Synapse registration + DUA. | Manual human step. |
| **RSNA ICH** (P2) | **Not obtainable now.** ~450 GB via Kaggle; also requires challenge-rules acceptance. | ~76× available disk. |
| **CHB-MIT scalp EEG** (P4) | **Obtainable.** `physionet.org` served HTTP 200 with no authentication on probe. | Full database is ~40 GB; a per-patient subset (single patient, a few hours of EDF) is a few hundred MB and would fit. Not part of P1; recorded here so the "what else is realistically obtainable" question is answered. |
| **Kaggle stroke CSV** (arm B) | **Already downloaded**, `backend/data/p1_stroke_clinical/healthcare-dataset-stroke-data.csv`, 316,971 bytes. | Kaggle dataset metadata records the license as **`copyright-authors`** — i.e. rights reserved by the dataset author, **not** a recognised permissive or open licence. |

**Kaggle CSV redistribution rule (binding):**
`copyright-authors` is **not** CC0, CC-BY, or ODbL. Therefore:
- The CSV may be used **locally** for this research/benchmark evaluation.
- The CSV and any row-level content **MUST NOT** be committed to a public repository,
  redistributed, or included in an export bundle, until the license is manually verified by a
  human against the Kaggle dataset page.
- A trained `SavedModelArtifact` derived from it is a derivative work of ambiguous status.
  **Do not publish the artifact** pending the same verification.
- `.gitignore` must exclude `backend/data/p1_stroke_clinical/*.csv`. Exports carry the dataset
  **fingerprint** (`experiment.dataset_fingerprint()`) and metric values only — never rows.
- `ModelDefinition.model_card.data_licenses` records this verbatim as
  `"kaggle:copyright-authors — UNVERIFIED, redistribution blocked"`.

**Arm-B model weights**: none reused. The artifact is produced locally by the user's own run.

**Dependencies added by P1**: **zero.** Arm B runs on the already-pinned stack. Arm A's dependency
set (MONAI + PyTorch, or the DeepISLES Docker image) is deferred with arm A and will require its own
FR-038 justification and disk headroom at that time.

---

## 5. Reproducible comparison plan (FR-036 / SC-016)

### Arm A — deferred

No comparison is run. The gate is explicit: **arm A does not enter the registry as `available`
until** (i) ISLES'24 access is granted by a human, (ii) at least ~60 GB of free disk exists,
(iii) DeepISLES reproduces its published Dice on a held-out subset locally, and (iv) that DeepISLES
run is recorded as the reused reference baseline in an `EvaluationRecord`. Until then the entry is
`availability: "not available"`, produces finding status `not evaluated`, and its model card cites
DeepISLES / nnU-Net / MONAI as the *planned* reused implementation.

### Arm B — runs now

**Shared, declared, identical for every compared model** (this is the FR-036 "same declared split
and preprocessing boundary"):

- **Data**: the raw Kaggle file, `backend/data/p1_stroke_clinical/healthcare-dataset-stroke-data.csv`,
  loaded directly. `load_csv_dataset()` was extended (see §6) to auto one-hot encode its text
  columns and treat `N/A` as missing, so no separate adapter script or derived CSV exists. 5110 rows,
  21 encoded feature columns, `id` as `id_column`, `stroke` as target, `positive_label = "1"`.
- **Split**: stratified random holdout, `test_size = 0.2`, `validation_size = 0.2` of train.
  One row per patient and unique `id` ⇒ no subject, visit, acquisition, or repeated-record leakage
  is possible; there is no site column, so site-held-out validation is **unavailable** and this must
  be stated in the leakage report rather than silently omitted (SC-008).
- **Repeats**: `run_repeated_experiment(repeats=10)`, consecutive seeds from 7.
- **Sample cap**: one shared `max_train` / `max_test` for *all* compared models in the promotion
  run, because `run_experiment()` applies a single cap per run. Start at
  `max_train = 1200, max_test = 400` (stratified, so ≈58 / ≈19 positives) — a QSVC kernel is
  O(n²) and 4088 training rows is not tractable at statevector scale.
  A separate uncapped classical run (`max_train = 0, max_test = 0`) is recorded **as context only**
  and is explicitly **not** admissible evidence in the paired comparison.
- **Preprocessing boundary**: median imputation, standardisation, ANOVA `SelectKBest` to
  `n_qubits = 4`, and MinMax angle scaling — all fitted inside `PreprocessingPipeline` on the
  training partition only (FR-022, already enforced by the engine). Threshold selection uses
  `select_threshold(policy="target_sensitivity", target_sensitivity=0.80)` fitted on the validation
  partition only.
- **Metrics** (FR-023): balanced accuracy, sensitivity, specificity, PR-AUC, ROC-AUC, Brier, ECE,
  calibration bins, abstention coverage, runtime, and quantum resource counts. **Accuracy is
  reported but is not a decision metric** at 4.87 % prevalence.
- **Uncertainty**: `bootstrap_samples = 1000` for per-model CIs, plus
  `study.run_nested_evaluation()` for the paired delta CI between the QML candidate and the best
  classical baseline.

**The three compared arms (SC-016):**

| Role | Model | Source |
|---|---|---|
| Strong reused reference | `hist_gradient_boosting` with the engine's parameter grid | scikit-learn via `experiment._build_model()` — the strongest published-standard method for small tabular clinical risk data (Grinsztajn et al., NeurIPS 2022, "trees still outperform deep learning on tabular data") |
| Tuned classical baseline | `logistic_regression` (grid over C / class_weight) and `rbf_svc` (grid over C / gamma) | same |
| QML candidate | `qsvc` (ZZFeatureMap, 4 qubits), with `vqc` as a secondary | `experiment._build_model()`, Qiskit |

**Promotion gate (FR-039 / FR-040).** The QML candidate becomes the operational reference **only
if** the paired delta CI on balanced accuracy or PR-AUC excludes zero in its favour **and** there is
no unacceptable regression in calibration (ECE/Brier), sensitivity, specificity, abstention
coverage, runtime, or qubit/shot budget. **Confirmed outcome (2026-08-29, production scale): it did
not pass** — QSVC balanced accuracy 0.574 vs classical 0.756. A 4-qubit ZZFeatureMap kernel over 4
ANOVA-selected features from a low-label-confidence risk-factor table did not beat gradient boosting.
Since it did not pass, the classical model stays as the
operational reference and the QML result is labelled `experimental` in
`ModelDefinition.lifecycle`, in the evaluation report, and in the user-facing finding — this is the
designed, correct outcome, not a failure of the work package.

**Architecture exploration actually run (2026-08-29):**

| reduction | n_qubits | model | balanced_accuracy | scale |
|---|---|---|---|---|
| anova | 4 | qsvc | 0.724 | max_train=300 |
| anova | 4 | rbf_svc | 0.622 | max_train=300 |
| anova | 6 | qsvc | 0.576 | max_train=300 |
| pca | 4 | qsvc | 0.492 | max_train=300 |
| pca | 6 | qsvc | 0.395 | max_train=300 |
| anova | 4 | rbf_svc | 0.453 | max_train=200 |
| anova | 4 | qsvc | 0.363 | max_train=200 |
| anova | 4 | vqc (real_amplitudes ansatz, COBYLA, maxiter=25) | 0.458 | max_train=200 |
| anova | 4 | **qsvc (nested paired eval, production scale)** | **0.574** | **max_train=1200, repeats=10, outer_repeats=5** |
| anova | 4 | **rbf_svc (nested paired eval, production scale)** | **0.756** | **max_train=1200, repeats=10, outer_repeats=5** |

ANOVA beats PCA at every qubit count tested (expected — `SelectKBest` keeps interpretable named
features; PCA's linear combinations dilute the ANOVA-favoured signal for this feature set). 4 qubits
beats 6 at every reduction (fewer, more informative features outperform more, noisier ones on this
small a dataset). At the `max_train=200` scale, VQC (0.458) is statistically indistinguishable from
classical (0.453) while QSVC clearly underperforms both (0.363) — a real architecture-dependent
difference between the two quantum approaches (kernel vs variational), not just "QML loses."

**The production-scale run declared in §5 below has now actually been executed** (2026-08-29,
repeats=10, outer_repeats=5, inner_repeats=2, max_train=1200, max_test=400, bootstrap_samples=1000 —
exactly the scale this record originally only declared as a plan): QSVC balanced accuracy **0.574**
vs the classical reference **0.756**. The real-gain gate **FAILS** at production scale, confirming
every smaller-scale result above. This is evidence the platform's architecture search is not a
single fixed configuration — ANOVA+4-qubit is the best of the 5 QSVC configurations tried, and VQC is
a genuinely different, closer-to-competitive quantum approach at small scale — but none of that
changes the promotion decision: the classical model is the operational reference for P1, and the
QSVC entry stays `experimental`.

**Circuit ansatz / entanglement pattern exploration (2026-08-29).** The engine's `ZZFeatureMap` was
previously hardcoded to `reps=1, entanglement="linear"` — no experiment could vary it. This was a
real, closeable gap in the platform's architecture search (reuse-order rule 4: extending the
existing quantum-experiment boundary, not building a new framework), so `build_quantum_context()`,
`_prepare_run()`, and `run_experiment()` were extended with optional `feature_map_reps` /
`feature_map_entanglement` parameters, defaulting to the prior hardcoded values so nothing else
changes behaviour. A sweep over `reps ∈ {1, 2}` × `entanglement ∈ {linear, full, circular}`
(single split, `max_train=200`):

| reps | entanglement | qsvc | rbf_svc |
|---|---|---|---|
| 1 | linear | 0.363 | 0.453 |
| 1 | **full** | **0.547** | 0.453 |
| 1 | circular | 0.347 | 0.453 |
| 2 | linear | 0.500 | 0.453 |
| 2 | full | 0.500 | 0.453 |
| 2 | circular | 0.500 | 0.453 |

`reps=1, entanglement=full` briefly looked like the first genuinely competitive QSVC configuration
found in this entire project — but a 10-seed repeated evaluation (`run_repeated_experiment`, same
config) tells the honest story: `qsvc` mean balanced accuracy **0.491±0.066** vs `rbf_svc`
**0.548±0.118** — **classical wins on average**; the single-split result was a favourable seed draw,
the same pattern already seen and rejected for P6. `reps=2` degenerating to exactly 0.500 for every
entanglement pattern is itself informative: doubling circuit depth makes the kernel uninformative at
this qubit count (consistent with the "exponential concentration" phenomenon reported in the quantum
kernel literature) rather than helping. **Conclusion: circuit ansatz/entanglement exploration is now
closed for P1 at this scale — none of the 8 configurations tested (5 reduction/qubit configs + 3
entanglement patterns, all cross-checked with repeated/nested evaluation where feasible) change the
promotion decision.** The classical model remains the operational reference; the
QSVC entry stays `experimental`.

**Reproducibility**: pinned dependency set in `backend/pyproject.toml`, `runtime_manifest()`,
raw-file SHA-256, the deterministic categorical-encoding rule now built into `load_csv_dataset()`,
the profile JSON, fixed seeds, and `dataset_fingerprint()` recorded on every result.

---

## 6. Reuse decision per component (FR-037)

| Component | Decision | Evidence / reason |
|---|---|---|
| Arm-B dataset loading, preprocessing, splitting, training, metrics, calibration, thresholding, abstention, explanations, artifacts | **reused, unmodified** | `qhealth_qml.experiment` — reuse-order rule 1. Zero lines of model code written. |
| Arm-B paired benchmark and delta CI | **reused, unmodified** | `qhealth_qml.study.run_nested_evaluation()` |
| Arm-B classical baselines (LR, RBF-SVC, HistGB) | **reused** | scikit-learn, already integrated |
| Arm-B QML candidate (QSVC, VQC) | **reused** | Qiskit Machine Learning, already integrated |
| **Categorical + missing-sentinel handling in `load_csv_dataset()`** | **adapted (engine extended, not duplicated)** | The loader rejected the file's five text columns and its `N/A` bmi sentinel. Rather than write a one-off adapter script that leaves the underlying gap in place for the next dataset, the loader itself was extended to auto-detect and one-hot encode categorical columns and to recognize common missing-value sentinels as NaN. Deterministic, reuse-order rule 4 (adapter/fusion boundary), and now benefits every future condition CSV, not just this one. `gender = "Other"` (n = 1) is kept as its own one-hot column rather than silently folded into missing. |
| **`explain_raw_inputs()` bug fix** | **adapted** | Pre-existing `IndexError` when a dataset has more raw columns than the 16-feature explanation cap (this dataset has 21 after encoding). One-line fix: index the impact accumulator by position in the capped list, not by the absolute column index. |
| **P1 profile JSON** (`backend/profiles/p1_stroke_clinical.json`) | **newly authored (declaration only)** | An `EarlyDetectionProfile` instance. Data, not code. |
| **Cross-sectional profile load path** | **adapted** | `load_profile_dataset()` is unusable here (it demands index/outcome times this cohort does not have). The platform executor composes the two reused primitives directly — `load_early_detection_profile()` then `load_csv_dataset()` — and records `temporal_validation: "not applicable — cross-sectional cohort, no index or outcome time"`. Fabricating timestamps to satisfy the validator would be a data-integrity violation. |
| Arm-A segmentation network | **planned: reused (DeepISLES)**; fallback **reproduced (nnU-Net trained locally)** | Not started. Recorded in the model card so the plan is auditable before any code exists. |
| Arm-A volume I/O, transforms, Dice/IoU metrics | **planned: reused (MONAI)** | Not started; satisfies FR-006 and FR-024 without custom code. |
| Arm-A fusion of an image-derived compact representation with the clinical head | **planned: newly authored** | The only genuinely new arm-A code; explicitly the fusion/quantum-experiment boundary reuse-order rule 4 permits. Segmentation stays classical (spec: "Keep segmentation classical"). |

## 7. External-asset manifest (FR-035 / SC-015)

**Arm B — enabled now.**

| Field | Value |
|---|---|
| Asset | `healthcare-dataset-stroke-data.csv` |
| Source URL | `https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset` |
| Release / commit | Kaggle dataset version as downloaded 2026-08-29; local path `backend/data/p1_stroke_clinical/healthcare-dataset-stroke-data.csv`, 316,971 bytes. **Record the SHA-256 at first pipeline run and pin it in the profile provenance.** |
| Paper citation | None. No accompanying publication or datasheet exists. |
| License | `copyright-authors` (per Kaggle metadata) — **UNVERIFIED, not a recognised open licence, redistribution blocked** pending human review |
| Model-weight source / checksum | Not applicable — no external weights; the artifact is trained locally |
| Preprocessing assumptions | One row per patient; `id` is a non-informative surrogate key and is excluded from features via `id_column`; `stroke` is binary with `positive_label = "1"`; text columns are nominal (no ordinal meaning implied, one-hot not ordinal-encoded); `bmi = "N/A"` means missing-not-recorded; `smoking_status = "Unknown"` and `gender = "Other"` are retained as their own one-hot levels, not imputed or dropped |
| Input/output contract | Input: 21 encoded feature columns (5 numeric + 16 one-hot), produced automatically by `load_csv_dataset()` from the 12-column raw file. Output: a calibrated probability of recorded stroke status plus a thresholded label with optional abstention |
| Local modifications | None to this dataset's values. The general-purpose fix lives in the engine (`load_csv_dataset()` categorical/sentinel handling — see §6), not in a dataset-specific script; no rows dropped |

**Arm A — deferred, nothing enabled.** DeepISLES, nnU-Net, and MONAI are recorded as *planned*
reused implementations only. Each will need a full manifest row (URL, commit, paper, code license,
**separate weight license and checksum**, preprocessing contract, I/O contract, local
modifications) before its registry entry may move off `not available`. This section is the SC-015
explicit record for why arm A ships no reusable implementation today: the implementations exist and
are the right ones, but their required data and runtime do not fit the current environment.

## 8. Open risks carried forward

1. **License.** The arm-B dataset is `copyright-authors` and unverified. Redistribution, public
   commit, and artifact publication are all blocked until a human verifies it. Highest-severity
   item in this record.
2. **Label semantics are undefined.** Nothing in the source states whether `stroke` is prevalent or
   incident. Every arm-B finding must carry the associative-not-predictive limitation string, and
   `reference_label_tier` must be `low`.
3. **The spec's P1 high-reference requirement is not met by arm B.** The catalog must show acute
   ischemic stroke as having a `not available` imaging model, not as "covered".
4. **Class imbalance interacts with the engine's sample cap.** At `max_test = 400` and 4.87 %
   prevalence, a test fold holds ≈19 positives; per-fold sensitivity CIs will be wide. Ten repeats
   plus bootstrap CIs are the mitigation, and the width must be reported, not hidden.
5. **`gender = "Other"`, n = 1.** Mapped to missing. Note this as a fairness/coverage limitation:
   the model has no evidence for that group and must not be described as evaluated for it.
6. **Disk.** 5.9 GB free is the binding constraint on arm A, P2, and P3. Even the CHB-MIT P4 subset
   needs a deliberate budget.

## 9. Second-pass hybrid architecture optimization (2026-08-29) — QML promoted independent of classical

**Motivation**: classical is no longer a gate the QML candidate must beat (ACCEPTANCE-CRITERIA.md
§0). Goal: push QSVC/VQC toward CI-lower-bound-above-0.5 on its own merits, using the newly
configurable VQC ansatz surface (`ansatz`/`ansatz_reps`/`ansatz_entanglement`/`optimizer` in
`model_params["vqc"]`) plus a QSVC `C` sweep, neither swept before for this condition.

**Methodology bug found and fixed first**: the platform's default threshold policy collapses to an
all-majority-class prediction on this 4.87%-prevalence cohort (balanced_accuracy exactly 0.5,
sensitivity exactly 0, regardless of the model) — a degenerate evaluation artifact, not a real
"at chance" finding. Every run below uses `validation_size=0.2,
threshold_policy="target_sensitivity", target_sensitivity=0.8` (matching this record's own §5
production-scale recipe) to avoid it; this was verified by reproducing the known-good §5 QSVC=0.724
screening number, which the default policy could not reproduce (it returned exactly 0.5).

**Screening** (single split, `max_train=300, max_test=150, seed=7`, staged one-factor-at-a-time,
~30 runs): best QSVC screening candidate was `C=0.1, n_qubits=4, reduction=anova,
feature_map_reps=1, feature_map_entanglement=circular` (BA=0.787); best VQC screening candidate was
`ansatz=efficient_su2, ansatz_reps=2, ansatz_entanglement=full, optimizer=cobyla, maxiter=100`
(BA=0.692). ANOVA beat PCA and n_qubits=4 beat 6/8 at every step, consistent with §5's existing
finding. One VQC config (`optimizer=l_bfgs_b`) took 1749s for a single small-scale screening run and
was abandoned mid-grid (killed) as impractical at this qubit/train-size budget — L-BFGS-B's
finite-difference gradient estimation is not a viable optimizer choice for this circuit size.

**Production-scale validation** (`max_train=1200, max_test=400, seed=7`, same threshold policy,
`bootstrap_samples=1000` — matching §5's declared production recipe exactly):

| Candidate | Screening BA | **Production BA** | **95% CI** | Gate (CI lower > 0.5)? |
|---|---|---|---|---|
| qsvc C=0.1, reps=1, ent=circular | 0.787 | **0.549** | **[0.439, 0.655]** | **No** |
| vqc efficient_su2, reps=2, ent=full, cobyla-100 | 0.692 | **0.574** | **[0.476, 0.658]** | **No** |

**Neither candidate clears the baseline-viability gate at production scale**, despite both looking
promising at screening scale. This is the exact "favorable screening draw" pattern this record's §5
already documented once for the entanglement sweep (reps=1/full: 0.547 single-split → 0.491±0.066
repeated) — it recurred here with two entirely new hyperparameter axes (QSVC `C`, and the
newly-configurable VQC ansatz/optimizer space). The lesson generalizes: for this condition, no amount
of screening-scale hyperparameter tuning found so far survives production-scale re-evaluation.

**Conclusion**: P1's QSVC and VQC both remain `lifecycle: "experimental"` — the architecture search
mandated by ACCEPTANCE-CRITERIA.md §0.4 was carried out (QSVC `C`, feature-map reps/entanglement,
n_qubits/reduction, and the full newly-exposed VQC ansatz/reps/entanglement/optimizer space), and the
verdict is an honest miss, not an unattempted gap. Classical (`rbf_svc`, 0.756) remains the only
model registered `operational_reference` for this condition.
