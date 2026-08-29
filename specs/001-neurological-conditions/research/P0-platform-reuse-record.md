# P0 Platform — Research and Reuse Record

**Feature**: `001-neurological-conditions`
**Work package**: P0 — model registry, canonical case bundle, routing, evaluation, provenance, safety states
**Record version**: 1.0.0
**Date**: 2026-08-29
**Author**: tech@centai.in
**Gate**: FR-034 / FR-035 / FR-036 / FR-037 / FR-038, SC-014, SC-015
**Status**: complete — required before any P0 platform code is written

P0 contains **no disease model**. Its deliverable is the shared substrate every later condition
plugs into. The reuse question for P0 is therefore not "which segmentation network" but "how much
of the existing `qhealth_qml` engine can carry the registry/routing/evaluation contracts without a
rewrite".

---

## 1. Existing repository implementation and dependencies inspected

Package `quantum-health` 0.1.0, `backend/src/qhealth_qml/`, Python >= 3.10.
Pinned dependencies (`backend/pyproject.toml`): `numpy>=2.0,<3`, `scikit-learn>=1.4,<2`,
`qiskit~=2.5.1`, `qiskit-aer~=0.17.1`, `qiskit-machine-learning~=0.9.1`, `qiskit-algorithms~=0.4.0`;
optional extra `hardware = ["qiskit-ibm-runtime~=0.47.0"]`. No pandas, no torch, no imaging or
signal I/O dependency exists.

| File | What it already provides | Relevance to P0 |
|---|---|---|
| `protocol.py` (180 lines) | `EarlyDetectionProfile` frozen dataclass (18 fields: name, dataset_path, target_column, positive_label, group/index_time/outcome_time/site/id columns, horizon_days, outcome_definition, leakage_columns, subgroup_columns, modality, reduction, task_type); `load_early_detection_profile()`; `resolve_profile_dataset()`; `validate_early_detection_profile()` (temporal ordering, group separation, leakage-column, subgroup-presence checks) | This **is** the repo's declarative dataset contract. `ModelDefinition` must reference a profile rather than restate dataset fields. FR-021/FR-022 leakage semantics already live here. |
| `experiment.py` (2258 lines, `SCHEMA_VERSION = 4`) | `load_csv_dataset()`, `load_profile_dataset()`, `LoadedDataset`, `PreprocessingPipeline` (median impute → StandardScaler → SelectKBest-ANOVA or PCA to `n_qubits` → MinMax angle scaling), `_split_indices()` (group+chronological / group-only / chronological / site-holdout / stratified random), `_cap_indices()` (stratified cap; `cap <= 0` = no cap), classical models (LogisticRegression, RBF-SVC, HistGradientBoosting), quantum models (QSVC, PegasosQSVC, VQC/ZZFeatureMap+RealAmplitudes), `classification_metrics()`, `decision_curve()`, `calibration_bins()`, `subgroup_metrics()`, `select_threshold()` (validation-only), `bootstrap_confidence_intervals()`, `input_sensitivity()` / `explain_raw_inputs()`, `SavedModelArtifact` + `save_model_artifact()` / `load_model_artifact()` / `predict_with_model_artifact()`, `model_artifact_manifest()`, `run_experiment()`, `run_repeated_experiment()`, `dataset_fingerprint()`, `runtime_manifest()` | Covers **FR-013, FR-014, FR-019, FR-021, FR-022, FR-023, FR-028** for tabular tasks already. P0 must wrap, not re-implement. |
| `study.py` (20 KB) | `run_nested_evaluation()` (nested repeated holdout, per-model parameter grids, paired delta CIs, site-holdout), `run_resource_sweep()` | Covers the FR-036 "same split, paired comparison, confidence interval" machinery. |
| `cli.py`, `study_cli.py` | Full flag surface over the above (`--profile`, `--group-column`, `--site-column`, `--models`, `--backend`, `--n-qubits`, `--shots`, `--threshold-policy`, `--abstain-margin`, `--calibrate`, `--save-model`, `--predict-model`, …) | Reference for the argument set the platform executor must pass through. |
| `dashboard.py` (17 KB) | `SimpleHTTPRequestHandler` subclass with `_send_json`/`_request_json`, a `train_lock`, CORS headers, and routes `POST /api/train`, `GET /api/result`, `GET /api/model`, `GET /api/samples`, `POST /api/predict` | **Reuse the server**. P0 adds routes to this handler; it does not introduce Flask/FastAPI (FR-038, SC-017). |
| `backend/tests/test_smoke.py` | End-to-end coverage of loaders, splits, metrics, artifacts, CLI, dashboard routes against small CSV fixtures | The pattern P0 tests must follow. |
| `src/lib/qmlApi.ts` | `TrainRequest`, `QmlResult`, `QmlModelResult`, `QmlMetrics`, `QmlModelInfo`, `QmlPrediction` + `request<T>()` fetch helper against `VITE_QML_API_URL` | Extend with platform types; do not fork the fetch layer. |
| `src/lib/pipeline/{types,graph,runner}.ts`, `src/hooks/usePipeline.ts` | 11-node hardcoded graph (ingest → clean → features → baseline/encode → vqc → measure → classical-infer → benchmark → explain → results), `createApiRunner()` / `createMockRunner()`, `PipelineEvent` | Extend `PipelineEvent` with a `modelId`; add an assessment runner beside the existing two. |

**Verified gaps — nothing in the repository implements these today:**
`ConditionDefinition`/`ModelDefinition` registry; condition catalog; `DataBundle`/`ModalityAsset`
canonical case model; multi-model routing; the FR-009 nine-state model status set; the FR-017
six-state finding status set; DICOM/NIfTI/EDF ingestion; per-finding evidence objects;
demo/synthetic labelling as a data field; PHI-safe error redaction; export bundle.

**Verified hard constraints in the reused code that the P0 design must respect:**

1. `load_csv_dataset()` raises `ValueError: feature <name> ... is not numeric` on any non-numeric,
   non-empty cell. Every tabular adapter must emit a numeric-or-empty CSV. Empty → `np.nan` →
   handled by the existing median imputer.
2. `load_profile_dataset()` calls `validate_early_detection_profile()`, which **requires**
   `group_column`, `index_time_column`, `outcome_time_column`, `horizon_days`, and a non-empty
   `outcome_definition`. Cross-sectional datasets cannot use `load_profile_dataset()`. P0's
   executor therefore composes `load_early_detection_profile()` + `load_csv_dataset()` directly and
   records `temporal_validation: "not applicable — cross-sectional"` in provenance.
3. `run_experiment()` applies one `max_train`/`max_test` cap to **all** models in a run. Any
   FR-036 comparison must therefore share one cap across the compared models.
4. `run_repeated_experiment(repeats>1)` refuses `model_artifact_path` and disables explanations
   after the first seed. Artifact production and repeated benchmarking are separate calls.

---

## 2. Mature open-source libraries and reference implementations considered

P0 is infrastructure, so the candidates are schema/registry frameworks rather than models.

| Candidate | What it is | License | Decision |
|---|---|---|---|
| **pydantic v2** | Runtime-validated data models, JSON schema generation | MIT | **Rejected.** Adds a compiled dependency to satisfy validation that ~200 lines of `dataclasses` + explicit `from_dict` checks already cover. FR-038/SC-017 require a documented capability gap; there is none. The repo has zero validation libraries today and its own hand-rolled JSON contract loaders (`load_early_detection_profile`). |
| **MLflow Model Registry** | Model versioning, staging, artifact store, tracking server | Apache-2.0 | **Rejected for V1.** Would satisfy FR-026/FR-027 mechanically but drags in a server, a DB, and a large dependency tree onto a 5.9 GB-free disk, and its model card is free-form tags rather than the exact FR-027 field set. Reconsider only if multi-user registry hosting becomes a requirement. |
| **Hugging Face `huggingface_hub` model cards** | Model card schema + hosting | Apache-2.0 | **Partially adopted as a schema reference, not a dependency.** The `ModelCard` field set in the P0 design is aligned with the Mitchell et al. *Model Cards for Model Reporting* structure (intended use / excluded use / factors / metrics / training data / ethical considerations) that FR-027 restates, but is written as a local dataclass. |
| **FHIR R4 resource models (`fhir.resources`)** | Python bindings for the FHIR spec | MIT | **Rejected as a dependency, adopted as a naming reference.** FR-005's canonical entities (patient, encounter, observation, specimen, imaging study, provenance) map onto FHIR resource names, and the design borrows those names so a future FHIR adapter is a field-mapping exercise. Importing the full R4 model set for a 6-entity internal layer is disproportionate (FR-038). |
| **MONAI `LoadImage` / `pydicom` / `nibabel` / `pyedflib`** | Imaging and signal I/O | Apache-2.0 / MIT / MIT / BSD-3 | **Deferred, not rejected.** FR-006 lists DICOM/NIfTI/EDF, but P0 ships no model that consumes them, and the P1 imaging arm is deferred (see the P1 record). P0 defines the `ModalityAsset` contract with `modality`/`format` fields so these loaders slot in behind `platform/adapters.py` with no schema change. Registering an ingestion format with no model to consume it would produce catalog entries that can never leave "not available". |
| **`jsonschema`** | JSON Schema validation for registry files | MIT | **Rejected.** Registry files are authored in-repo and loaded by one code path; `from_dict` with explicit enum checks gives better error messages and no dependency. |

**Net new dependency count for P0: zero.** This is the FR-038/SC-017 answer.

---

## 3. Relevant published methods, data assumptions, validation design, limitations

P0 imports methodology, not code.

- **Mitchell et al., *Model Cards for Model Reporting* (FAT\* 2019).** Source of the FR-027 field
  set. Assumption: a card is authored per *version*, not per model family. Limitation: cards are
  descriptive; they do not enforce that the claims match the evaluation record. P0 mitigates this by
  making `ModelDefinition.evaluation_record_ids` a required non-empty list for any model whose
  `availability` is `available`.
- **Gebru et al., *Datasheets for Datasets* (CACM 2021).** Source of the
  `ConditionDefinition.reference_datasets` / label-policy fields and of the reference-label-tier
  concept the spec's Product Decision section requires. Limitation: datasheets are voluntary and
  most public biomedical CSVs (including the P1 Kaggle file) do not have one — which is exactly why
  the platform needs an explicit `reference_label_tier` enum rather than inferring quality.
- **TRIPOD+AI (Collins et al., *BMJ* 2024) reporting checklist for clinical prediction models.**
  Source of the required evaluation fields: participants/population, predictors, outcome definition,
  sample size, missing-data handling, model-building procedure, performance measures including
  **calibration**, and validation type (internal / temporal / external). Data assumption: a declared
  outcome and index time. Validation design: internal validation must be resampling-based, not a
  single split. Limitation: TRIPOD is a reporting standard, not a promotion gate — the spec's
  "What counts as a real gain" section supplies the gate. `EvaluationRecord`'s field list is a
  superset of the TRIPOD+AI performance items, and `run_repeated_experiment()` already emits every
  one of them for tabular tasks.
- **Van Calster et al., *Calibration: the Achilles heel of predictive analytics* (BMC Med 2019).**
  Justifies `calibration_status` being a first-class enum (`calibrated` / `uncalibrated` /
  `not_assessed` / `failed`) on `Finding.uncertainty` rather than a boolean, and justifies FR-023
  treating calibration as non-optional. The repo already computes Brier score, ECE, and calibration
  bins; P0 only has to surface them.
- **Vickers & Elkin, *Decision curve analysis* (Med Decis Making 2006).** Already implemented as
  `decision_curve()`. Limitation: net benefit is only interpretable when the prevalence in the
  evaluation sample reflects the deployment population — false for case-enriched benchmark sets.
  Recorded as a standing limitation string on any `EvaluationRecord` built from a resampled cohort.

---

## 4. License, data-access, model-weight, and redistribution constraints

- **Own code**: `quantum-health` 0.1.0 has no license declared in `backend/pyproject.toml`.
  **Action required before any distribution**: add a `license` field. Absent one, the default is
  "all rights reserved", which blocks external redistribution of the platform. Flagged, not assumed.
- **Runtime dependencies**: numpy (BSD-3), scikit-learn (BSD-3), Qiskit / Qiskit-Aer /
  Qiskit-Machine-Learning / Qiskit-Algorithms (Apache-2.0), qiskit-ibm-runtime (Apache-2.0).
  All permissive; no copyleft in the P0 path. P0 adds none.
- **Model weights**: P0 ships none. `SavedModelArtifact` is a local pickle produced by the user's
  own training run. **Constraint**: pickles execute arbitrary code on load;
  `load_model_artifact()` must only ever be pointed at registry-declared paths under the configured
  runtime directory, never at a user-supplied path from an HTTP body. This is a P0 security
  requirement, recorded here because it is a consequence of the reused artifact format.
- **Data**: P0 ships no dataset. The only bundled fixture data are the synthetic CSVs in
  `backend/tests/`. FR-030/SC-011 apply: the platform must operate only on approved benchmark or
  de-identified research data, and error strings must not echo row values.
- **Redistribution of P0 exports**: an export contains model/dataset fingerprints and metric values,
  not source rows. No dataset license is transitively triggered by exporting an
  `EvaluationRecord` — but exporting a `DataBundle` would be, so the export contract carries asset
  hashes and metadata only, never asset bytes.

---

## 5. Reproducible comparison plan

P0 has no model, so the FR-036 three-way comparison (reused reference / tuned classical / QML
candidate) is **not applicable at this level** — it is discharged per condition model, starting with
P1. What P0 must prove instead is that its wrapper is *behaviour-preserving*: routing and the
registry must not change any number the existing engine produces.

**Equivalence test (the P0 acceptance benchmark), `backend/tests/test_platform.py`:**

| # | Test | Pass condition |
|---|---|---|
| P0-1 | Run `run_repeated_experiment()` directly on the P1 encoded CSV with a fixed seed, and run the same configuration through `platform/execution.py`. | Byte-identical `metrics`, `split`, `preprocessing`, and `dataset.fingerprint` blocks. Any divergence means the wrapper is doing modelling work and must be reverted. |
| P0-2 | Route a fixture bundle carrying only `structured_clinical` against three registered models (tabular stroke, imaging stroke, an EEG stub). | Statuses exactly `ready`, `not available`, `incompatible`; zero findings with status `negative`. Covers SC-002 / SC-003. |
| P0-3 | Route a bundle whose imaging asset is present but `validation_status == "quality_failed"`. | Status `insufficient data`, finding status `insufficient data`, not `negative`. Covers FR-011 / SC-003. |
| P0-4 | Fault injection: one registered model raises inside its executor. | Its routing status is `failed` with `model_id` and reason; every other model's finding is still present in `AssessmentRun.findings`. Covers FR-012 / SC-009. |
| P0-5 | Register a new model JSON with no code change and re-run the catalog and routing paths. | The model appears in the catalog and is routed. Covers FR-026 / SC-012. |
| P0-6 | Export a completed run containing one completed, one not-evaluated, and one failed model. | Export contains all three statuses, reasons, model versions, provenance, disclaimer, and `synthetic` flag. Covers FR-029 / SC-010. |
| P0-7 | Ingest a bundle with a deliberately PHI-shaped field and force a validation error. | The raised message and every log line contain the asset id and canonical field name only, never a source value. Covers FR-030 / SC-011. |

Reproducibility: pinned via `backend/pyproject.toml`, `runtime_manifest()` (package + qiskit +
sklearn + numpy versions), the registry file's `registry_version`, and
`AssessmentRun.fingerprints` (model / dataset / preprocessing / package / backend / input).

---

## 6. Reuse decision per component (FR-037)

| Component | Decision | Evidence / reason |
|---|---|---|
| Tabular dataset loading | **adapted** | `experiment.load_csv_dataset()` — generalized to auto one-hot encode categorical text columns and recognize `N/A`-style missing sentinels, closing a gap that blocked every CSV with non-numeric columns (not just P1's), per the reuse-then-extend-fully principle rather than working around it in a one-off script |
| Dataset contract declaration | **reused** unchanged | `protocol.EarlyDetectionProfile` + `load_early_detection_profile()` |
| Early-detection temporal validation | **reused, deliberately not invoked for cross-sectional profiles** | `protocol.validate_early_detection_profile()` requires index/outcome times that a cross-sectional cohort does not have; the executor records `temporal_validation: "not applicable"` instead of faking timestamps |
| Preprocessing / feature reduction / angle scaling | **reused** unchanged | `experiment.PreprocessingPipeline` |
| Split strategies and leakage control | **reused** unchanged | `experiment._split_indices()`, `_cap_indices()` |
| Classical + quantum model training | **reused** unchanged | `experiment._build_model()`, `run_experiment()` |
| Metrics, calibration, decision curve, subgroups, bootstrap CIs | **reused** unchanged | `experiment.classification_metrics()` and siblings |
| Threshold and abstention policy | **reused** unchanged | `experiment.select_threshold()`, `abstain_margin` |
| Explanations | **adapted** | `experiment.input_sensitivity()`, `explain_raw_inputs()` — fixed a pre-existing `IndexError` in `explain_raw_inputs()` on any dataset with more than 16 raw feature columns (it indexed the impact accumulator by absolute column index instead of position in the capped list) |
| Model artifact save/load/predict | **reused** unchanged | `experiment.SavedModelArtifact` and friends |
| Nested / paired evaluation | **reused** unchanged | `study.run_nested_evaluation()` |
| Local HTTP server | **adapted** — new routes added to the existing handler | `dashboard.DashboardHandler`; no new web framework |
| Frontend API client | **adapted** — new typed fetchers beside the existing ones | `src/lib/qmlApi.ts` `request<T>()` |
| Frontend pipeline event model | **adapted** — `modelId` added to `PipelineEvent` | `src/lib/pipeline/types.ts` |
| Model-card schema | **reproduced** from Mitchell et al. 2019, as a local dataclass | No dependency added |
| Evaluation-record field set | **reproduced** from TRIPOD+AI 2024, as a local dataclass | No dependency added |
| Registry, catalog, routing, safety states | **newly authored** | Verified absent from the repository; no permissively licensed component matches the FR-009/FR-017 status contract; this is the adapter/evaluation boundary that reuse-order rule 4 explicitly permits |
| Canonical case bundle + modality assets | **newly authored** | Same; FHIR resource *names* borrowed, no FHIR dependency |
| Imaging/signal ingestion | **deferred, not authored** | No P0 or P1 model consumes it; see the P1 record |

## 7. External-asset manifest (FR-035 / SC-015)

P0 uses **no external library, checkpoint, dataset adapter, or reproduced method** beyond the
already-pinned runtime dependencies listed in section 4. Two published methods are *reproduced as
schema* (Mitchell et al. 2019 model cards; TRIPOD+AI 2024 evaluation fields); neither contributes
code, weights, or data, so neither carries a checksum or preprocessing assumption. This paragraph is
the SC-015 "explicit record explaining why no compatible reusable implementation exists" for the
registry/routing/safety layer.

## 8. Open risks carried forward

1. **`quantum-health` has no declared license.** Blocks external distribution of the platform.
2. **`SavedModelArtifact` is a pickle.** Load paths must be registry-constrained, never
   request-supplied.
3. **`run_experiment()`'s single train/test cap** applies to all models in a run, which constrains
   how the FR-036 comparison must be configured (one shared cap).
4. **FR-006 lists formats no registered model consumes.** The catalog must not imply DICOM/NIfTI/EDF
   support; those conditions stay `not available` until a model exists.
