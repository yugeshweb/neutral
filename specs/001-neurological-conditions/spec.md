# Feature Specification: Neurological Condition Assessment

**Feature Branch**: `001-neurological-conditions`
**Created**: 2026-08-29
**Status**: Draft
**Input**: User description: "Add a Neurological Conditions feature that lets a user submit heterogeneous biomedical data and receive findings from a collection of validated hybrid quantum-classical models."

## Product Decision

This feature is a **neurological domain entry point**, not one universal neurological classifier.
The user selects **Neurological conditions** once; the platform then discovers the available
modalities and runs every registered model that is compatible with the supplied data.

The platform MUST distinguish three different meanings of confidence:

1. **Reference-label confidence**: how reliable the training/evaluation reference standard is.
2. **Model confidence**: a calibrated prediction score and uncertainty estimate for the current case.
3. **Data coverage**: whether the required, valid modalities were actually available.

The platform MUST NOT present any of these as a definitive medical diagnosis.

## Recommended Build Order

The priority order is based on four factors: reference-label strength, multimodal platform value,
availability of mature open-source implementations, and the cost of acquiring or preparing the data.
P0 is shared platform work; P1 is the first disease model.

| Priority | Work package | Why it comes here | Initial open-source/research base | Quantum role |
|---|---|---|---|---|
| P0 | Model registry, canonical case bundle, routing, evaluation, provenance, and safety states | Required once for every later condition | Existing `qhealth_qml` Qiskit/scikit-learn pipeline, artifacts, grouped/time/site evaluation, and result contracts | Reuse the current QSVC/VQC path; do not add another quantum framework yet |
| P1 | Acute ischemic stroke lesion detection/segmentation plus a case-level clinical/outcome head | Best first end-to-end demonstration of early recognition, imaging, structured clinical data, expert labels, and site generalization | [DeepISLES](https://github.com/ezequieldlrosa/DeepIsles), [DeepISLES paper](https://www.nature.com/articles/s41467-025-62373-x), [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [MONAI](https://github.com/Project-MONAI/MONAI) | Keep segmentation classical; test QML on a compact fused image/clinical representation for classification or regression |
| P2 | Acute intracranial hemorrhage and five CT subtypes | Strong radiologist labels, large benchmark, and a relatively contained CT/multilabel task | [RSNA ICH dataset/challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019), MONAI imaging components | Test a multilabel quantum head only after a strong CT baseline and calibration |
| P3 | Glioma/brain-tumor lesion characterization | Extends the platform to multiparametric MRI plus clinical/genomic/pathology-linked data | [BraTS](https://www.med.upenn.edu/cbica/brats2020/data.html), [TCGA-GBM/TCIA](https://www.cancerimagingarchive.net/collection/tcga-gbm/), nnU-Net, MONAI | Test quantum fusion for grade, subtype, or outcome; do not replace a proven segmentation baseline |
| P4 | EEG seizure activity/onset | Adds physiological time-series ingestion and event-level evaluation | [MNE-Python](https://github.com/mne-tools/mne-python), [Braindecode](https://github.com/braindecode/braindecode), EEGNet, [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) | Test a compact quantum event classifier on validated signal features; output is not an epilepsy diagnosis |
| P5 | Alzheimer’s MCI-to-AD progression risk | Rich multimodal and longitudinal data, but less definitive labels and stricter leakage risks | [ADNI](https://adni.loni.usc.edu/data-samples/adni-data/), [OASIS](https://dev.oasis-brains.org/), classical longitudinal baselines | QML is an explicitly exploratory hypothesis, not a default improvement |
| P6 | Parkinson’s/prodromal Parkinson’s risk | Broad PPMI modalities, but controlled data access and probabilistic prodromal labels | [PPMI](https://www.ppmi-info.org/access-data-specimens/data) and classical multimodal baselines | Test only after the missing-modality and longitudinal model contracts are proven |

The first disease model MUST therefore be P1 stroke. P2 should be the first fast follow-up model;
P3 should be the first model that demonstrates imaging plus clinical/genomic fusion. P5 and P6 may
appear in the catalog earlier, but they MUST remain research-only until their label and calibration
requirements are met.

## Mandatory Development Research and Reuse Gate

This is a blocking development instruction for every condition model. Before implementing,
adapting, or promoting model-specific code, the developer MUST complete a versioned research and
reuse record. The record MUST be reviewed with the model change and MUST answer what existing work
was inspected, what will be reused, what gap remains, why new code is necessary, and how any claimed
gain will be tested. A model implementation is not ready for merge or registry promotion without
this record.

The record MUST cover, in order:

1. The existing repository implementation and dependencies.
2. Mature open-source libraries, checkpoints, or reference implementations for the task.
3. Relevant published methods, including their data assumptions, validation design, and reported
   limitations.
4. License, data-access, model-weight, and redistribution constraints.
5. A reproducible comparison plan covering a strong reused baseline, a tuned classical baseline,
   and the proposed change.

The project MUST follow this reuse order before writing new model code:

1. Reuse an existing repository implementation already present in this project.
2. Reuse a mature open-source library or released model with a compatible license and reproducible
   inputs/outputs.
3. Reproduce a published method when its code or algorithm is available and the task matches.
4. Write new code only at the adapter, fusion, evaluation, or quantum-experiment boundary where the
   existing work does not satisfy the contract.

Every reused implementation MUST have a reuse manifest containing its repository URL, commit or
release, paper citation, license, model-weight source and checksum when applicable, preprocessing
assumptions, input/output contract, and any local modifications. Dependency and model-weight
licenses MUST be checked before distribution.

The current repository already contains the Qiskit Machine Learning execution path, grouped and
chronological evaluation support, preprocessing artifacts, calibration/abstention fields, and result
contracts. New condition models SHOULD extend those paths rather than create parallel QML runners.
Qiskit remains the default quantum stack. PennyLane or another framework may be added only when a
required published method cannot be implemented with the existing stack and a controlled benchmark
shows a material benefit.

### What counts as a real gain

A new model, quantum component, or research method MUST NOT be called an improvement merely because
it has a higher single-split accuracy. A promotion claim requires:

- The same patients, acquisition units, preprocessing boundary, and declared train/validation/test
  split for every compared model.
- A strong open-source reference implementation and a tuned classical baseline, not only a weak
  custom baseline.
- Patient-level leakage checks and site-held-out or external validation where the dataset permits it.
- Confidence intervals or repeated evaluation and a paired comparison when predictions are paired.
- Improvement in a task-appropriate metric without an unacceptable regression in calibration,
  sensitivity, specificity, abstention coverage, runtime, or resource use.
- A reproducible run from pinned code, data version, model weights, configuration, and environment.

If the QML model does not pass this gate, it MUST remain an experimental benchmark result. The
platform MUST still be allowed to ship the stronger classical or reused model as the operational
reference.

The development stop rule is strict: if an existing implementation satisfies the model contract,
the default work is an adapter, reproducibility harness, or evaluation integration. Rewriting the
covered capability requires a documented technical reason. Adding a new quantum framework or
dependency requires the same evidence and a benchmark showing a material benefit; “quantum” or
architectural novelty alone is not sufficient.

## Supported Condition Catalog

The following catalog is the complete launch scope for this feature. “High-reference” means the
task has a comparatively strong reference standard for research evaluation; it does not mean the
result is clinically guaranteed.

### High-reference-confidence launch candidates

| Condition/task | Primary data | Reference standard | Output | Evidence source |
|---|---|---|---|---|
| Acute ischemic stroke lesion/core detection | Presentation CT/CTA/CT-perfusion, optional structured clinical data | Expert lesion annotation and/or follow-up diffusion MRI | Case risk, lesion mask/regions, infarct-core estimate | [ISLES’24](https://pubs.rsna.org/doi/full/10.1148/ryai.250603) |
| Acute intracranial hemorrhage | Non-contrast head CT, optional structured metadata | Radiologist-labeled hemorrhage and subtype annotations | Any-hemorrhage risk plus subtype findings: epidural, intraparenchymal, intraventricular, subarachnoid, subdural | [RSNA ICH Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019) |
| Seizure activity/onset | Continuous EEG, optional patient metadata | Expert-annotated seizure onset and end times | Seizure probability, event intervals, latency | [CHB-MIT PhysioNet](https://physionet.org/content/chbmit/1.0.0/) |
| Glioma/brain-tumor lesion characterization | Multiparametric MRI; optional clinical, genomic, or pathology-linked data | Pathologically confirmed tumor cohort and expert MRI annotations | Tumor regions, glioma characterization, grade/risk where supported | [BraTS data](https://www.med.upenn.edu/cbica/brats2020/data.html), [TCGA-GBM/TCIA](https://www.cancerimagingarchive.net/collection/tcga-gbm/) |

These tasks are high-reference-confidence because they use expert image/event annotations or
pathology-linked cohorts. They still have important limits: stroke and hemorrhage are acute
recognition tasks, CHB-MIT seizure output is not an epilepsy diagnosis, and BraTS/TCGA primarily
contain known tumor cases rather than population-screening controls.

### Research-only launch candidates

These conditions can be registered in the catalog, but their findings MUST be labelled as
research risk/progression predictions until the model has an approved label policy and a
biomarker-supported evaluation set.

| Condition/task | Primary data | Reference standard | Output | Evidence source |
|---|---|---|---|---|
| Alzheimer’s disease progression | Cognitive/clinical data, MRI/PET, biomarkers, genetics | Longitudinal MCI-to-AD outcome and/or biomarker-supported cohort label | Progression risk, not confirmed diagnosis | [ADNI data](https://adni.loni.usc.edu/data-samples/adni-data/), [OASIS](https://dev.oasis-brains.org/) |
| Parkinson’s disease/prodromal Parkinson’s | Clinical data, imaging, sensors, genetics, biomarkers | Longitudinal clinical/prodromal cohort label with available biomarker evidence | Classification or progression risk, not confirmed diagnosis | [PPMI data](https://www.ppmi-info.org/access-data-specimens/data) |

The feature MUST NOT claim that the catalog exhausts all neurological diseases. It is exhaustive
only for the conditions supported by a registered dataset, label definition, model artifact,
evaluation record, and safety metadata.

## Scope Boundaries

### In scope

- A **Neurological conditions** selection in the product entry flow.
- A registry of condition definitions and compatible model versions.
- Dataset-specific adapters that map heterogeneous sources into a canonical internal bundle.
- Structured clinical data, medical imaging, physiological signals, biomarkers, and selected
  genomic inputs when a registered model declares support for them.
- Routing one case to multiple compatible models and presenting independent findings.
- Classical preprocessing and feature extraction followed by a hybrid quantum-classical model.
- A paired classical baseline for every QML model.
- Finding-level explanations, uncertainty, provenance, and missing-data states.
- Research/demo mode with explicit synthetic-result labelling.

### Out of scope for this feature

- A single model that claims to diagnose every neurological disease.
- A universal cross-condition “neurological risk” score.
- Autonomous diagnosis, treatment recommendations, triage decisions, or medical actions.
- Accepting arbitrary hospital EHR exports without an adapter or schema mapping.
- Treating a PDF report as a normalized EHR record. PDF/OCR/NLP ingestion is a later feature.
- Feeding raw 3D images, raw EEG streams, or full omics matrices directly into a quantum circuit.
- Enabling multiple sclerosis, ALS, Huntington’s disease, migraine, traumatic brain injury,
  neuropathies, or other conditions without a registered dataset and evaluation record.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Discover neurological coverage (Priority: P1)

As a user, I want to select **Neurological conditions** and see what the platform can assess so
that I understand the available coverage before providing data.

**Why this priority**: The category is the primary entry point and prevents the user from assuming
that the platform can assess every neurological condition.

**Independent Test**: Can be fully tested from the launch screen with no uploaded data and delivers
the supported catalog, required modalities, task type, and readiness level.

**Acceptance Scenarios**:

1. **Given** the user is on the launch screen, **When** they select Neurological conditions,
   **Then** the platform displays the high-reference and research-only condition groups.
2. **Given** a condition is listed, **When** the user opens its details, **Then** the platform shows
   its task, required modalities, expected output, reference dataset, readiness level, and limitation.
3. **Given** a condition has no registered model artifact, **When** the catalog is displayed,
   **Then** it is marked unavailable and cannot be represented as a negative finding.

### User Story 2 - Provide heterogeneous case data (Priority: P1)

As a user or dataset operator, I want to provide structured clinical data and supported biomedical
assets in one case bundle so that different models can use the modalities they require.

**Why this priority**: The platform’s value depends on handling more than one narrow file type or
single modality.

**Independent Test**: Can be tested with fixture bundles containing structured data, an imaging
asset, and an optional signal or biomarker asset; validation reports what was accepted, rejected,
or missing.

**Acceptance Scenarios**:

1. **Given** a bundle contains valid canonical clinical data and a supported imaging asset,
   **When** ingestion completes, **Then** the platform displays the parsed modalities, dimensions,
   timestamps, provenance, and validation status.
2. **Given** an external source uses vendor-specific EHR fields, **When** its adapter maps the
   source to the canonical schema, **Then** the original field and mapping are retained as provenance.
3. **Given** a bundle contains an unsupported file or invalid unit, **When** validation runs,
   **Then** the asset is rejected with an actionable reason and valid assets remain available.

### User Story 3 - Run all compatible neurological models (Priority: P1)

As a user, I want one assessment run to execute every compatible condition model so that I receive
all supported findings from the data I supplied.

**Why this priority**: The user should select the neurological domain rather than manually launching
individual disease models.

**Independent Test**: Can be tested with a fixture bundle that satisfies two model contracts and
omits a third modality; the two compatible models complete and the third reports not evaluated.

**Acceptance Scenarios**:

1. **Given** a case contains the required inputs for stroke and hemorrhage models, **When** the
   user starts an assessment, **Then** both models are scheduled and their findings are shown
   independently.
2. **Given** a case lacks EEG, **When** the assessment runs, **Then** the seizure model reports
   “not evaluated — EEG unavailable” rather than “no seizure detected.”
3. **Given** one model fails, **When** other compatible models finish, **Then** their findings remain
   visible and the failed model shows an error with its model identifier and reason.
4. **Given** no compatible model can run, **When** the user starts an assessment, **Then** the
   platform explains why and does not generate a reassuring or negative neurological result.

### User Story 4 - Understand each finding (Priority: P2)

As a user, I want to inspect the evidence behind a finding so that I can distinguish a model score
from a definitive diagnosis.

**Why this priority**: A multimodel report without input coverage, uncertainty, and evidence would
be unsafe and difficult to interpret.

**Independent Test**: Can be tested by opening a completed finding and verifying that all required
  provenance, uncertainty, evidence, and limitation fields are rendered.

**Acceptance Scenarios**:

1. **Given** a model produces a case-level result, **When** the user opens the finding, **Then** the
   platform shows the target, score type, calibrated status, uncertainty or abstention state,
   model version, modalities used, and limitations.
2. **Given** a model produces image regions, **When** the user opens the finding, **Then** the
   platform highlights the returned regions and shows their labels, measurements, and confidence.
3. **Given** a model produces an EEG event, **When** the user opens the finding, **Then** the
   platform shows the event interval and the signal window used for the explanation.
4. **Given** a model has no valid explanation for a result, **When** the user opens the finding,
   **Then** the platform says explanation unavailable rather than inventing evidence.

### User Story 5 - Compare hybrid and classical approaches (Priority: P2)

As a researcher, I want to compare a hybrid quantum-classical model with a classical baseline on
the same evaluation protocol so that I can assess whether the quantum component adds value.

**Why this priority**: Benchmarking against classical models is a core requirement of the project
and prevents unsupported claims of quantum advantage.

**Independent Test**: Can be tested with a registered training/evaluation profile and verifies that
both model families use the same subject-level split, target, preprocessing boundary, and metrics.

**Acceptance Scenarios**:

1. **Given** a QML model has an evaluation profile, **When** benchmarking completes, **Then** the
   platform reports its metrics beside the paired classical baseline metrics.
2. **Given** the hybrid model performs worse or is not statistically distinguishable, **When** the
   comparison is displayed, **Then** the platform reports that result without claiming an advantage.
3. **Given** a model uses a patient, site, or temporal grouping rule, **When** benchmarking runs,
   **Then** the same grouping rule is visible in the comparison metadata.

### User Story 6 - Export a reproducible assessment (Priority: P3)

As a researcher, I want to export the findings and run metadata so that another person can review
what data and model produced the result.

**Why this priority**: Reproducibility and auditability are necessary for biomedical benchmarking,
but are downstream of correct routing and findings.

**Independent Test**: Can be tested by exporting a completed run and checking that the exported
artifact contains the same findings, statuses, model versions, and provenance shown in the UI.

**Acceptance Scenarios**:

1. **Given** an assessment has completed, **When** the user exports it, **Then** the export contains
   the run identifier, condition/task, model version, data identifiers, input coverage, score,
   uncertainty, explanations, limitations, and synthetic flag.
2. **Given** the run used demo data, **When** the export is opened outside the application,
   **Then** the synthetic/demo status and research-only disclaimer remain visible.
3. **Given** an assessment contains a failed or not-evaluated model, **When** it is exported,
   **Then** that status and reason are preserved rather than omitted.

## Edge Cases

- A user provides only structured clinical data; image-, EEG-, and signal-dependent models remain
  not evaluated unless they explicitly support the available inputs.
- A user provides an imaging file without a patient or study identifier; the platform requests a
  case identifier or assigns a temporary local identifier and clearly marks it as such.
- Multiple scans or visits belong to one patient; the platform preserves visit timestamps and does
  not silently select one record without reporting the selection rule.
- A case contains repeated observations, duplicate studies, conflicting units, or impossible
  timestamps; validation flags the issue and does not silently overwrite source values.
- A model requires a modality that is present but fails quality control; the model abstains or
  reports insufficient data instead of treating the modality as normal.
- A model supports optional modalities but was not trained for missingness; it must not run with
  improvised imputation.
- The same patient appears in multiple recordings, scans, or visits; train/validation/test splits
  must prevent subject leakage.
- A dataset contains site or scanner identifiers that could become shortcuts; the evaluation must
  expose the grouping strategy and permit site-held-out validation where available.
- A QML backend is unavailable, too large for the configured qubit budget, or exceeds a resource
  limit; the run reports a quantum execution error and does not substitute a classical result under
  the quantum model’s name.
- A model returns a low-margin score or fails calibration checks; it must abstain or show an
  uncalibrated score rather than a high-confidence finding.
- A model has a high score but the data coverage is low; the UI must show the coverage limitation
  prominently.
- One model errors while other models complete; completed findings remain available.
- No model supports the supplied data; the result is “no compatible assessment,” not “no disease.”
- Uploaded content contains identifiable health information; raw values must not appear in logs or
  demo exports, and the feature remains restricted to approved/de-identified research data in V1.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a Neurological conditions domain entry point.
- **FR-002**: The system MUST display the launch condition catalog grouped by readiness and reference-label confidence.
- **FR-003**: Each catalog entry MUST declare its condition, prediction task, supported population,
  required and optional modalities, expected output type, reference dataset, label definition,
  limitations, and model availability.
- **FR-004**: The system MUST represent each assessment as a case bundle that can contain one or
  more structured, imaging, physiological-signal, biomarker, genomic, or document-derived assets.
- **FR-005**: The system MUST support a canonical internal representation for patient, encounter,
  observation, specimen, imaging study, signal series, and provenance data.
- **FR-006**: The system MUST allow source-specific adapters to map CSV, JSON, FHIR-like, DICOM,
  NIfTI, EDF/EDF+, and approved dataset formats into the canonical representation.
- **FR-007**: The system MUST preserve the source field, source value, code system, unit, timestamp,
  mapping decision, and transformation provenance for mapped data when available.
- **FR-008**: The system MUST validate required identifiers, modality type, file integrity, schema,
  units, dimensions, timestamps, and model-specific quality constraints before model execution.
- **FR-009**: The system MUST classify each registered model as compatible, incompatible, not
  available, insufficient data, ready, running, completed, abstained, or failed for a case.
- **FR-010**: The system MUST route a case only to models whose declared input and population
  contracts are satisfied.
- **FR-011**: Missing, invalid, or low-quality inputs MUST NOT be converted into a negative disease
  result unless the model was explicitly trained and evaluated for that missing-input condition.
- **FR-012**: The system MUST run compatible model assessments independently so that one model’s
  failure does not suppress completed findings from other models.
- **FR-013**: Each QML model MUST expose its classical preprocessing, feature-reduction method,
  quantum encoding, circuit version, backend, qubit count, shot configuration, and model artifact.
- **FR-014**: Each QML model MUST have a paired classical baseline evaluated against the same target,
  subject-level split, preprocessing boundary, and test cases.
- **FR-015**: The system MUST support model outputs as binary or multilabel scores, segmentation
  regions/masks, event intervals, regression values, or progression-risk scores according to the
  model contract.
- **FR-016**: Every finding MUST include condition, task, assessment status, model score or output,
  score type, model version, input modalities used, reference-label tier, data quality/coverage,
  uncertainty or abstention state, and limitations.
- **FR-017**: The system MUST distinguish “not evaluated,” “not available,” “insufficient data,”
  “negative,” “positive,” and “abstained” in both the UI and exported results.
- **FR-018**: The system MUST display model scores as research predictions or risk estimates and
  MUST NOT label them as definitive diagnoses.
- **FR-019**: The system MUST provide model-specific explanations when available, including top
  structured features, image regions, signal windows, or biomarker contributions.
- **FR-020**: The system MUST state when an explanation is unavailable, not applicable, or derived
  from a surrogate feature representation.
- **FR-021**: The system MUST prevent patient, visit, acquisition, or repeated-record leakage across
  training, validation, and test splits when the model is trained or evaluated.
- **FR-022**: Preprocessing, feature selection, dimensionality reduction, imputation, and threshold
  selection MUST be fitted only on the permitted training/validation partitions and recorded.
- **FR-023**: Evaluation records MUST include sensitivity, specificity, balanced accuracy, ROC-AUC,
  PR-AUC, calibration or calibration status, coverage/abstention, resource usage, and runtime when
  the metric applies to the task.
- **FR-024**: Segmentation models MUST additionally report an appropriate overlap metric such as
  Dice or IoU; event models MUST report event-level detection and latency metrics.
- **FR-025**: The system MUST report the difference between hybrid and classical results without
  asserting quantum advantage unless a documented evaluation gate supports that claim.
- **FR-026**: The system MUST allow a model to be added or replaced through the model registry without
  changing domain selection or generic finding aggregation behavior.
- **FR-027**: A model registry entry MUST include a model card containing intended use, excluded use,
  training/evaluation population, label policy, data sources and licenses, limitations, calibration
  status, explanation method, and responsible maintainer.
- **FR-028**: The system MUST preserve model, dataset, preprocessing, code/package, backend, and
  input fingerprints in an assessment record.
- **FR-029**: Exports MUST include all findings, not-evaluated states, failures, provenance, model
  versions, evaluation metadata, disclaimer text, and synthetic/demo status.
- **FR-030**: V1 MUST operate only on approved benchmark or de-identified research data and MUST NOT
  write raw protected health information into application logs, metrics, or error messages.
- **FR-031**: Demo-mode output MUST be visibly marked synthetic and MUST NOT be represented as model
  inference, clinical evidence, or a real finding.
- **FR-032**: The initial high-reference-confidence catalog MUST include acute ischemic stroke,
  acute intracranial hemorrhage, seizure activity/onset, and glioma/brain-tumor characterization.
- **FR-033**: Alzheimer’s progression and Parkinson’s/prodromal Parkinson’s MUST remain research-only
  catalog entries unless their model cards document biomarker/longitudinal label policy and approved
  evaluation evidence.
- **FR-034**: Before model-specific implementation begins, the system repository MUST contain a
  versioned research/reuse record covering existing local code, relevant open-source work,
  published methods, licensing/access constraints, and the comparison plan.
- **FR-035**: Every external library, checkpoint, dataset adapter, or reproduced method used by a
  model MUST record its source URL, release or commit, paper citation when applicable, license,
  model-weight source and checksum when applicable, preprocessing assumptions, input/output
  contract, and local modifications.
- **FR-036**: Every QML candidate MUST be compared with a strong reused or published reference
  implementation and a tuned classical baseline on the same declared data split, preprocessing
  boundary, target, and evaluation metrics before it can be promoted beyond experimental status.
- **FR-037**: The platform MUST record whether each model component is reused, adapted, reproduced,
  or newly authored, and MUST preserve the evidence and decision for that classification in the
  model card.
- **FR-038**: A new framework, dependency, encoder, or model implementation MUST NOT duplicate an
  existing capability without a documented gap, compatibility need, or benchmarked material gain.
- **FR-039**: A QML component MUST NOT replace a stronger classical or open-source component in the
  operational path unless the real-gain gate passes without an unacceptable regression in the
  task metric, calibration, sensitivity, specificity, coverage, runtime, or resource use.
- **FR-040**: When a candidate fails the real-gain gate, the system MUST retain the stronger baseline
  as the reference implementation and label the candidate as experimental in code, registry,
  evaluation reports, and user-facing research output.

### Key Entities

- **ConditionDefinition**: A neurological condition and task supported by the platform. Includes
  condition identifier, task type, readiness tier, target definition, population, required
  modalities, optional modalities, expected output, evidence links, and limitations.
- **ModelDefinition**: A versioned model registered for a condition. Includes input contract,
  preprocessing, classical baseline, quantum configuration, artifact references, calibration,
  explainability, evaluation, and safety metadata.
- **DataBundle**: The case-level collection submitted for assessment. Includes a pseudonymous case
  identifier, visits, assets, timestamps, source information, validation results, and data hashes.
- **ModalityAsset**: A structured observation, clinical document, imaging study, physiological signal,
  biomarker panel, genomic feature set, or derived feature representation.
- **AssessmentRun**: A reproducible execution over a DataBundle. Includes run status, selected models,
  routing decisions, stage events, resource usage, and exported provenance.
- **Finding**: A model-specific result. Includes condition, task, status, score/output, uncertainty,
  evidence, input coverage, model version, explanation, and limitation text.
- **EvidenceItem**: A feature contribution, image region, signal interval, biomarker, quality flag,
  or reference link supporting or qualifying a Finding.
- **EvaluationRecord**: The benchmark record for a model and baseline. Includes dataset profile,
  split strategy, leakage checks, metrics, calibration, abstention, resource measurements, and
  confidence intervals where available.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can enter the Neurological conditions domain and see every launch-catalog entry,
  its readiness tier, required modalities, task, limitations, and availability without uploading data.
- **SC-002**: For a representative fixture bundle, 100% of compatible registered models are routed
  for execution and 100% of incompatible models are shown as not evaluated with a reason.
- **SC-003**: Across validation tests, 0 missing, invalid, or unsupported modality cases are reported
  as a negative disease finding.
- **SC-004**: At launch, the registry contains four high-reference-confidence model contracts:
  stroke, intracranial hemorrhage, seizure activity/onset, and glioma/brain-tumor characterization.
- **SC-005**: Alzheimer’s and Parkinson’s entries, if displayed, are visibly marked research-only
  and produce risk/progression language rather than definitive-diagnosis language.
- **SC-006**: 100% of completed findings contain a condition, task, assessment status, model version,
  data coverage, score/output, uncertainty or calibration status, limitations, and explanation state.
- **SC-007**: 100% of benchmark reports pair each QML result with a classical baseline evaluated on
  the same declared test cases and split strategy.
- **SC-008**: Every benchmark run produces an auditable leakage report covering subject identity,
  repeated acquisition, site grouping, and training-only preprocessing.
- **SC-009**: A failure in one model does not remove completed findings from other models in 100% of
  multi-model fault-injection tests.
- **SC-010**: Exported reports preserve 100% of displayed findings, statuses, model versions,
  provenance, limitations, and synthetic/demo indicators.
- **SC-011**: No raw protected health information is emitted in application logs during ingestion,
  validation, model execution, failure handling, or export tests.
- **SC-012**: A new registry-compliant model can be added and discovered by the neurological domain
  without changes to generic routing, status handling, or finding aggregation.
- **SC-013**: No launch model is described as clinically validated or quantum-advantaged unless its
  model card contains the required evaluation, calibration, and external-validation evidence.
- **SC-014**: 100% of model-specific implementation and promotion changes link a completed research/
  reuse record before review or registry activation.
- **SC-015**: 100% of enabled models have a complete external-asset manifest or an explicit record
  explaining why no compatible reusable implementation exists.
- **SC-016**: 100% of QML benchmark reports include a strong reused/reference baseline, a tuned
  classical baseline, the declared split and preprocessing boundary, and the real-gain decision.
- **SC-017**: 0 new framework or dependency additions pass review without a documented capability
  gap, compatibility requirement, or benchmarked material benefit.
- **SC-018**: 0 QML candidates that fail the real-gain gate are presented as the operational reference
  or as evidence of quantum advantage.

## Assumptions

- V1 is a research and demonstration platform, not a regulated clinical device or autonomous
  diagnostic service.
- The initial user supplies approved benchmark or de-identified research data; live patient PHI is
  outside the V1 operating boundary.
- “EHR input” means structured clinical data or a mapped source export. A report PDF is treated as
  an unstructured document and is not assumed to be a normalized EHR.
- The canonical schema is an internal interoperability layer, not a promise that all hospitals use
  the same source format or clinical vocabulary.
- Dataset access, licensing, and usage restrictions may limit which model artifacts can be bundled
  with the application.
- Raw imaging and signal data are handled by classical feature extractors or encoders before the
  compact representation reaches the quantum model.
- Model-specific abstention is preferable to an unsupported prediction when data quality, coverage,
  calibration, or resource constraints are inadequate.
- The current application may retain an explicit synthetic/demo mode while real model execution is
  introduced; synthetic outputs must remain visibly separated from real evaluation results.
- No minimum accuracy threshold is shared across all conditions. Promotion criteria are declared per
  model because binary classification, multilabel detection, segmentation, event detection, and
  progression prediction have different valid metrics and clinical meanings.

## Research Basis

The condition catalog and confidence policy are based on the following primary sources:

- [GitHub Spec Kit specification template](https://raw.githubusercontent.com/github/spec-kit/main/templates/spec-template.md)
- [ISLES’24 multimodal stroke dataset](https://pubs.rsna.org/doi/full/10.1148/ryai.250603)
- [RSNA acute intracranial hemorrhage challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019)
- [CHB-MIT scalp EEG database](https://physionet.org/content/chbmit/1.0.0/)
- [BraTS multimodal glioma MRI data](https://www.med.upenn.edu/cbica/brats2020/data.html)
- [TCGA-GBM matched imaging and clinical/genomic collection](https://www.cancerimagingarchive.net/collection/tcga-gbm/)
- [ADNI multimodal Alzheimer’s research data](https://adni.loni.usc.edu/data-samples/adni-data/)
- [OASIS multimodal aging and Alzheimer’s data](https://dev.oasis-brains.org/)
- [PPMI Parkinson’s data](https://www.ppmi-info.org/access-data-specimens/data)
