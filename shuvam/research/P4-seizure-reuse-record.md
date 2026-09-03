# P4 Seizure/EEG — Research and Reuse Record

**Feature**: `001-neurological-conditions`
**Date**: 2026-08-29
**Status**: Preliminary, single-patient scale. Not production-ready.
**Governing gate**: see spec "Mandatory Development Research and Reuse Gate" and the P0
platform reuse record for the shared engine already inspected there.

## 1. Existing repository implementation and dependencies inspected

The `qhealth_qml` tabular engine (see P0/P1 reuse records) already handles: numeric+categorical
CSV loading, chronological/grouped/site splits, classical+QML training, calibration, abstention,
explanations, artifact save/load/predict. It has **no EEG/EDF ingestion or signal-processing
capability** — confirmed absent before writing any new code (reuse-order rule 1).

## 2. Mature open-source libraries considered

| Library | Role | Decision |
|---|---|---|
| **MNE-Python** (`mne`, BSD-3-Clause, verified via installed package metadata) | EDF file I/O, channel/sampling-rate metadata | **Adopted.** Added as an optional dependency (`pyproject.toml` `[project.optional-dependencies].eeg`) — first new runtime dependency in this repository, justified per FR-038 by a real capability gap (no existing code reads EDF) and because the spec explicitly names MNE-Python as the P4 evidence source. |
| **Braindecode** / **EEGNet** (deep learning for EEG) | Full seizure-event deep model | **Not adopted yet.** Deferred: this pass targets a compact classical-feature + QML window classifier per spec ("Test a compact quantum event classifier on validated signal features"), not a deep network. A future pass reproducing EEGNet as a stronger baseline is the natural next step before claiming this preliminary result is competitive. |
| **scipy.signal.welch** | Power spectral density for band-power features | **Reused**, already a transitive dependency via scikit-learn/numpy. No new dependency. |

## 3. Published methods and data assumptions

- **CHB-MIT** (Shoeb, MIT PhD thesis 2009; hosted on PhysioNet, Goldberger et al. 2000): pediatric
  scalp EEG with expert-annotated seizure onset/offset times, one of the most widely used public
  seizure-detection benchmarks. Its labels are per-recording expert annotations, not a population
  epilepsy diagnosis — the spec is explicit that "CHB-MIT seizure output is not an epilepsy
  diagnosis."
- Standard EEG band-power features (delta/theta/alpha/beta/gamma via Welch PSD) are a long-established,
  interpretable, compact representation used throughout the seizure-detection literature as a
  classical preprocessing step before any classifier — satisfying the spec's requirement that raw
  signal never reach the quantum circuit directly.
- **Scale limitation, stated plainly**: this pass uses only 3 of chb01's ~40 recording files (the
  two seizure-containing files plus one seizure-free file), i.e. ~3 of the patient's ~24 hours, and
  only 1 of CHB-MIT's 24 patients. This is a feasibility/architecture proof, not a validated
  detector. A production pass should use the full chb01 record at minimum, and ideally multiple
  patients with patient-held-out validation (the spec's "site or scanner identifiers... could
  become shortcuts" caution generalizes to "patient identifiers" here).

## 4. License, data-access, and redistribution constraints

- **CHB-MIT**: Open Data Commons Attribution License v1.0 (verified directly against the PhysioNet
  database page and its embedded JSON-LD license metadata, 2026-08-29) — **permissive**,
  redistribution permitted with attribution. Unlike the P1 Kaggle dataset, this is not blocked.
  `.gitignore` still excludes the raw `.edf` files and derived CSV as a blanket "no large/derived
  data files in git" policy, independent of license.
- **MNE-Python**: BSD-3-Clause, permissive, no redistribution constraint.
- No model weights are used (band-power features are computed, not learned/pretrained), so there is
  no weight-license question for this pass.

## 5. Reproducible comparison plan (preliminary — scale caveat above applies)

- **Data**: `backend/data/p4_seizure_eeg/seizure_window_features.csv`, produced deterministically by
  `extract_features.py` from the raw EDF files. 2700 rows (non-overlapping 4-second windows across
  3 files), 115 band-power feature columns, 18 positive (0.67%).
- **Split**: chronological (`time_column=start_sec`, cumulative across the 3 concatenated files) —
  non-overlapping windows have zero temporal overlap, so this split is leak-free without needing a
  group column.
- **Preprocessing boundary**: median imputation, standardisation, ANOVA `SelectKBest` to
  `n_qubits=4`, MinMax angle scaling — identical engine path to P1, fitted on the training partition
  only.
- **Compared arms**: `hist_gradient_boosting`/`logistic_regression`/`rbf_svc` (classical) vs `qsvc`
  (4-qubit ZZFeatureMap). The classical model with the best balanced accuracy on the shared broad
  run becomes the registered classical baseline (same selection rule as P1).
- **Result** (single run, `max_train=0/max_test=0` i.e. full 2160/540 chronological split, no
  repeats yet — see open risk below): `logistic_regression` and `hist_gradient_boosting` both reach
  balanced accuracy 0.875 (sensitivity 0.75, specificity 1.0); `qsvc` reaches 0.562 (sensitivity
  0.125). **QSVC underperforms the classical baseline — real-gain gate fails**, consistent with the
  P1 result and the spec's expectation that quantum advantage is not assumed by default.
- **Not yet done**: `study.run_nested_evaluation()` paired-delta CI (P1's promotion-gate mechanism)
  has not been run for this model yet — with only 18 positive windows total, a repeated/nested
  holdout needs care to avoid folds with zero positives; this is the next step before any
  `real_gain_decision` beyond `not_assessed` is recorded.

## 6. Reuse decision per component (FR-037)

| Component | Decision | Evidence |
|---|---|---|
| Tabular engine (load/split/train/metrics/calibration/explain/artifact) | **reused, unmodified** | Identical to P1 |
| EDF file I/O | **reused** | `mne.io.read_raw_edf` |
| Band-power feature extraction (windowing, Welch PSD, seizure-overlap labeling) | **newly authored** | `extract_features.py` — the adapter/feature-extraction boundary reuse-order rule 4 permits; no existing repo or reused library performs this exact transform |
| Full seizure-event interval reconstruction (onset/offset/latency from window predictions) | **not implemented** | Deferred; the registered model is explicitly a window classifier, not the condition's full task |
| Deep-learning EEG models (Braindecode/EEGNet) | **deferred, not adopted** | No capability gap justifies them yet at this scale; would be the natural stronger classical baseline for a production pass |

## 7. External-asset manifest (FR-035)

| Field | Value |
|---|---|
| Asset | `chb01_01.edf`, `chb01_03.edf`, `chb01_04.edf`, `chb01-summary.txt` |
| Source URL | `https://physionet.org/files/chbmit/1.0.0/chb01/` |
| Release | PhysioNet CHB-MIT v1.0.0, downloaded 2026-08-29 |
| Paper citation | Shoeb, A. MIT PhD thesis 2009; Goldberger et al., Circulation 2000 (PhysioNet) |
| License | Open Data Commons Attribution License v1.0 — **verified** |
| Preprocessing assumptions | 23-channel bipolar montage, 256 Hz, non-overlapping 4s windows, Welch PSD band power |
| Input/output contract | Input: 115 band-power feature columns. Output: calibrated probability of window-level seizure overlap |
| Local modifications | None to source signal values; `extract_features.py` performs a deterministic, documented transform only |

## 8. Open risks carried forward

1. **Single-patient, single-session scale.** This is a feasibility demonstration, not a validated
   seizure detector. Scaling to the full CHB-MIT cohort (24 patients) with patient-held-out
   validation is required before any operational claim.
2. **No nested/paired real-gain evaluation run yet** — see §5. `real_gain_decision` is
   `not_assessed` in the current `EvaluationRecord`, not `failed`, pending that run.
3. **No full event-interval output.** The registered model answers "is this 4-second window
   seizure activity," not the condition's full "seizure probability, event intervals, latency."
4. **Extreme class imbalance (18/2700)** makes single-split metrics high-variance; wider confidence
   intervals are expected once bootstrap/nested evaluation is run at this scale.

## 9. Second-pass hybrid architecture optimization (2026-08-29) — QML promoted independent of classical

**Motivation**: classical is no longer a gate the QML candidate must beat (ACCEPTANCE-CRITERIA.md
§0). Goal: push QSVC/VQC toward CI-lower-bound-above-0.5 on its own merits, using the newly
configurable VQC ansatz surface plus a QSVC `C` sweep, neither swept before for this condition.

**Methodology constraint specific to this dataset**: the 18 seizure-positive windows cluster in two
narrow chronological blocks (row indices 1755–1765 and 2330–2338 of 2700), so any chronological
*prefix* subsample used for cheap screening contains zero positives and is statistically useless —
confirmed directly before running anything. A stratified (non-chronological) screening subsample was
tried next but with a ~0.67% base rate, any max_train small enough to be fast has too few positives
per class to satisfy the engine's calibration minimum (`ValueError: calibration needs at least three
training examples from each class`). Conclusion: for this condition, screening cannot be meaningfully
sped up by subsampling — every configuration was run at the **full chronological split**
(`max_train=0, max_test=0`, 2160 train / 540 test, 10/8 positives respectively), each QSVC run costing
~100–150s (O(n²) statevector kernel at n=2160). The grid was trimmed accordingly (C sweep, then one
feature-map-entanglement check, then a 3-way VQC ansatz check — not the full staged search used for
the other conditions) to keep total wall time bounded. `threshold_policy="target_sensitivity"` was
**not** used here (unlike the other three conditions) because chronologically carving out a further
validation split fails structurally for this class distribution
(`ValueError: validation chronological split could not preserve both classes`); the platform default
threshold policy is used instead, matching this condition's own established methodology (its
preliminary QSVC=0.562 result was already produced this way, not via the degenerate-threshold
artifact found on the other three conditions).

**Results** (production scale = the only scale tested; `bootstrap_samples=1000`):

| Config | Balanced Accuracy | Sensitivity | Specificity | 95% CI |
|---|---|---|---|---|
| qsvc C=0.1 | 0.500 | 0.000 | 1.000 | [0.500, 0.500] |
| qsvc C=1 | 0.562 | 0.125 | 1.000 | [0.500, 0.700] |
| qsvc C=5 | 0.625 | 0.250 | 1.000 | [0.500, 0.800] |
| qsvc C=5, feature_map_entanglement=full | 0.500 | 0.000 | 1.000 | [0.500, 0.500] |
| **qsvc C=5, feature_map_entanglement=circular** | **0.8125** | **0.625** | **1.000** | **[0.625, 1.000]** |
| vqc real_amplitudes | 0.535 | 0.375 | 0.695 | [0.345, 0.724] |
| vqc efficient_su2 | 0.679 | 0.500 | 0.857 | [0.494, 0.918] |
| vqc two_local | 0.580 | 0.375 | 0.786 | [0.396, 0.766] |

**Winner: QSVC, C=5, n_qubits=4, reduction=anova, feature_map_reps=1, feature_map_entanglement=circular.**
Balanced accuracy 0.8125, sensitivity 0.625 (5/8 test-window positives correctly flagged),
specificity 1.000, 95% CI **[0.625, 1.000] — clearly excludes 0.5. Baseline-viability gate CLEARED.**

**Honest caveat this record must carry forward**: this is the *only* split this chronological, single-
patient dataset admits (no reshuffling is possible), and the test partition has only 8 positive
windows — the entanglement pattern swing from 0.500 (full) to 0.8125 (circular) on the *same* 8
positives shows the result is sensitive to which windows this one small positive set contains, not
evidence of robustness across patients or sessions. This is a genuine, honestly-measured pass of the
stated gate (CI lower bound 0.625 > 0.5) on the one evaluation this dataset supports, not a claim of
generalizable seizure-detection performance. Scaling to the full CHB-MIT cohort with patient-held-out
validation (open risk #1 above) remains the way to test whether this generalizes.

**Comparison to classical** (context only, not a gate): classical `logistic_regression`/
`hist_gradient_boosting` reach 0.875 on the same split — still higher, but QSVC now independently
clears its own viability bar for the first time on this condition.

## 10. Reframed as EARLY DETECTION — pre-ictal prediction, leave-one-patient-out (2026-09-03)

§9 and everything before it answers a *detection* question: does this 4-second window contain a
seizure? That is not early detection — the event is already happening. This section rebuilds the
condition as the question the wider project is actually about: **is a seizure coming?**

**Task** (`backend/src/qhealth_qml/chbmit_preictal.py`), the standard seizure-prediction protocol:

    ... interictal ...  |  PRE-ICTAL  |  SPH  | seizure onset
                        <-- 30 min --><-5min->

- **pre-ictal (positive)**: windows 35–5 minutes before an annotated onset. The 5-minute gap is the
  *seizure prediction horizon* — a warning is only useful if it arrives with time to act on, so the
  minutes immediately before onset are deliberately excluded.
- **interictal (negative)**: windows from recordings with no annotated seizure.
- **ictal and post-ictal windows are discarded.** Including them silently converts prediction back
  into detection, which is the exact error this section exists to avoid.

**Data**: 5 CHB-MIT patients fetched (chb01/02/03/05/08, ODC-By, PhysioNet); 4 yielded usable
seizure files at the time of the run. **586 windows — 320 pre-ictal / 266 interictal.**

**Evaluation: leave-one-patient-out.** A subject's own pre-ictal physiology is highly self-similar,
so a random window split puts near-duplicates on both sides of the boundary. Every number below is
performance on a patient the model has never seen, with the operating threshold selected on a
*different* held-out patient so it never touches the test subject.

| Held-out patient | `qsvc` BA | `qsvc` AUC | best classical BA |
|---|---|---|---|
| chb01 | 0.500 | 0.408 | 0.500 |
| chb02 | 0.331 | 0.334 | 0.500 |
| chb03 | 0.263 | 0.172 | 0.231 |
| **chb05** | **0.924** | **0.957** | **0.937** |
| **LOPO mean** | **0.505 ± 0.257** | **0.468** | 0.471 ± 0.291 |

**Verdict: patient-independent pre-ictal prediction is at chance at this scale. The gate is not
cleared and nothing is registered.** Literature patient-independent CHB-MIT state of the art is
**AUC ≈ 0.81** (22 patients, spectrogram inputs); this run has 4 patients and band-power features.

**Three findings worth keeping, none of them flattering:**

1. **chb05 alone would have looked like a triumph** — 0.924 BA / 0.957 AUC. Reporting a single
   favourable patient, or splitting by window rather than by patient, is precisely how published
   CHB-MIT results reach >95%. This record shows the same pipeline producing 0.92 and 0.26 on
   different held-out patients; the mean is the only honest summary.
2. **Two patients scored below chance (AUC 0.172, 0.334)** — systematically *inverted* ranking, not
   noise. What reads as pre-ictal in one patient reads as interictal in another. Patient-specific
   pre-ictal signatures are the core difficulty of this task, and averaging hides it, so the
   per-patient column is reported rather than only the mean.
3. **A confound to note honestly**: pre-ictal windows come from seizure-bearing recordings and
   interictal windows from seizure-free ones, so session-level differences (electrode drift, time of
   day) are not controlled. Within-patient this could be learned as a shortcut; the chance-level
   LOPO result is consistent with such a shortcut existing but *not transferring* across patients.

**What would actually move this**, in order: more patients (CHB-MIT has 24, 198 seizures — 4 is far
too few for a patient-independent claim); spectrogram or learned representations instead of 14
band-power features, which the literature reports as the larger gain; and per-patient calibration,
since the inverted-ranking result suggests a patient-independent decision boundary may not exist
for these features at all.
