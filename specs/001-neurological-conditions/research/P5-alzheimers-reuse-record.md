# P5 Alzheimer's — Research and Reuse Record

**Feature**: `001-neurological-conditions`
**Date**: 2026-08-29
**Status**: Research-only per spec FR-033. Same-visit association proxy, not progression risk.

## 1. Existing repository implementation inspected

The `qhealth_qml` tabular engine (see P0/P1 reuse records) is reused unchanged. No existing
Alzheimer's-specific code or dataset existed before this pass.

## 2. Mature open-source libraries / reference implementations considered

The spec names ADNI and OASIS as P5 evidence sources. ADNI requires a formal data-use
application (institutional affiliation, human-subject review) that cannot be completed by an
automated agent — **not pursued**. OASIS has two releases: OASIS-3 (also requires application) and
**OASIS-1** (`oasis-brains.org`), a cross-sectional release that has historically been mirrored
without a gated application process; a Kaggle republish (`jboysen/mri-and-alzheimers`, CC0-1.0)
was used directly. No model/checkpoint library was reused — this is a from-CSV tabular pass, same
posture as P1 arm B.

## 3. Published methods and data assumptions

- Marcus et al. (2007), *OASIS: Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and
  Demented Older Adults*, describes the cohort and the Clinical Dementia Rating (CDR) protocol.
- **CDR is a same-visit clinical rating, not a longitudinal outcome.** OASIS-1 cross-sectional has
  exactly one scan per subject — there is no index/outcome time pair to define a progression
  horizon from this release alone. The registered model is therefore explicitly a **same-visit
  dementia-association screening proxy**, not the spec's P5 task (MCI-to-AD progression risk). This
  is stated in the profile's `outcome_definition`, the model's `safety.limitations`, and the model
  card — it is not silently reframed as something it is not.
- Of 436 subjects, only 235 have a recorded CDR (201 were never clinically assessed) — dropping the
  unrated subjects is standard practice in OASIS-based classification papers, but it is also a
  documented selection-bias risk (rated subjects may be a systematically different population).
- **MMSE is not an independent risk factor.** The Mini-Mental State Exam is itself a cognitive
  screening instrument clinicians use as one input toward the CDR determination — its strong
  predictive value in this model reflects proxy-label correlation, not a novel biomarker finding.
  This is called out explicitly rather than presented as a discovery.

## 4. License and redistribution constraints

- **Kaggle `jboysen/mri-and-alzheimers`**: license **CC0-1.0** (public domain equivalent) —
  verified from the Kaggle dataset metadata at download time. Unlike the P1 stroke dataset, there is
  **no redistribution restriction**; the derived CSV could be committed if ever needed, though the
  repository's blanket "no data files in git" `.gitignore` policy still applies for consistency.

## 5. Reproducible comparison plan (preliminary — single split, not yet repeated/nested)

- **Data**: `backend/data/p5_alzheimers_clinical/oasis_cross_sectional_labeled.csv`, produced by
  `prepare.py` (drops unrated subjects, binarizes CDR>0 → `dementia`). 235 rows, 100 positive (42.6%).
- **Split**: stratified random (no group/time column — one scan per subject, no repeated-visit
  structure in this release).
- **Preprocessing boundary**: identical engine path to P1 — median imputation, standardisation,
  ANOVA `SelectKBest` to `n_qubits=4`, MinMax angle scaling, fitted on train only. `CDR` is declared
  a `leakage_column` so it is visible for provenance but never used as a feature (it determines the
  label by construction).
- **Initial single-split result** (full data, `threshold_policy=target_sensitivity` 0.8): `rbf_svc`
  balanced accuracy 0.856; `qsvc` (4-qubit ZZFeatureMap) balanced accuracy 0.709 — QSVC underperforms.
- **Architecture sweep** (2026-08-29, single split, ANOVA vs PCA × 4 vs 6 qubits):

  | reduction | n_qubits | qsvc | rbf_svc |
  |---|---|---|---|
  | anova | 4 | 0.709 | 0.856 |
  | anova | 6 | 0.758 | 0.788 |
  | pca | 4 | 0.559 | 0.738 |
  | pca | 6 | 0.430 | 0.738 |

  ANOVA beats PCA at every qubit count (same pattern as P1). 6 qubits narrows the QSVC gap versus 4
  (0.758 vs 0.788 classical, closest margin of any config tested across P1/P4/P5/P6) — this dataset's
  better class balance (42.6% positive) may make the extra qubits more useful than on the rarer-event
  conditions. Still not evidence of an advantage: this is a single split, not a paired comparison.
- **Nested paired evaluation, initial pass** (`study.run_nested_evaluation`, `outer_repeats=2`,
  production ANOVA/4-qubit config, run 2026-08-29): `qsvc` balanced accuracy 0.619 vs classical
  (HistGB) 0.863 — QSVC underperforms decisively once repeated folds average out the single-split's
  more favourable draw.
- **Nested paired evaluation, production rigor** (`outer_repeats=5`, `inner_repeats=2`, `repeats=10`,
  `bootstrap_samples=1000` — matching P1's declared production scale; data volume itself was already
  maximal at 235 rows, so this increases statistical rigor rather than sample size, run 2026-08-29):
  `qsvc` balanced accuracy **0.581** vs classical **0.823**. **QSVC underperforms decisively.**
  `real_gain_decision`: **failed**, consistent with every other condition tested.

## 6. Reuse decision per component (FR-037)

| Component | Decision | Evidence |
|---|---|---|
| Tabular engine | **reused, unmodified** | Identical to P1/P4 |
| CDR binarization / unrated-subject filtering | **newly authored** | `prepare.py` — label-definition adapter boundary, reuse-order rule 4 |
| Source dataset | **reused** | OASIS-1 cross-sectional via Kaggle, CC0-1.0 |

## 7. External-asset manifest (FR-035)

| Field | Value |
|---|---|
| Asset | `oasis_cross-sectional.csv` (436 rows) → `oasis_cross_sectional_labeled.csv` (235 rows, derived) |
| Source URL | `https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers` |
| Paper citation | Marcus et al., J. Cognitive Neuroscience, 2007 |
| License | CC0-1.0 — verified |
| Preprocessing assumptions | One scan per subject; CDR>0 = dementia; unrated subjects dropped |
| Local modifications | `prepare.py`: drop rows without CDR, add binary `dementia` column. No source values altered. |

## 8. Open risks carried forward

1. **This is not the spec's P5 task.** It is a same-visit association proxy, not a longitudinal
   MCI-to-AD progression risk. A real P5 progression model needs longitudinal data (e.g. OASIS-2/3
   or ADNI, both gated) — out of reach without a human completing a data-access application.
2. **Selection bias** from dropping the 201 unrated subjects is not corrected for.
3. **No repeated/nested real-gain evaluation yet** — `real_gain_decision` should stay `not_assessed`
   until that is run.
4. **Small cohort (235 rows)** — single-split metrics carry wide uncertainty.

## 9. Second-pass hybrid architecture optimization (2026-08-29) — QML promoted independent of classical

**Objective**: With the newly configurable VQC ansatz/optimizer parameters in the shared engine and the
revised baseline-viability gate (CI lower bound > 0.5, independent of classical's 0.823), conduct a
staged hyperparameter search to push QML performance toward baseline viability.

**Methodology**: Single-split screening (seed=7, full data) with corrected threshold policy
(`validation_size=0.2, threshold_policy="target_sensitivity", target_sensitivity=0.8`) followed by
production-scale validation. This threshold policy correction was critical: prior default settings
collapsed to degenerate 0.5 results (majority-class-only predictions); corrected settings restore
meaningful signal.

**Screening Results (single-split, seed=7)**:

QSVC parameter sweep (C regularization):

| C   | Balanced Accuracy |
|-----|-------------------|
| 0.1 | 0.728             |
| 0.5 | 0.728             |
| 1.0 | 0.709             |
| **5.0** | **0.745 (best)**      |
| 10  | 0.739             |

Best QSVC candidate: **C=5, n_qubits=4, reduction=anova, feature_map_reps=1, feature_map_entanglement=linear**
with single-split balanced accuracy **0.745**.

VQC screening (ansatz type, maxiter=25 for speed):
- real_amplitudes: 0.521
- efficient_su2: 0.577 (best)
- two_local: 0.509

**Best candidate selected for production validation**: QSVC with **C=5**
(0.745 > 0.577 single-split screening).

**Production-scale single-split result** (`max_train=0, max_test=0, seed=7`, `bootstrap_samples=1000`,
corrected threshold policy): balanced accuracy **0.7454**, 95% CI **[0.6177, 0.8714]** — independently
reproduced by the orchestrator with a fresh 200-sample bootstrap run: CI [0.6265, 0.8730], consistent.
**Taken alone this looked like a gate pass.**

**Correction — 10-seed repeated evaluation (orchestrator, 2026-08-29), same config**
(`run_repeated_experiment(repeats=10, ...)`, same C=5/n_qubits=4/anova/threshold-policy settings,
10 different stratified train/test splits): **mean balanced accuracy 0.564, std 0.082** (n=10). A
rough 95% CI on that mean (t-distribution, 9 df) is approximately **[0.51, 0.62]** — barely, not
clearly, above chance. **This directly contradicts the single-split bootstrap CI above.**

**Why the two disagree, and which one governs**: bootstrap CI resamples predictions on ONE fixed
test set — it measures sampling noise within that split, not variability across which patients land
in train vs. test. `seed=7` happens to be a favorable split for this config; the 10-seed spread shows
the true split-to-split variance is much larger than the bootstrap CI implied. This is the exact
"favorable single-split draw" trap the P1 record already documented once (a config that looked like
0.547 single-split was actually 0.491±0.066 — worse than classical — under repeated evaluation). The
**10-seed repeated result is the one that governs** promotion decisions, per that established lesson.

**Baseline viability gate result: NOT robustly cleared.** The point estimate (0.564) is nominally
above 0.5, but the repeated-evaluation CI is too close to 0.5 to call this a confident pass — this is
a marginal, not-clearly-above-chance result, not the "0.7454, decisively above chance" picture the
single-split number suggested. **This QML candidate remains `lifecycle: "experimental"`.** (An
earlier draft of this section, written before the 10-seed check completed, incorrectly reported this
as gate-cleared and promoted — corrected here before any registry change was made.)

**Comparison to classical baseline** (context only, not a gate): classical 0.823 vs. QSVC's honest
repeated-evaluation estimate 0.564±0.082 — a much larger, decisive gap than the single-split number
suggested.

**Final config for best candidate (QSVC)**:
```python
model_params = {
    "qsvc": {"C": 5}
}
# Plus hyperparameters:
# n_qubits=4, reduction="anova", feature_map_reps=1, feature_map_entanglement="linear"
```

**Note on threshold policy**: All results in this pass used `validation_size=0.2,
threshold_policy="target_sensitivity", target_sensitivity=0.8` to ensure proper balanced-accuracy
evaluation. Prior default settings produced artifactual 0.5 results and should not be used for
QML model assessment on this dataset.
