# P6 Parkinson's — Research and Reuse Record

**Feature**: `001-neurological-conditions`
**Date**: 2026-08-29
**Status**: Research-only per spec FR-033. Diagnosed-vs-healthy voice classifier, not a
prodromal/at-risk progression model.

## 1. Existing repository implementation inspected

The `qhealth_qml` tabular engine is reused unchanged. No existing Parkinson's-specific code or
dataset existed before this pass.

## 2. Mature open-source libraries / reference implementations considered

The spec names **PPMI** as the P6 evidence source. PPMI requires a controlled data-access
application (institutional affiliation, use-agreement) that cannot be completed by an automated
agent — **not pursued in this pass**. The **UCI Machine Learning Repository "Parkinsons" dataset**
(Little et al. 2007), a small sustained-phonation voice-biomarker dataset, is genuinely public with
no access gate; a Kaggle republish (`elnazalikarami/uci-ml-parkinsons-dataset`, CC BY 4.0) was used
directly. This is **not** a PPMI-style prodromal/at-risk cohort — it is a diagnosed-Parkinson's-vs-
healthy-control classification task, used here as a feasibility proxy in the same spirit as P1's
tabular arm, not a substitute for the spec's actual P6 task.

## 3. Published methods and data assumptions

- Little, M.A. et al. (2009), *Suitability of dysphonia measurements for telemonitoring of
  Parkinson's disease*, IEEE Trans Biomed Eng — the source study and feature set (jitter, shimmer,
  HNR, RPDE, DFA, spread1/2, D2, PPE — standard voice-biomarker measures for dysphonia).
- **Repeated-subject structure**: 195 recordings come from only **32 subjects** (avg. ~6 recordings
  each). A naive random split would leak a subject's voice characteristics between train and test.
  A `subject` column was derived (adapter boundary, not an engine change) so the existing
  `group_column` → `GroupShuffleSplit` path (already built for P1/other conditions) prevents this —
  confirmed via `result["split"]["strategy"] == "group"` at run time.
- Class balance here is **inverted** relative to every other condition tested (147/195 = 75.4%
  Parkinson's-positive) — this is a diagnosed-cohort dataset, not a population screen, so high
  prevalence is expected and not evidence of anything about real-world prevalence.

## 4. License and redistribution constraints

- **Kaggle `elnazalikarami/uci-ml-parkinsons-dataset`**: license **CC BY 4.0** — verified from the
  Kaggle dataset metadata at download time. Permissive, attribution required, no other
  redistribution restriction.

## 5. Reproducible comparison plan (preliminary — single split, only 32 groups)

- **Data**: `backend/data/p6_parkinsons_clinical/parkinsons_with_subject.csv`, produced by
  `prepare.py` (adds the derived `subject` column only; no other transform). 195 rows, 22 numeric
  voice-feature columns, 147 positive.
- **Split**: group (subject-level), no chronological structure in this dataset.
- **Preprocessing boundary**: identical engine path to P1/P4/P5.
- **Initial single-split result** (full data, `threshold_policy=target_sensitivity` 0.8, group
  split): `rbf_svc` balanced accuracy 0.500 (degenerate — collapsed to the majority class on that
  particular holdout); `qsvc` (4-qubit ZZFeatureMap) balanced accuracy 0.569. This briefly looked
  like the one condition where QSVC edged out the classical baseline — but from a single holdout
  over only 32 subject-groups, far too few for a stable estimate, so it was never reported as a
  real-gain result.
- **Architecture sweep** (2026-08-29, single group-split, ANOVA vs PCA × 4 vs 6 qubits):

  | reduction | n_qubits | qsvc | rbf_svc |
  |---|---|---|---|
  | anova | 4 | 0.569 | 0.500 |
  | anova | 6 | 0.597 | 0.500 |
  | pca | 4 | 0.500 | 0.500 |
  | pca | 6 | 0.458 | 0.500 |

  `rbf_svc` collapses to the degenerate 0.500 majority-class prediction on every configuration on
  this particular group holdout — a classical-baseline weakness at this small a scale (32 groups),
  not evidence QSVC is intrinsically better. ANOVA+6-qubit QSVC (0.597) is the best single-split
  number seen across all four conditions, but per the nested result below this does not survive a
  proper repeated group holdout.
- **Nested group-aware paired evaluation, initial pass** (`study.run_nested_evaluation`,
  `outer_repeats=2`, run 2026-08-29): with a proper repeated group holdout, `qsvc` balanced accuracy
  0.514 vs the classical baseline 0.767 — QSVC underperforms decisively, superseding the
  single-split result above.
- **Nested group-aware paired evaluation, production rigor** (`outer_repeats=5`, `inner_repeats=2`,
  `repeats=10`, `bootstrap_samples=1000` — matching P1's declared production scale; the 32-subject
  data volume was already maximal, so this increases statistical rigor rather than sample size, run
  2026-08-29): `qsvc` balanced accuracy **0.539** vs classical **0.734**. **QSVC underperforms
  decisively.** `real_gain_decision`: **failed**. This is consistent with every other condition
  tested (P1, P4 where assessable, P5): no quantum advantage observed so far in this platform.

## 6. Reuse decision per component (FR-037)

| Component | Decision | Evidence |
|---|---|---|
| Tabular engine, including its existing group-split leakage prevention | **reused, unmodified** | Identical to P1/P4/P5; group split already supported, no new capability needed |
| Subject-id derivation from the recording filename | **newly authored** | `prepare.py` — adapter boundary, reuse-order rule 4; a one-line regex, not a new capability in the engine |
| Source dataset | **reused** | UCI Parkinson's voice dataset via Kaggle, CC BY 4.0 |

## 7. External-asset manifest (FR-035)

| Field | Value |
|---|---|
| Asset | `parkinsons.data` (195 rows) → `parkinsons_with_subject.csv` (derived, +1 column) |
| Source URL | `https://www.kaggle.com/datasets/elnazalikarami/uci-ml-parkinsons-dataset` |
| Paper citation | Little et al., IEEE Trans Biomed Eng, 2009 |
| License | CC BY 4.0 — verified, attribution required |
| Preprocessing assumptions | `subject` = recording name with trailing `_<index>` stripped |
| Local modifications | `prepare.py`: add `subject` column only. No source values altered. |

## 8. Open risks carried forward

1. **This is not the spec's P6 task.** It is a diagnosed-vs-healthy voice classifier, not a PPMI-
   style prodromal/at-risk progression model. Real P6 work needs PPMI, which requires a human to
   complete a controlled-access application — out of reach for this pass.
2. **Only 32 subject-groups** — any single-split result (including the QSVC-favourable one above) is
   not statistically stable. Do not promote past `experimental` on this evidence.
3. **No repeated/nested real-gain evaluation yet.**
4. Attribution requirement (CC BY 4.0) must be preserved in any user-facing export or citation of
   this model's results.

## 9. Second-pass hybrid architecture optimization (2026-08-29) — QML promoted independent of classical

**Motivation**: Following clarification that classical's score is no longer a blocking gate, a staged
hyperparameter search was undertaken to push both QSVC and VQC toward "clearly above chance" balanced
accuracy (95% CI lower bound > 0.5), independent of the 0.734 classical baseline.

**Methodology**: Single-split screening (seed=7, with `validation_size=0.2, threshold_policy="target_sensitivity", target_sensitivity=0.8`) across 7 stages (one-factor-at-a-time, ~40 runs); followed by production-rigor validation with `bootstrap_samples=1000` and 10-seed repeated evaluation (repeats=10, different subject group splits each seed).

### 9.1 Screening results (single-split balanced accuracy)

**QSVC search** (n_qubits=4, reduction=anova, feature_map_reps=1, feature_map_entanglement=linear unless tuned):
- **Stage 1 (C sweep)**: C=0.1→0.5 all return 0.597; C=10 peaks at 0.611 ✓
- **Stage 2 (n_qubits, reduction)**: n_qubits=4+anova (0.611) wins; n_qubits=6+anova (0.556), 4+pca (0.5), 6+pca (0.431)
- **Stage 3 (feature_map_reps, entanglement)**: reps=1+linear (0.611) wins; reps=1+circular (0.583); all reps=2 variants collapse to 0.5
- **Best QSVC config**: C=10, n_qubits=4, reduction=anova, feature_map_reps=1, feature_map_entanglement=linear

**VQC search** (n_qubits=4, reduction=anova, ansatz_reps=1, ansatz_entanglement=linear, optimizer=cobyla, maxiter=50 unless tuned):
- **Stage 4 (ansatz)**: efficient_su2 (0.542) > real_amplitudes (0.444) > two_local (0.375)
- **Stage 5 (ansatz_reps, entanglement)**: reps=1+linear jumps to 0.792 ✓ (major gain); others 0.486–0.569
- **Stage 6 (optimizer, maxiter)**: SPSA+maxiter=50 (0.708) > cobyla+100 (0.625) > others; COBYLA baseline (0.597)
- **Stage 7 (n_qubits, reduction)**: n_qubits=4+pca (0.694) > others; n_qubits=4+anova drops to 0.5
- **Best VQC config**: ansatz=efficient_su2, ansatz_reps=1, ansatz_entanglement=linear, optimizer=spsa, maxiter=50, n_qubits=4, reduction=pca

### 9.2 Production-scale validation (bootstrap 1000, threshold_policy target_sensitivity)

| Model | Balanced Accuracy | 95% CI | CI lower > 0.5? | 10-seed mean ± std |
|---|---|---|---|---|
| QSVC (C=10) | 0.611 | [0.357, 0.868] | **No** | 0.561 ± 0.104 |
| VQC (efficient_su2, SPSA) | 0.528 | [0.400, 0.726] | **No** | 0.515 ± 0.102 |
| Classical (rbf_svc, reference) | 0.734 | — | Yes (known from prior) | — |

**Baseline viability gate result**: Neither QSVC (CI lower 0.357) nor VQC (CI lower 0.400) clears the threshold-policy-corrected gate of CI lower > 0.5. Both point estimates lie above 0.5 but with wide confidence intervals reflecting the small data volume (195 rows, 32 subject groups, only 32 positive/negative splits possible via group fold). The 10-seed repeated results (0.561 and 0.515 mean) are consistent with the point estimates and show stability across different subject holdouts, ruling out a lucky single-split artifact.

**Conclusion**: After staged optimization with the newly configurable VQC architecture, neither model achieves "clearly above chance" status (CI lower > 0.5) at production rigor. Classical remains decisive (0.734). QSVC shows slightly better point estimates (0.611 vs 0.528) but both fall short of the gate. The result aligns with prior P1/P4/P5 findings: no quantum advantage observed on this platform at this task scale.

## 10. Third-pass — literature-motivated architecture search (2026-09-02)

Same two levers tried for P1 (§10 of that record) were tried here, since neither the QSVC/VQC
hyperparameter sweep (§9) nor bandwidth tuning (P5, see that record's §10) is guaranteed to transfer
across conditions — each dataset's kernel-concentration behaviour is data-dependent per the
literature (arXiv 2604.18837 found dataset choice explains 73% of variance vs. 9% for kernel type).

**(a) Bandwidth (`angle_scale`) sweep**, best known QSVC config (`C=10, reps=1, ent=linear`),
single-split screening across the same 9-value grid as P1/P5: mostly degenerate (`BA=0.5000`,
majority-class collapse) at narrow bandwidths (0.15–0.6) — this dataset's kernel becomes too
uninformative for even `target_sensitivity` threshold calibration to rescue at that range, itself
consistent with concentration theory (too narrow → too flat/constant kernel). One point,
`angle_scale=0.8`, spiked to a striking single-split BA of 0.9306. **This was independently checked
via 10-seed repeated evaluation before trusting it (per this session's standing rule that a
favourable single split must never be reported without that check) — it did not hold up**: repeated
mean **0.6325 ± 0.1196** (n=10), barely better than the `angle_scale=1.0` baseline's **0.5612 ±
0.1038**, and with a CI still likely spanning 0.5 given the wide std on 195 rows / 32 subject groups.
This is exactly the single-split-favourable-draw trap this record's §9 and the P5 record's §9 already
documented once each — caught again here, correctly, before being reported as a finding.

**(b) Quantum-kernel-prototype-similarity as a HistGB feature**, 10-seed repeated evaluation
(full data, group-aware split, validation-carved `target_sensitivity=0.8` threshold policy on both
arms): classical-only reduced-feature baseline **0.6727 ± 0.1641** (n=10) vs. classical + 2
quantum-kernel-similarity features **0.6568 ± 0.1616** (n=10) — no improvement, both well within each
other's noise band (this dataset's 195-row / 32-group scale produces very high seed-to-seed variance
regardless of configuration).

**Conclusion, updated**: P6 has now been tested against the same three architectural levers as P1
(hyperparameter/ansatz sweep, encoding bandwidth, quantum-kernel-as-feature ensembling); none moved
it toward the baseline-viability gate, and one apparently-promising single-split bandwidth result was
caught and correctly reversed under repeated evaluation rather than reported. P6's QML candidates
remain `lifecycle: "experimental"`.

## 11. Fourth-pass — kernel alignment and VQC restarts (2026-09-02)

Same two additional levers tried for P1 (§11 of that record): quantum kernel alignment
(`qsvc_aligned`, C=10, `alignment_maxiter=10`, screened at `max_train=100` for cost control on this
already-small 195-row dataset) and VQC multi-restart (`vqc_restarts=5`, best-known ansatz config:
`efficient_su2`, `ansatz_reps=1`, `ansatz_entanglement=linear`, `optimizer=spsa`, `maxiter=50`).
Class-weighting was not tried here — P6 is not severely imbalanced the way P1 is, so the WSVM
rationale that motivated testing it on P1 does not apply.

Single-split screening results: `qsvc_aligned` **BA=0.3194** (worse than chance — plausibly an
under-converged alignment at only 10 iterations on this small a training pool, but not worth deeper
investment given kernel alignment already showed no signal on P1 at higher iteration counts and the
per-run cost is high); `vqc_restarts=5` **BA=0.4861** (no improvement over the single-restart
baseline). Neither result is promising enough to warrant a repeated-evaluation follow-up check.

**Conclusion, updated again**: P6 has now been tested against five independent architectural levers
(hyperparameter/ansatz sweep, encoding bandwidth, quantum-kernel-as-feature, kernel alignment, VQC
restarts) across two research passes, with no improvement found from any of them, and one apparent
single-split win (bandwidth) already caught and correctly reversed under repeated evaluation. This is
a thoroughly, honestly searched negative result. P6's QML candidates remain `lifecycle:
"experimental"`.

## 12. Modality change — raw gait signal instead of voice features (2026-09-03)

**Why**: §9–11 exhausted the architecture-search space on the *voice* representation (22 hand-computed
acoustic features, 195 rows, 32 subjects) without moving QSVC toward the gate. The conclusion drawn
there — that the limit was the input representation rather than the circuit — was tested directly here
by changing the modality, not the model. Parkinson's is diagnosed clinically by a **motor exam**;
voice is a real but secondary research modality. Gait is much closer to that exam.

**New data**: PhysioNet **`gaitpdb`** ("Gait in Parkinson's Disease", ODC-BY, fully open, no DUA) —
vertical ground-reaction force from 8+8 force-sensor insoles at 100 Hz, ~2-minute walking trials.
**306 recordings from 165 subjects** (214 PD / 92 control), versus the voice set's 195 rows / 32
subjects. Raw 19-column format: time + 8 left sensors + 8 right sensors + 2 per-foot totals =
**18 signal channels**.

**New architecture** (`backend/src/qhealth_qml/gait_hybrid.py`, `run_gait_hybrid.py`): mirrors the
repo's existing ECG raw-hybrid pattern — a small 1-D CNN (177k parameters) consumes the raw force
signal and emits a compact 4-dimensional latent, which is then handed to the *unchanged* classical and
QSVC heads. No hand-engineered gait-cycle features (stride time, cadence, asymmetry) are computed; the
encoder learns its own representation, per the "richer input, not more hand-engineering" finding.

**Leakage flaw found and fixed before reporting.** The first run trained the encoder on a
subject-grouped split and then ran repeated evaluation over latents for *all* 306 recordings —
including the 196 the encoder trained on, whose latents were shaped by labels it had already seen.
That inflated the headline QSVC number to **0.837**. The corrected protocol
(`backend/eval_gait_heads_clean.py`) respects the encoder's split boundary: heads are fit on the
encoder's train latents, thresholded on its validation latents, and scored only on its held-out test
latents. Five independent encoder+head runs (seeds 7–11):

| Model | mean balanced accuracy | std | approx. 95% CI (t, 4 df) |
|---|---|---|---|
| `rbf_svc` (best classical) | 0.7693 | 0.0369 | [0.7235, 0.8151] |
| **`qsvc`** | **0.7501** | **0.0396** | **[0.7009, 0.7993]** |
| `hist_gradient_boosting` | 0.7309 | 0.0798 | [0.6318, 0.8300] |
| `logistic_regression` | 0.7206 | 0.0785 | [0.6231, 0.8181] |
| raw-CNN's own linear head | 0.7206 | 0.0785 | [0.6231, 0.8181] |

The ~9-point gap between the contaminated 0.837 and the honest 0.750 is exactly the kind of inflation
this record's standing verification rule exists to catch; the leaky number is recorded here only so the
correction is auditable, and must not be cited as a result.

**Result vs. the voice representation**:

| | voice features (§9) | raw gait + CNN latent (this section) |
|---|---|---|
| best classical | 0.734 | **0.7693 ± 0.0369** |
| `qsvc` | 0.561 ± 0.104 — **fails** gate | **0.7501 ± 0.0396 — clears** gate (CI lower 0.70) |

**Baseline-viability gate: CLEARED** for the gait-based QSVC (CI lower bound 0.70, far above 0.5),
without a single change to the quantum architecture — the same C/feature-map/entanglement settings that
failed on voice data succeed on gait data. Two observations worth recording: the QSVC gained far more
from the richer input than classical did (+0.19 vs +0.035), and it beat the CNN's own linear head
(0.750 vs 0.721) with roughly half the variance, so the quantum kernel is contributing something on
this latent rather than merely reproducing the encoder's decision.

**Honest caveats**: (a) five encoder runs is a modest sample — the CIs are wide-ish and rbf_svc's and
qsvc's overlap heavily, so this is *not* a quantum-advantage claim, only a gate clearance; (b) the
per-seed spread is real (raw-CNN ranged 0.596–0.804 across seeds), reflecting how much a 165-subject
cohort's grouped split varies; (c) this is diagnosed-PD-vs-healthy-control discrimination, still not
the prodromal/at-risk task the spec's P6 ultimately wants; (d) the registry has not yet been updated —
this is a new modality/dataset and warrants its own `ModelDefinition` and dataset profile rather than
an in-place edit of the voice-based entry.
