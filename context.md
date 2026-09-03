# Heart ECG Model Context

Last updated: 2026-09-02 (Asia/Kolkata)

## Objective

Build a research-use hybrid quantum/classical platform for early disease-pattern
detection from raw ECG waveforms. The output is detection, not future risk:

- detected / not detected
- calibrated detection score
- validation-selected operating threshold
- normalized detection confidence
- optional review/abstention state near the threshold

Detection confidence is normalized distance from the operating threshold. It is
not clinical certainty, a confidence interval, or a calibrated probability of
future disease.

## Dataset and labels

Primary dataset: PTB-XL 1.0.3, downloaded as raw WFDB .hea/.dat records. The
project uses human-validated studies, 12 standard leads, and official
stratified folds. A raw record is approximately a 10-second ECG study, usually
sampled at 500 Hz.

Supported endpoints:

1. mi_pattern_ecg: positive when PTB-XL diagnostic classes contain MI and are
   not NORM; negative when the diagnostic class is NORM only.
2. clinically_abnormal_ecg: positive for any non-NORM diagnostic class;
   negative for NORM only.

Split contract:

- folds 1–8: encoder/head fitting
- fold 9: checkpoint and threshold selection
- fold 10: locked final test
- patient groups must not overlap between partitions

## Current model architecture

raw 12-lead waveform
  -> resample to 2,500 samples + per-lead median/std normalization
  -> GPU 1-D CNN: 12->32->64->128, adaptive pool, 128->64->4
  -> 4-dimensional latent vector
  -> Logistic/RBF-SVC/HistGB and QSVC/PegasosQSVC/VQC heads
  -> detection decision + confidence + optional review state

The CNN handles the high-dimensional waveform representation. The QML head
operates on the compact latent vector because direct encoding of thousands of
ECG samples is not practical on near-term quantum hardware.

## Important files

- backend/src/qhealth_qml/raw_hybrid.py: raw loading, normalization, CNN,
  latent encoding, combined artifact loading and inference.
- backend/run_raw_hybrid.py: end-to-end raw training, head sweep, full
  fold-10 QML evaluation, report and combined artifact writer.
- backend/src/qhealth_qml/experiment.py: preprocessing, Qiskit models,
  metrics, thresholding, calibration, artifacts, and explicit validation-index
  support.
- backend/src/qhealth_qml/dashboard.py: raw waveform API at POST /api/predict-ecg.
- backend/predict_raw_ecg.py: CLI inference for feature or raw-hybrid artifacts.
- README.md: delivery table, architecture and run commands.

## Existing measured baseline

These are deterministic-feature results, not raw-CNN results:

- MI endpoint, Logistic Regression: fold-10 balanced accuracy 0.8572,
  AUROC 0.9346, sensitivity 0.8870, specificity 0.8274.
- Abnormal ECG endpoint, RBF-SVC: fold-10 balanced accuracy 0.8250,
  AUROC 0.9169, sensitivity 0.8767, specificity 0.7733.
- Earlier compact QML candidates did not beat those references. Do not claim
  quantum advantage unless the new raw-hybrid held-out results demonstrate it.

## Hardware and storage

- GPU: NVIDIA GeForce RTX 2050, 4 GiB.
- Driver: 595.84; nvidia-smi works; CUDA driver reports 13.2.
- Root filesystem is nearly full; large runtimes and datasets belong on SSD.
- External SSD: /dev/sda2, NTFS, mounted at /mnt/ssd, approximately 226 GiB
  free after reconnection on 2026-09-02.
- A previous CUDA PyTorch install stalled when the USB SSD temporarily returned
  Not Ready read/write timeouts. The installer was terminated. No final CUDA
  PyTorch runtime is assumed installed until the smoke test passes.

## Next execution sequence

1. Install CUDA-enabled PyTorch into an SSD-backed target directory.
2. Verify torch.cuda.is_available(), device name, and a CUDA tensor operation.
3. Run a short raw-CNN smoke pass.
4. Run the full raw hybrid experiment with PTB-XL records.
5. Review validation-selected head and fold-10 sensitivity, specificity,
   AUROC, AUPRC, Brier score, confidence and coverage.
6. Only then call the result an optimized/final research model.

Suggested commands:

PIP_CACHE_DIR=/mnt/ssd/Neutral/pip-cache \
TMPDIR=/mnt/ssd/Neutral/tmp \
backend/.venv/bin/python -m pip install \
  --target /mnt/ssd/Neutral/python-site \
  'torch==2.12.1' \
  --index-url https://download.pytorch.org/whl/cu132

PYTHONPATH=/mnt/ssd/Neutral/python-site:backend/src \
backend/.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

PYTHONPATH=/mnt/ssd/Neutral/python-site:backend/src \
backend/.venv/bin/python backend/run_raw_hybrid.py \
  --data-root /mnt/ssd/Neutral/ptb-xl/1.0.3 \
  --target mi --max-records 6000 --download --device auto

## Validation completed

- Python compile checks passed.
- git diff --check passed.
- Backend tests excluding the known Playwright-dependent dashboard test:
  31 passed.
- Frontend npm run build passed.
- Dashboard and raw inference CLI help/import checks passed.

## Clinical limitations

PTB-XL is a benchmark dataset, not a hospital deployment cohort. Its labels are
dataset diagnostic annotations, not prospective early-stage outcomes. Clinical
deployment requires external validation, calibration review, quality-control
gates, prospective testing, and applicable regulatory review.

## Session update (2026-09-02, continued) — research pass + engine additions, raw-hybrid still not run

Picked up from this file and `heart_models_transcript.md` at the start of the session. Status
check on the live machine found the documented "next execution sequence" had not actually
progressed since it was written: `torch` was not installed in `backend/.venv` at all, and a prior
CUDA-PyTorch install attempt had died silently (empty target dir, no running process). **The
raw-hybrid CNN->latent->QML path has still never been run end-to-end — zero raw-hybrid fold-10
numbers exist.** The only real measured results remain the deterministic-feature baselines in
"Existing measured baseline" above.

### Research conducted (5 parallel deep-dive passes, arXiv/Qiskit-docs/GitHub)

- **No published QML benchmark exists on PTB-XL, or on MI-detection from 12-lead ECG, anywhere.**
  Whatever raw-hybrid result we eventually get will be a first data point on this exact dataset,
  not a number to replicate.
- **Terminal quantum head vs. mid-network placement.** A realistic hybrid-QNN ablation on
  ECG/MedMNIST-style 1D signals found a *terminal* quantum head (CNN -> frozen latent -> separate
  QSVC/VQC fit, our current architecture) adds no substantial improvement over classical-only.
  Nearly every ECG/biomedical paper that reports a genuine gain instead places the quantum layer
  *mid-network* ("dressed quantum circuit," Mari et al. arXiv 1912.08278), trained end-to-end via
  backprop through it. Qiskit ML's `EstimatorQNN`/`SamplerQNN` + `TorchConnector` can do this with
  our existing stack (no PennyLane needed) — flagged as the most promising *architectural* next
  step once a terminal-head baseline exists, not yet implemented.
- **Quantum kernel concentration** (Thanasilp et al., Nat. Commun. 2024, arXiv 2208.11060): `full`
  entanglement is most exposed to kernel-value collapse; `linear`/`circular` are safer — matches
  this platform's own empirical finding (a sibling seizure-EEG condition: BA 0.500 with `full` vs
  0.8125 with `circular`, same positives). Our defaults were already `linear`; no default changed.
- **VQC single-split instability is initialization-range sensitivity, not barren plateaus.**
  4-6 qubits / 1-2 reps is well inside the field's "trainable" regime — barren plateaus are not the
  explanation for why the same config produces wildly different single-split results across seeds
  on this platform. arXiv 2412.06462 identifies the *initial-point sampling range* (Qiskit's
  default `[0, 2*pi)`) as the dominant lever; narrowing it, plus multi-restart COBYLA, is a cheap,
  low-effort fix — implemented below.
- **Quantum kernel alignment is a real, maintained Qiskit API**, not a paper-only idea:
  `qiskit_machine_learning.kernels.algorithms.QuantumKernelTrainer` +
  `TrainableFidelityQuantumKernel` trains the feature map itself against SVC-margin loss before
  QSVC uses it — directly counters the concentration problem above instead of hand-picking
  entanglement. Implemented below as an opt-in model. (The neuro-conditions session's own testing
  of this same API on P1 stroke found it ~2000x more expensive than plain QSVC and ultimately
  ineffective there — cost/benefit may differ on ECG-scale data, worth checking early.)
- **Data re-uploading circuits deprioritized.** Two independent clinical-tabular-data sources
  (a Clinical Chemistry 2025 paper, our own MedMNIST-ablation finding) show re-uploading is
  competitive at best, not clearly better, and degrades past ~4-8 input features (our regime).
  Qiskit ML has no maintained re-uploading implementation — a community PR was closed unmerged.
- **Explainability: don't explain the quantum step.** A 4-8 dim quantum latent space has no
  clinical meaning. Every comparable hybrid-CNN-ECG paper attributes back to the *original
  waveform* instead (Grad-CAM/gradient-x-input saliency on the CNN encoder — which lead, which
  time segment drove the decision). This is the recommended way to satisfy the delivery table's
  explainability requirement, implementable with plain PyTorch hooks once the CNN trains. SHAP on
  the 4-dim latent vector (`shap.KernelExplainer` treating QSVC/VQC as black-box `predict_proba`)
  is a secondary, internal diagnostic only, not the clinician-facing explanation.
- **Expressibility/effective-dimension metrics** (Sim, Johnson & Aspuru-Guzik, arXiv 1905.10876;
  `triple_e` package) can score ansatz x entanglement combinations *before* training, turning
  one-factor-at-a-time grid search into a principled pre-filter. Not yet wired in — noted as a
  future step if/when a broader VQC sweep is run on real raw-hybrid data.

### Engine capabilities added this session (`backend/src/qhealth_qml/experiment.py`)

- **VQC initial-point narrowing (default behavior change, all VQC builds).** Every VQC now gets an
  explicit `initial_point` drawn from `[-pi/4, pi/4]` (configurable via
  `model_params["vqc"]["initial_point_scale"]`) instead of Qiskit's default `[0, 2*pi)` draw,
  targeting the instability finding above.
- **`vqc_restarts`** (opt-in, `model_params["vqc"]["vqc_restarts"]`, default 1 = old behavior):
  fits N independently-seeded VQC candidates and keeps the best training-balanced-accuracy one,
  via new `_best_of_vqc_restarts()`.
- **`qsvc_aligned`** (new model name alongside `qsvc`/`pegasos_qsvc`/`vqc`): QSVC whose feature map
  is trained via `QuantumKernelTrainer`/SVC-loss kernel alignment before QSVC uses it, via new
  `_build_qsvc_aligned()`. Params: `C`, `alignment_optimizer` (cobyla/spsa), `alignment_maxiter`.
- Validated: full backend pytest suite still 31 passed (no regressions from the VQC default-behavior
  change); both new paths smoke-tested end-to-end on the sklearn breast-cancer benchmark (ran
  without error — numbers from that run are not a performance claim, just a plumbing check, given
  the deliberately tiny `maxiter`/sample sizes used for speed).
- Note: `backend/src/qhealth_qml/experiment.py` is being edited concurrently by the separate
  neuro-conditions session (see below the `---` divider) — its unrelated additions (`angle_scale`,
  `class_weight="balanced"`) are visible in the same working tree and are not part of this
  session's heart-ECG work.

### New blocker found this session: the external SSD is flaky again

Mid-session, `/mnt/ssd` started throwing `Input/output error` on plain directory listings
(`ls /mnt/ssd/Neutral` failed), the same failure signature as the earlier documented USB/NTFS
"Not Ready" incident, and a CUDA-PyTorch install attempt targeting the SSD hung silently (process
alive, zero CPU, blocked in `hrtimer_nanosleep`, its own log file inode showed `(deleted)`). Killing
the process and waiting did not restore access. The previously-cached PTB-XL data under
`/mnt/ssd/Neutral/ptb-xl/1.0.3/records500` was unreachable at last check (I/O error), not confirmed
lost, just currently inaccessible — do not assume it needs re-downloading without checking access
again first.

**Pivot in progress**: the user has a standing GCP GPU VM available
(`gcloud compute ssh cent-data-science-gpu --zone=us-central1-b`, see
`reference_gcp_gpu_vm.md` memory) as the fallback for the CUDA install + raw-hybrid run instead of
continuing to fight the local SSD. As of this update, `gcloud auth list` shows the local gcloud
credentials need an interactive `gcloud auth login` before the VM is reachable from this shell —
not yet done.

### Updated next steps (supersedes the SSD-only "Next execution sequence" above)

1. Get GCP VM access working (`gcloud auth login`, interactive, needs the user), or confirm the
   local SSD has genuinely recovered before retrying it.
2. On whichever compute lands: install CUDA PyTorch, verify `torch.cuda.is_available()`, run
   `backend/run_raw_hybrid.py` for real against PTB-XL (`--target mi`, `--max-records 6000`) — this
   produces the project's first-ever raw-hybrid fold-10 numbers.
3. Once that baseline exists: re-run the `qsvc_aligned` and `vqc_restarts` additions above against
   the *actual* raw-hybrid latent vectors (not the breast-cancer smoke test) and compare with
   bootstrap CI, per this platform's own established discipline of never trusting a single-split
   result.
4. If mid-network placement (dressed-quantum-circuit via `TorchConnector`) is pursued, treat it as
   a second-generation architecture experiment after step 2's terminal-head baseline, not a
   replacement for it — we want an honest before/after comparison.
5. Grad-CAM/saliency on the CNN encoder for waveform-level explainability, once the CNN trains.

## Session update (2026-09-03) — modality research + label/feature tuning applied

Directive for this pass: use whatever input data forms are most useful for detecting the
cardiovascular condition; research it, then tune the models accordingly. Two research passes plus a
round of tuning that required **no GPU and no SSD** — all validated against real PTB-XL metadata
pulled fresh from PhysioNet (the 6.6 MB `ptbxl_database.csv`, not the waveforms, which remain
stranded on the failed SSD).

### Research findings that changed what we do

- **Our classical baseline is not a weak target.** Published PTB-XL SOTA (JEPA-pretrained ViT-XS)
  reports AUC 0.940; our existing hand-crafted-feature baseline is AUROC **0.9346** with no deep
  learning and no pretraining. The raw-CNN+QML path is therefore competing against a near-SOTA
  number, not a strawman. State this whenever the hybrid result is eventually reported.
- **4-dim latent is near the documented failure edge.** Hybrid CNN+quantum literature puts 4-5
  dimensions where hybrids lose badly to classical, and **6-10** where they reach parity. `--latent-dim`
  was already configurable, so sweeping {4, 6, 8} is a config-only experiment, not new engineering.
- **Multimodal fusion beats ECG-alone** (ECG+troponin reaching AUROC 0.921 in one published model),
  but PTB-XL carries no lab values — troponin fusion would need a whole second dataset. Demographic
  fusion, by contrast, is free and already sitting in the metadata.
- **Coronary angiography is a viable new modality, unlike ICH/stroke-imaging.** `CADICA`
  (data.mendeley.com/datasets/p9bpx9ctcv, CC BY 4.0 confirmed independently in the Mendeley API
  response headers, 3.08 GB single zip, no DUA, no login) is genuinely accessible: 42 patients,
  invasive coronary angiography with lesion bounding boxes and a video-level lesion/non-lesion split.
  ARCADE was **rejected** — its Zenodo deposit is restricted behind challenge registration and
  access approval, the same DUA-equivalent pattern already rejected for BraTS/CT-ICH in P2/P3.
  Critically, angiography *already has a published hybrid-quantum precedent with public code*
  (Ovalle-Magallanes et al., "Hybrid classical-quantum CNN for stenosis detection in X-ray coronary
  angiography," github.com/eovallemagallanes/H-CQN-Stenosis-Detection — reporting +9% accuracy,
  +20% recall, +11% F1 over a classical-only CNN), which is more than ECG has. Not yet downloaded:
  3 GB onto a 9 GB-free root disk while the SSD is dead is not worth it until storage is resolved.

### Tuning applied (all CPU-verified against real PTB-XL metadata)

1. **`age=300` de-identification bug — fixed, affects the existing baseline.** PTB-XL stores every
   age above 89 as the literal `300` (293 records). Left raw, the MinMax angle scaler squeezed every
   real age into **29.2% of the quantum encoding angle range**; clamping at 89 restores the full
   **100%** — a ~3.4x expansion of usable encoding resolution for that feature. New `ptbxl_age()` in
   `ecg.py` is now used by *both* the deterministic feature path and the raw-hybrid path.
   **Consequence: the 0.8572 BA / 0.9346 AUROC MI baseline was measured with this bug present and
   should be re-measured.** Expect a small change, direction unknown until re-run.
2. **Demographic fusion into the raw-hybrid latent** (`--fuse-demographics`, new). The raw-hybrid
   path previously discarded age/sex entirely, even though the deterministic path used them. Now
   `RawECGDataset` carries a `demographics` array and the flag appends it to the CNN latent before
   the head. Deliberately **only `age` and `sex_male`** — height and weight are present for just
   32% / 43% of records, so fusing them would spend one qubit each on a mostly-median-imputed
   constant. With `latent_dim=4` this makes the head 6-wide, landing in the 6-10 sweet spot above.
   The head's qubit count now follows the actual fused width (`head_qubits`) instead of being
   hardcoded to `latent_dim`, which would otherwise have let PCA squeeze the demographics straight
   back out. The combined artifact records `fused_demographics` so inference rebuilds the head input
   exactly as training did.
3. **Acute-MI-stage filtering** (`--acute-mi-only`, new, on both runners) — implemented, correct,
   tested, and **deliberately not the default**, because measuring it against real metadata showed
   why: restricting MI positives to `infarction_stadium1` Stage I/I-II collapses the cohort from
   **3,476 positives to 98**, leaving only **14 positives in the locked fold-10 test set**. That is
   the same statistical-fragility trap that produced P4 seizure's unstable 8-positive result and
   P6's 0.93-to-0.63 collapse. Keep it as a sensitivity analysis; never report it as the headline
   endpoint. (Two minutes of CPU work; would have been hours of wasted GPU time to learn later.)
4. `select_ptbxl_rows` / `load_ptbxl_ecg_dataset` gained `mi_stages`, and the cached-dataset loader
   now **refuses a cache built under a different stage filter** rather than silently returning the
   wrong cohort.

Validation: 35 backend tests pass (33 + 2 new for stage filtering and age clamping). Real-metadata
value distributions confirmed the `"Stadium I"` / `"Stadium II-III"` string format the parser assumes.

### Still blocked / next

- SSD remains in `Input/output error`; GCP VM still needs an interactive `gcloud auth login`.
  Everything above was done without either, but **running** anything — the raw-hybrid baseline, the
  latent-dim sweep, the re-measured post-age-fix baseline — still needs one of them.
- Priority order once compute exists: (a) re-measure the deterministic baseline post-age-fix;
  (b) first-ever raw-hybrid run; (c) `--latent-dim` {4,6,8} sweep; (d) `--fuse-demographics` on/off
  ablation; (e) `qsvc_aligned` / `vqc_restarts` against real latents. Each with the project's
  standing bootstrap + repeated-seed discipline, never a single-split number.

## Session update (2026-09-03, later) — multimodal detection built and running

Directive extended to "build the actual models to work with all these multimodal data". Delivered
`backend/src/qhealth_qml/multimodal.py` plus `backend/run_multimodal.py`, both working today on
real cardiovascular data with no GPU and no SSD.

### The binding constraint, and why the architecture is what it is

**The modality cohorts share no patients.** PTB-XL (ECG), CADICA (angiography) and any EHR export
are entirely different people. Two consequences, the second of which was not obvious:

1. Joint/intermediate fusion (per-modality encoders trained end-to-end into a shared head) needs
   per-patient paired modalities. Untrainable here.
2. **A learned stacking meta-learner is *also* untrainable.** Fitting a stacker needs rows where
   every modality's probability is observed for the same patient; with zero cohort overlap that
   design matrix does not exist. Only a *fixed-rule* combiner is available.

So late fusion is not a compromise here, it is the only honest option. Supporting evidence: at
comparable scale (N~2,100) late-fusion stacking reached AUC 0.7213 where deep joint attention
fusion reached 0.6612 while hitting >0.95 *training* AUC — too few positives to resolve stable
cross-modal weights, so it memorised noise (arXiv 2512.14712). Latent imputation of an absent
modality is separately reported to inject noise specifically at small N (arXiv 2309.15529). No
open paired cardiovascular multimodal cohort exists: UK Biobank ECG+CMR is application-gated,
MIMIC-IV is credentialed, and the published ECG+angiography cohorts are private hospital data.

### What was built

- **`fuse_modalities`** — skill-weighted mean over *threshold-aligned* calibrated probabilities.
  The threshold alignment matters and is easy to miss: averaging raw probabilities across models
  with different operating points silently favours whichever model has the lowest threshold, so
  each model's score is remapped so its own threshold sits at exactly 0.5 before averaging.
- **Missing modalities are omitted, never imputed**, with weights renormalised over what remains.
  A case with no usable evidence raises rather than returning 0.5, which would disguise "no
  evidence" as "uncertain".
- **`skill_weight`** — a modality's influence comes from its own validated balanced accuracy
  (0.5 = chance = zero weight). **`train_modality_model`** derives that weight from the model's
  *validation* score, never its test score, so the locked test fold cannot leak into the combiner.
- **`fuse_latents`** — for the paired case if it ever arrives: packs per-modality latents into one
  vector under a fixed per-modality qubit budget, so no modality's circuit enters the
  exponential-concentration regime. Absent modality zeroes its own slice (a real imputation — pair
  with modality dropout if used).
- **`run_multimodal.py`** — trains each modality on its own cohort, and with `--paired` (same
  patients across modalities) scores each modality alone *and* the fused detector on one shared
  locked test split. Quantum head width auto-caps to a modality's feature count.

### Measured, on real cardiovascular data (5,110-patient stroke cohort, 249 positives)

Two genuinely distinct hospital data sources over the same patients — EHR problem-list/demographics
vs measured labs/vitals:

| Configuration | Locked-test balanced accuracy |
|---|---|
| structured_clinical (EHR) alone | 0.7772 |
| biomarker (glucose/BMI) alone | 0.5254 |
| **Skill-weighted fusion** | **0.7792** |
| Equal-weight fusion (ablation) | 0.7767 |

The near-chance biomarker modality earned weight **0.0231** and was effectively muted; under equal
weights it dragged the fusion *below* the best single modality. That is the skill-weighting doing
its job, and it is the one design claim this run actually supports.

**Honest reading: the deltas are ~0.002, well inside single-split noise.** The ordering matches
theory but nothing here is significant without bootstrap and repeated seeds. Literature independently
predicts the multimodal gain itself is modest below ~6k samples. Treat as a screening number.

**Never report a fused accuracy across the disjoint PTB-XL/CADICA cohorts** — no patient there has
been evaluated under two modalities, so no such number is measurable. The module docstring carries
this warning so it survives past this session.

Validation: 48 backend tests pass (14 new for the multimodal layer, including end-to-end fusion over
two really-trained artifacts and graceful degradation when a modality is absent).

### Angiography modality — deterministic path built (`angiography.py`)

Rather than leave imaging blocked behind torch, the imaging modality got the same treatment ECG has:
`angiography.py` is the imaging counterpart to `ecg.py` — a dependency-light deterministic feature
extractor (scipy.ndimage + PIL only, no new dependencies) so imaging works with no GPU at all. The
learned 2-D CNN encoder remains the better representation when a GPU exists, exactly as
`raw_hybrid.py` is for ECG.

Features are domain-chosen, not generic image statistics: the core descriptor is **multi-scale
Hessian vesselness** (Frangi et al., MICCAI 1998), implemented directly from Gaussian derivatives.
A coronary angiogram is a contrast-filled vessel tree and a stenosis is a *local narrowing of vessel
calibre*, so sweeping the smoothing scale reports which calibres are present, plus a
`dominant_calibre_scale_index` and a coarse-to-fine ratio.

Three real defects were found by testing against synthetic vessels, and all three are worth knowing:

1. **Frangi's `c` normalisation manufactures vessels in empty frames.** `c` is conventionally half
   the maximum Hessian norm *in that image* — purely relative, so on a near-blank frame the maximum
   is itself tiny, every noise ripple clears it, and pure noise scored *higher* vesselness than a
   real vessel. Fixed with an absolute floor on `c` (valid because frames are normalised to [0,1]).
2. **Percentile normalisation amplifies a blank acquisition's sensor noise to full contrast**, which
   then reads as genuine structure downstream. `normalize_frame` now reports a frame as empty when
   its 1-99 percentile spread is a negligible fraction of its own level, rather than stretching it.
   There is a regression test asserting a blank frame yields exactly zero vesselness.
3. **`vesselness_p90` was useless and read exactly 0.0000 for a real vessel at three of four
   scales.** A vessel tree covers only a few percent of a frame, so mid percentiles land in empty
   background regardless of vessel strength. Replaced with `vesselness_max` + `vesselness_p99`;
   after the change every peak/tail feature separates a vessel frame from vessel-free soft tissue at
   every scale, where the mean did not separate them at all. General lesson, and the second time it
   bit this session (the scale-selection test hit the same thing): **for sparse structure, peak and
   far-tail statistics discriminate and the mean does not.**

Still needed for the imaging modality to actually run: CADICA downloaded (3.08 GB) and labels wired
in. The fusion layer already accepts it.

## Session update (2026-09-03) — the two problem-statement objectives, assessed honestly

Direct challenge raised: are we actually meeting the problem statement's objectives, namely (1) a
hybrid quantum-classical architecture and (2) accurate *early* disease prediction? Assessed against
evidence rather than intent, the answer was **no on both**, and one of the two has since been fixed.

### Objective 2 "early detection" — was not being met at all, now genuinely is

Checked all five profiles programmatically: **every one declared `task_type: early_detection` and
not one supplied a prediction horizon or outcome time.** The platform's own validator already knew —
it rejected `p1_stroke_clinical` when `run_multimodal.py` was pointed at it, and that profile's
description says outright "no prediction horizon can be defined... Treat as an associative
risk-factor model, not early detection." Every cohort was doing *concurrent detection of disease
already present and already annotated*. No amount of model tuning fixes that; only different data does.

**Fixed with the MUSIC study** (PhysioNet, ODbL, verified open — HTTP 200, no credentialing, no DUA,
`subject-info.csv` is 301 KB and separable from the 90 GB of ECG recordings):
`https://physionet.org/content/music-sudden-cardiac-death/1.0.1/`. 992 ambulatory chronic-heart-failure
patients, measurements taken at enrolment 2003-2004, followed a median of 44 months. Prospective by
construction, so the outcome genuinely postdates every predictor.

`backend/data/music_cardiac_death/prepare.py` + `backend/profiles/music_cardiac_death.json` build the
cohort: endpoint is the study's own primary outcome, cardiac death (sudden cardiac death or
pump-failure death) within a **1461-day (4-year) horizon**. Outcome codes were verified against the
published cohort counts by crosstab before being trusted (exit=3/cause=3 -> 94 SCD, cause=6 -> 100 PFD,
cause=1 -> 61 non-cardiac, exit=2 -> 20 transplants, exit=1 -> 11 lost). Landmark labelling: 185
positives, 703 negatives observed event-free through the full horizon, and **104 patients excluded
rather than mislabelled** (38 censored before the horizon with unknown status, 66 competing-risk
non-cardiac deaths). Calling those 104 negatives would have manufactured a better-looking model.

**This is the first profile in the project's history that passes the early-detection contract.**

Three data-quality defects were found and fixed while preparing it, all of which would have degraded
the model silently:
1. **`Age` contains the string `">89"`** — the same de-identification pattern as PTB-XL's `age=300`,
   in a different disguise. Being non-numeric it turned the entire Age column categorical, and age is
   among the strongest prognostic variables in heart failure. Clamped to 89.
2. **`Holter onset (hh:mm:ss)`** one-hot expanded into ~590 dummy columns on an 888-row cohort.
   Converted to hour-of-day, one numeric feature, circadian information retained.
3. **`Number of ventricular premature contractions per hour` is irrecoverably mangled** — the
   exporter dropped the decimal point and used commas as thousands separators, so magnitude cannot be
   recovered from the string: `"4,970,833,333"` is 49.708 (that patient's 24h count is 1193) while
   `"7,946,666,667"` is 794.67 (24h count 19072). Verified by cross-checking against the clean 24h
   column. Dropped rather than guessed — VPC/hour is just the clean 24h column over 24, so nothing is
   lost, and guessing would have injected noise into precisely the ventricular-ectopy signal that
   predicts the sudden-cardiac-death endpoint.

Net effect: 1012 features (mostly junk dummies) -> **97 clean numeric features**, 888 patients.

**First real early-detection results** (patient-grouped split, target sensitivity 0.8):

| Model | Balanced acc | Sensitivity | Specificity | AUROC |
|---|---|---|---|---|
| logistic_regression | 0.667 | 0.879 | 0.455 | **0.768** |
| hist_gradient_boosting | 0.632 | 0.879 | 0.386 | 0.762 |
| rbf_svc | 0.510 | 0.909 | 0.110 | 0.670 |
| qsvc angle_scale=0.2 | 0.567 | 0.879 | 0.255 | 0.698 |
| qsvc angle_scale=0.4 | 0.520 | 0.909 | 0.131 | 0.710 |
| qsvc angle_scale=0.1 | 0.517 | 0.909 | 0.124 | 0.683 |

AUROC 0.768 is a sanity check passed, not a disappointment: published heart-failure risk models
(Seattle Heart Failure Model, MAGGIC) sit around 0.70-0.75 on this kind of prediction. The gap from
the ECG task's 0.937 AUROC *is* the difference between prognosis and concurrent detection, and should
be presented that way rather than as a regression.

### Objective 1 "hybrid" — plumbing is real, the quantum contribution still is not

Running scorecard across everything this platform has measured: **zero wins, one parity, and losses
everywhere else.** P1 stroke 0.549 (loss), P4 seizure 0.8125 vs classical 0.875 (loss), P6 Parkinson's
0.611 vs 0.734 (loss), P3 glioma both at chance, P5 Alzheimer's 0.819 vs 0.823 (parity, and only after
`angle_scale` was swept). Now add two more from this session:

- **ECG MI, re-measured post-age-fix**: classical logistic BA **0.8620**, AUROC **0.9374** vs QSVC
  BA 0.6245, AUROC 0.6602 — a decisive loss.
- **MUSIC early detection**: best QSVC 0.567 vs classical 0.667 — another loss. `angle_scale` again
  behaved as a real lever (0.517 -> 0.567 going from scale 0.1 to 0.2), reproducing the P5 finding,
  but nowhere near enough to close the gap.

The commissioned research points at the cause being architectural rather than a tuning failure: a
*terminal* quantum head bolted onto a frozen compressed latent is exactly the design a published
ablation found adds nothing on 1D biosignals. The designs that do report gains place the quantum layer
**mid-network, trained end-to-end** (dressed circuits via `TorchConnector`), or use QCNN on imaging.
Until that redesign is tried, "hybrid" here means hybrid plumbing with an honestly-reported quantum
arm that loses — which is a legitimate scientific result, but is not the objective as written.

### Bonus: MUSIC also closes the paired-multimodal gap

MUSIC carries 936 Holter ECGs and 687 high-resolution ECGs **for the same patients** as the clinical
variables, under one shared outcome and horizon. That is the paired cardiovascular multimodal cohort
this session had concluded does not exist, and which forced the fixed-rule combiner. On MUSIC,
joint/intermediate fusion and learned stacking become trainable, `fuse_latents` becomes exercisable,
and "does adding ECG beat clinical variables alone?" becomes *measurable* on one shared locked test
split. The ECG recordings are a 90 GB download, so this is gated on storage, not on licensing.

### Side result: the age fix was worth making

Re-measured MI baseline after clamping PTB-XL's `age=300` sentinel: balanced accuracy
**0.8572 -> 0.8620**, AUROC **0.9346 -> 0.9374**. Small, in the predicted direction, and it means the
previously-reported numbers were measured on distorted age encoding.

### Storage: the external SSD is not usable, working set relocated

The SSD failed a third time. Sequence: I/O errors -> apparent recovery (reads worked, a WFDB record
verified) -> failure again under sustained load, with the torch download reporting "resume incomplete
download, attempt 5" then `No such file or directory` on a file it had just written, and
`records500/11000/` returning I/O errors mid-traversal. It holds data at rest but drops writes.
Treat `/mnt/ssd` as unreliable; do not plan work against it.

Working set moved to the internal disk at `backend/data/ptb-xl/1.0.3` (gitignore extended with
`data/**/*.dat`, `*.hea`, `*.npz` so 1.7 GB of waveforms cannot be committed). Root disk had only
~7 GB free, which does not fit both PTB-XL records (~1.7 GB) and CUDA torch (~5 GB), so records won:
the deterministic baseline needs no torch, while the raw-hybrid CNN needs a GPU anyway and belongs
on the GCP VM. **The GPU work is now genuinely gated on `gcloud auth login`.**

---

# Neurological Conditions Platform Context (2026-09-02)

A separate, concurrent effort in this same repo: a research-use hybrid quantum/classical
platform covering six neurological conditions (stroke, ICH, glioma, seizure, Alzheimer's,
Parkinson's). Detail lives under `specs/001-neurological-conditions/` — this section is a
narrative summary; per-condition numbers live in `specs/001-neurological-conditions/research/*.md`
and `ACCEPTANCE-CRITERIA.md`.

## Objective and governing policy

Classical baselines were already built and validated. This pass's directive: the classical
model is not the product — it is only a reported comparison point (spec FR-025). The product
is the hybrid quantum-classical architecture itself. A QML candidate is promoted to
`lifecycle: "operational_reference"` whenever it independently clears a **baseline-viability
gate** (95% CI lower bound on balanced accuracy > 0.5), regardless of whether classical still
scores higher — classical's delta is always reported, never hidden, but no longer gates
promotion (`ACCEPTANCE-CRITERIA.md` §0).

Two standing rules governed this pass, both given directly by the user and treated as durable:
- **Never trust a single-split result without independent verification** (typically 10-seed
  repeated evaluation) before reporting or registering it. This caught two real false leads
  this session (a P5 config that looked like 0.745 single-split but was 0.564±0.082 repeated;
  a P6 config that looked like 0.93 single-split but was 0.63±0.12 repeated) before either
  reached the registry.
- **Delegate mechanical/long-running execution to subagents; do direct verification yourself**
  for anything that informs a promotion decision.

## What data the platform actually runs on

Every condition with a working model today is small tabular data, ≤6 raw feature dimensions
ever reaching the quantum circuit (4 qubits in every actively-tuned condition):

| Condition | Source | Rows | Input | Reduced to |
|---|---|---|---|---|
| P1 stroke (tabular arm) | Kaggle `healthcare-dataset-stroke-data` | 5,110 | 10 raw clinical/demographic fields -> 21 one-hot encoded | 4 qubits |
| P4 seizure | PhysioNet CHB-MIT, single patient (chb01), raw EEG | 2,700 windows | 23-channel EEG -> 5-band Welch power-spectrum summary = 115 features | 4 qubits |
| P5 Alzheimer's | OASIS-1 cross-sectional (Kaggle mirror) | 235 | Age/Educ/SES/MMSE + 3 MRI-*derived* volumetric summary numbers (eTIV/nWBV/ASF) | 4 qubits |
| P6 Parkinson's | UCI Parkinson's voice dataset | 195 (32 subjects) | 22 acoustic/voice measures from sustained phonation | 4 qubits |

**Not built** — registered `not available`, zero or conclusively-failed models:
- **P1 arm A** (actual stroke lesion detection from CT/CTA/CTP) — deferred, blocked on data access.
- **P2 ICH** (hemorrhage subtyping from CT) — no lawful non-DUA dataset found after two independent
  exhaustive checks.
- **P3 glioma MGMT status** — *was* built on real MRI-derived radiomics (UPenn-GBM, 256 cases,
  CC BY 4.0, no DUA) and tested properly (24-config sweep), but failed conclusively — both
  classical (0.474) and QSVC (0.451) landed at chance. The platform's one case where real
  imaging-derived features were used end-to-end, and it still failed (informative — see below).
- **Seizure onset/event localization** (vs. the window-classification P4 actually does) — not built.

**Why not the data radiologists actually use?** Two distinct constraints:
1. **Legal/access barriers.** ADNI (PET/CSF-confirmed Alzheimer's), PPMI (Parkinson's
   imaging/biomarker consortium), OASIS-3 (longitudinal follow-up) all require a formal Data Use
   Agreement with institutional review and human signature — not something an automated agent can
   complete. Public, license-clear alternatives were used where they existed (OASIS-1's
   cross-sectional release, UCI's voice set); where none existed, the condition was registered
   `not available` rather than faked (P2).
2. **An architecture-level bottleneck.** Even where real imaging/raw signal *was* obtained (P3's
   mpMRI, P4's raw EEG), the quantum circuit only encodes one feature per qubit at 4-6 qubits — so
   the MRI had to be collapsed into radiomics texture statistics and the EEG into band-power
   summaries before reaching the circuit, rather than the raw image/waveform. P3's conclusive
   chance-level failure despite starting from real imaging demonstrates this bottleneck is real.

What would actually move this forward: (a) a human completing the DUA applications for
ADNI/PPMI/OASIS-3; (b) replacing the hand-engineered radiomics/band-power front end with a
*learned* embedding (e.g. a CNN compressing the image/waveform into a dense representation)
before the quantum layer — the technique that distinguished the strongest result found in
literature research (a 99%-accuracy EEG paper) from this platform's own approach.

## Literature research conducted (two passes, 8 parallel forks)

- **Exponential concentration** (arXiv 2208.11060): fidelity quantum kernels informative enough
  to be useful are provably classically simulable — more qubits/entanglement/reps push *toward*
  concentration, not away from it.
- **UCI tabular benchmark** (arXiv 2604.18837): classical wins decisively over QSVM on 9
  biomedical/physics datasets (0.830 vs 0.649 mean BA, 0/29 comparisons significant) — matches
  this platform's own P1/P5/P6 findings.
- **Barren plateaus**: not this platform's bottleneck at 4-6 qubits / 1-3 reps (onset reported at
  ~8+ qubits with deep circuits).
- **Bandwidth-tuned quantum kernels** (arXiv 2503.05602): scaling the angle-encoding range
  directly counteracts concentration — the one lever not yet swept, became this session's main
  engineering addition and its one major win (P5, below).
- **Quantum-kernel-as-classical-ensemble-feature**: one paper reported a modest real gain; tested
  here, no effect.
- **Class-imbalance-specific quantum techniques**: weighted/cost-sensitive SVM (WSVM) has solid
  classical precedent, portable via `sample_weight`; tried on P1, no benefit.
- **Quantum kernel ensembling/bagging, small-N-specific techniques**: no credible supporting
  literature found; not acted on.
- Healthcare-specific review (2026, 94 studies): QML-in-healthcare papers "achieve >95% accuracy
  in simulations but lack real-world clinical validation" — a caution matching this session's own
  experience with single-split overclaiming.

## Engine capabilities added this session (`backend/src/qhealth_qml/experiment.py`, `study.py`)

- **`angle_scale`**: multiplier on the ZZFeatureMap angle-encoding half-width (previously
  hardcoded to `pi/2`). Directly targets exponential concentration; produced the P5 result below.
- **`class_weight="balanced"`**: cost-sensitive fit via `compute_sample_weight`, passed to any
  estimator whose `.fit()` accepts `sample_weight` (falls back silently otherwise, e.g. VQC).
- **`run_nested_evaluation`/`benchmark_model()` extended** to forward `angle_scale`,
  `feature_map_reps`, `feature_map_entanglement` into inner-fold tuning and outer evaluation —
  closes a gap where the real-gain (paired classical-delta) comparison was running an *untuned*
  config even for a model promoted on a *tuned* one.
- **`qsvc_aligned`** (kernel-target alignment via `QuantumKernelTrainer`/`svc_loss`) and
  **`vqc_restarts`** (best-of-N independently-initialized VQC fits) — built earlier this session,
  verified against real platform data this pass.

## Results by condition

- **P4 seizure — promoted.** QSVC (C=5, feature_map_reps=1, entanglement=circular): production
  BA **0.8125**, 95% CI **[0.6247, 1.0]**. Caveat: single structurally-forced chronological split,
  only 8 test positives; real-gain nested comparison could not run (too few/clustered positives).
- **P5 Alzheimer's — promoted, the session's main finding.** Default-bandwidth QSVC (C=5) was a
  false lead (single-split 0.745, true repeated mean 0.564±0.082 — caught before promotion).
  `angle_scale=0.2` genuinely rescued it: 10-seed repeated mean **0.819±0.039**, bootstrap CI
  **[0.662, 0.903]**, confirmed across a 5-point neighborhood scan — near classical parity (0.823).
  After extending nested evaluation to support `angle_scale`, the real-gain comparison against
  classical (now on the *actual* promoted config) is **[0.0000, 0.0903]** — essentially tied,
  versus the untuned run's earlier **[-0.288, -0.108]** (looked like a decisive loss).
- **P1 stroke — thoroughly tested, honest negative.** Six levers tried (hyperparameter/ansatz
  sweep, encoding bandwidth, quantum-kernel-as-feature, kernel alignment, VQC restarts,
  class-weighted QSVC); none moved it toward the gate. Best production result: 0.549, CI
  [0.439, 0.655]. Kernel alignment confirmed prohibitively expensive (~2000x overhead, matching
  literature) as well as ineffective.
- **P6 Parkinson's — thoroughly tested, honest negative.** Five levers tried; one apparent
  bandwidth win (single-split 0.931) was caught and reversed by repeated evaluation (true mean
  0.633±0.120).

## What's left, if pursued further

- P1 and P6 have exhausted the reasonable literature-grounded search space at their current data
  scale and 4-qubit feature budget. Further improvement needs richer data (see above) or a
  fundamentally different encoding architecture (e.g. data re-uploading via a hand-built
  `EstimatorQNN`/`SamplerQNN` circuit instead of the high-level `VQC` API) — not more tuning on
  the current axis.
- The registry's baseline-viability gate (`registry.py` Rule 4) currently checks only the
  single-split bootstrap CI, not repeated-evaluation robustness — the exact gap that nearly caused
  a false P5 promotion this session before being caught by hand. Not yet hardened in the platform
  code itself.
- The GCP GPU VM (`gcloud compute ssh cent-data-science-gpu --zone=us-central1-b`) is available if
  local compute becomes the bottleneck — kernel alignment in particular is expensive enough that
  this may become relevant for future sweeps.

## Neuro session update (2026-09-03) — radiologist-grade modalities, five new ingest paths

Directive for this pass: *use the data a radiologist would actually use for each condition, find
that data, and train models on it.* The prior passes had exhausted the architecture-search space on
tabular/summary features and concluded the ceiling was the **input representation**, not the
circuit. This pass tested that by changing the data.

### Data access — five conditions unblocked, all open-licensed and fetched without any DUA

| Condition | Dataset | Licence | Size | Access |
|---|---|---|---|---|
| P6 Parkinson's | PhysioNet **gaitpdb** (16-sensor force-plate gait, 100 Hz) | ODC-BY | 288 MB | plain HTTP |
| P1 stroke | **ISLES 2022** (DWI/ADC/FLAIR MRI, 250 cases + expert masks) | CC BY 4.0 | 1.69 GB | Zenodo HTTP |
| P2 ICH | **BHSD** (192 head CTs, 5-subtype voxel annotations) | **MIT** | 1.45 GB | HuggingFace HTTP |
| P5 Alzheimer's | **Zenodo 3935636** (T1 MRI, 35 AD / 19 control / 24 bipolar) | CC BY 4.0 | 5.28 GB | Zenodo HTTP |
| P3 glioma | **UPenn-GBM** raw mpMRI (T1/T1-post/T2/FLAIR) + MGMT labels | CC BY 4.0 | streamed | **TCIA NBIA REST API** |

Two access findings worth keeping: **TCIA's REST API serves DICOM over plain HTTP with no
credentials** (the assumption that NBIA Data Retriever or Aspera is required is wrong), and **BHSD
is a re-annotation of RSNA ICH data**, which is how it distributes lawfully without RSNA's gate —
that single fact converted P2 from "requires a human to unblock" to trainable.

Confirmed still gated (institutional DUA, human signature — not obtainable by an agent): OASIS-1/2/3
raw image tiers, ADNI, PPMI, PhysioNet ct-ich, INSTANCE 2022, RSNA's own AWS mirror. Explicitly
rejected on provenance grounds: the Kaggle/HF "Alzheimer MRI" slice datasets (no traceable source,
near-certain same-patient leakage across train/test), and IXI-as-negative-class (pairing cohorts
means the model separates scanners, not pathology).

### New modules (all follow the existing raw_hybrid.py contract)

`gait_hybrid.py` (1-D force-plate CNN) · `imaging_hybrid.py` (shared 3-D volume encoder + leak-free
split/train scaffold) · `isles_stroke.py` · `bhsd_ich.py` · `ad_mri.py` (streams subjects straight
out of the zip) · `upenn_gbm.py` (streams DICOM series from TCIA) · `hybrid_qnn.py` (end-to-end
`TorchConnector` hybrid). Ingest downsamples to a small grid and caches to `.npy`, so raw trees are
deleted after processing — peak disk stays small regardless of collection size, which is what made
this feasible on a 96%-full disk.

### Task framings (each chosen against the data, not assumed)

Several releases contain *only* positive cases, so the obvious binary is unavailable and inventing
one would be dishonest. What was used instead:
- **ISLES**: infarct **core volume** ≥ **21 mL** — the DAWN thrombectomy cutoff. DEFUSE-3's 70 mL was
  rejected after measuring the distribution (median 6.7 mL) because it leaves only 22/250 positives.
  Labels come from the expert mask; the model sees only the images.
- **BHSD**: haemorrhage **subtype** (intraventricular, 104/192) — which compartment blood occupies is
  what changes management. CT is *windowed*, not percentile-normalised: the three channels are the
  brain/subdural/bone windows a radiologist toggles through.
- **P5**: AD vs healthy control, n=54 (bipolar arm held out).

### Results (leak-free: heads only ever see the encoder's held-out split; multiple seeds)

| Condition | Before (tabular/summary) | After (radiologist-grade) |
|---|---|---|
| **P6 Parkinson's** | QSVC 0.561 ± 0.104 — **failed** gate | **QSVC 0.750 ± 0.040, CI [0.701, 0.799] — clears** |
| **P1 stroke** | QSVC 0.549, CI [0.439, 0.655] — **failed** | **QSVC 0.810 ± 0.043, CI [0.704, 0.916] — clears** |
| P2 ICH | no model existed | first pass at chance (0.579, CI spans 0.5); resolution re-run in progress |

**The headline: two conditions that failed six independent architecture-search levers cleared the
viability gate immediately once the input modality changed, with the quantum configuration
untouched.** The QSVC also gained far more from richer input than classical did on P6 (+0.19 vs
+0.035). No quantum *advantage* is claimed anywhere — classical remains level or ahead, CIs overlap,
and seed counts are small.

### Honest negatives from this pass

- **End-to-end hybrid underperformed the decoupled design** on gait: 0.740 ± 0.132 vs 0.750 ± 0.040,
  with 3.3× the variance. A single seed hit 0.876 and was *not* reported as a result. Root-caused to
  a self-inflicted bottleneck: `EstimatorQNN`'s default observable is a single global Z⊗ⁿ, so a 177k
  -parameter encoder was communicating through one scalar. Fixed to per-qubit observables (Mari et
  al.); re-run deprioritised in favour of covering more conditions.
- **P2 first pass at chance**, with textbook overfitting (loss 0.23→0.07, val AUROC flat). Diagnosed
  as 8× in-plane downsampling (512→64) averaging away the thin hyperdense IVH signal; 128×128 re-run
  under way. ISLES tolerated 64³ because ≥21 mL infarct cores are large diffuse regions.

### Still open

P2 resolution verdict; P3 and P5 training (both fetched/fetching, ingest verified end-to-end);
registry entries for the new modalities (each warrants its own `ModelDefinition` rather than editing
the tabular entries); the end-to-end hybrid re-run with per-qubit observables.

## Neuro session update (2026-09-03, later) — production layer + existing-implementation adoption

Directive extended to: adopt the existing implementations properly, collect more data, and make
the result **integrable and usable somewhere else** — connected to an ingestion pipeline with
usable outputs.

### Honest status of the models themselves

| Condition | Measured | Deployable artifact |
|---|---|---|
| P1 stroke (ISLES MRI) | held-out BA **0.7765**, sens 0.769, spec 0.784, AUC 0.879 | ✅ `stroke-core-volume-mri.pkl`, registered |
| P6 Parkinson's (gait) | 0.750 [0.701, 0.799] | bundling in progress |
| P2 ICH / P3 glioma / P5 Alzheimer's | all at chance | ❌ not worth deploying — cohort-size limited |

One deployable model, not five. That is the accurate statement.

### Existing implementations — adopted, having initially only been installed

The first pass installed MONAI and used it in exactly one function. Now:
- **`monai_preprocess.py`** uses MONAI's real transforms (`LoadImage`, `Orientation`, `Spacing`,
  `ScaleIntensityRange`, `Resize`). **This fixed the D5 respacing defect as a side effect** —
  verified on three sequences at 1.0 / 2.0×3.0 / 0.5 mm spacing, which previously would have
  occupied three different physical fields of view in the same tensor.
- **`medical3d_encoder.py`** loads frozen **MedicalNet** 3-D ResNet18 (Tencent, MIT, 23 medical
  datasets) — an in-domain volumetric prior replacing ImageNet-on-slices.
- **`hybrid_qnn.py`** uses Qiskit's `TorchConnector` for a genuinely end-to-end hybrid.
- Canonical order now enforced in one place: `load → HU rescale → orient(RAS) → respace →
  intensity → grid → stack`.

### Encoder A/B on identical P5 data (3 seeds each)

| Encoder | trainable/total | `qsvc` | 95% CI |
|---|---|---|---|
| From-scratch 3-D CNN | 300k/300k | 0.4345 | [0.158, 0.711] |
| Frozen ImageNet R18, slice-wise | 795k/12.0M | 0.6071 | [0.209, 1.006] |
| Frozen MedicalNet 3-D | 140k/33.1M | 0.5238 | [0.407, 0.640] |

Transfer learning materially beats from-scratch; the in-domain prior is far more *stable*
(std 0.047 vs 0.160) though not higher-mean. No quantum head clears the gate on P5 under any
encoder — the architecture question is now answered as far as n=54 can answer it.

### Production layer (`serving.py`, `ingestion.py`, `serve_api.py`, `cohort_audit.py`)

- **HTTP service** — `POST /predict/{model_id}` with file paths or multipart upload. This is what
  makes it usable outside this repo; a consumer no longer needs `qhealth_qml` importable.
- Refusals are **HTTP 200 with `status:"rejected"` and machine-readable reasons**, not 500s.
- Every response carries `temporal_framing`, `limitations`, `research_use_only`.
- `requirements-serving.txt` (15 pinned deps) + `INTEGRATION.md` documenting the real
  portability blockers (pickle format, `qhealth_qml` import, runtime weight download).
- **69 tests passing**, 12 of them end-to-end acceptance tests.

### Two bugs the production layer caught that research code would have shipped

1. **Score-normalisation clipping.** Raw `decision_function` output (unbounded, often negative)
   was clipped to [0,1], flattening negatives to zero, driving the selected threshold to 0.0 and
   producing an all-positive predictor. The model was fine — ROC-AUC 0.819 — the deployment path
   destroyed it. Fixing it moved held-out balanced accuracy **0.500 → 0.7765** on identical data.
   Normalisation constants are now persisted in the artifact and regression-tested.
2. **Batch scoring rebuilt the encoder per sample** (peer-reported) — 33M parameters reconstructed
   per study. Fixed, with a call-count assertion.

### Extended rather than worked around

`protocol.py` hardcoded `task_type == "early_detection"` and a modality allowlist excluding
imaging. Widened both (`TASK_TYPES`, `PROFILE_MODALITIES`) rather than mislabelling a
characterisation task as early detection to satisfy a validator.

### Early detection

`chbmit_preictal.py` reframes P4 from detection to **pre-ictal prediction**: 35–5 min before
onset, with a deliberate 5-minute seizure-prediction-horizon gap, ictal/post-ictal discarded,
patient-grouped splits. Literature target is AUC ≈ 0.81 patient-independent; anything >95% in the
literature is window-wise leakage.

### Cross-session collaboration

A peer session (`neutral-75`) is building a unified ingestion pipeline (`design.md`). It found two
real bugs in this session's serving layer and the D5 defect. `cohort_audit.py` was written so D5
exposure is a measurement rather than a judgement call, and is now load-bearing in their design.
Phase 4 of that design replaces `InferenceBundle`'s fitted transforms with a single `Recipe`
object — agreed handover, not yet started.
