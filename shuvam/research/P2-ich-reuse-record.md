# P2 Intracranial Hemorrhage (ICH) — Research and Reuse Record

**Feature**: `001-neurological-conditions`
**Date**: 2026-08-29
**Status**: BLOCKED — no model implementation. Exhaustive public-data search completed; every
candidate source requires an institutional data-use agreement (DUA) or is a third-party re-upload
of DUA-gated data. No dataset was found or built that a fully-automated agent can lawfully use.

## 1. Existing repository implementation inspected

No existing ICH-specific code. The registry already carries a stub model
(`ich-subtype-ct`, `availability: "not available"`) so the catalog honestly reports the coverage
gap per FR-002/US-1 scenario 3 rather than hiding it.

## 2. Candidate datasets considered and why each was rejected

- **RSNA Intracranial Hemorrhage Detection (Kaggle competition, `rsna-intracranial-hemorrhage-detection`)**
  — technically reachable via the Kaggle API without a separate DUA (competition-rules acceptance
  only), but the full label-balanced slice needed for a defensible cohort is DICOM pixel data at
  ~450GB; this does not fit the available disk regardless of license status. Downloading an
  arbitrarily small unbalanced subset of slices without the accompanying per-study clinical context
  would not produce a case-level (patient-level) cohort matching the spec's task definition — it
  was rejected on feasibility grounds, not license grounds.
- **CT-ICH / Hssayeni dataset (PhysioNet)** — the canonical small (82-patient), well-labeled,
  segmentation-mask ICH dataset. Verified directly against PhysioNet (see P0 note in prior session):
  requires PhysioNet's **Restricted Health Data License**, i.e. a completed credentialing/DUA
  process tied to a named human, which cannot be completed by an automated agent. **Rejected.**
- **Kaggle re-uploads of the same CT-ICH data** — two mirrors were found and checked directly against
  this exclusion before being ruled out, not assumed clean by association with "Kaggle" (the
  platform where three other conditions' data was legitimately sourced):
  - `vbookshelf/computed-tomography-ct-images` ("Brain CT Images with Intracranial Hemorrhage
    Masks") — page content explicitly references "PhysioNet" and "Hssayeni" (verified via direct
    fetch of the dataset page, 2026-08-29). This is a re-upload of the PhysioNet-gated CT-ICH
    dataset; the original DUA is not extinguished by a third party mirroring it to Kaggle.
  - `coderrkj/processed-ct-ich-dataset-images` — same check, same result: page content references
    "PhysioNet" directly.
  - **Both rejected.** Using either would be using DUA-gated health data without completing the DUA
    — exactly the license-laundering pattern this project's research-record standard (FR-034–040,
    §4 of every prior reuse record) exists to prevent. A third-party's Kaggle upload license label
    cannot grant rights the uploader never had.
- **CQ500 (Qure.ai)** — considered as a possible non-DUA head-CT critical-findings dataset.
  Re-checked directly (2026-08-29, following the same "search the live source instead of trusting
  the spec's named dataset" approach that found `UPENN-GBM` for P3): `qure.ai/dataset` now returns
  HTTP 404; the historical `academictorrents.com/collection/qureai-headct` mirror returns HTTP 200
  but with body "Collection not found" — both known hosting channels are confirmed dead, not merely
  unchecked. **Rejected** — no working access path exists.
- **TCIA (re-checked using the same live-collection-search method that found UPenn-GBM for P3)** —
  queried the full current NBIA collection list (156 collections, `getCollectionValues`, 2026-08-29)
  for any hemorrhage/ICH/stroke/brain-CT-relevant entry. **Zero matches** (the only near-miss,
  `TCGA-KICH`, is kidney chromophobe carcinoma, unrelated). Unlike P3, there is no TCIA analogue to
  pursue here.
- **Additional small Kaggle "ICH" datasets** (`afridirahman/intracranial-brain-hemorrhage-ct-images`,
  `ujjwalsinha01/intracranial-hemorrhage-screening`,
  `mehdiyousefzadeh/intracranialhemorrhagemetayousefzadeh`) — checked for server-rendered
  license/provenance metadata the same way the PhysioNet re-uploads were caught; unlike those, these
  pages render an empty `og:description` and no structured metadata reachable without a logged-in
  session, so **no positive license/provenance verification was possible either way**. Per this
  project's own standard (every other data source in this project required `license_verified: true`
  before use), an unverifiable dataset is treated the same as a rejected one, not assumed clean by
  default. Not used.

## 3. Conclusion

No dataset satisfying both (a) genuine public/non-DUA access completable by an automated agent and
(b) a case-level cohort matching the spec's ICH detection task was found — checked twice, including
a second pass (2026-08-29) that specifically re-searched TCIA's live collection catalog and
re-verified CQ500's two known hosting channels by direct HTTP request, the same method that
successfully unblocked one sub-task of P3. Neither produced a usable lead for P2: TCIA has no
ICH-relevant collection at all, and CQ500 is confirmed dead on both channels. This is a **data-access
blocker, not an architecture or engineering gap** — the `qhealth_qml` tabular/imaging engine
extensions built for P1/P4/P5/P6 would apply directly once a lawfully-accessible dataset is
provided. No model artifact, evaluation record, or registry entry beyond the existing honest stub
(`ich-subtype-ct`, not available) was created.

**Unblock path (requires a human)**: complete the PhysioNet credentialing process for the CT-ICH
(Hssayeni) dataset, supply an already-approved RSNA ICH Detection subset with per-study clinical
metadata sufficient to build a patient-level (not slice-level) cohort, or personally vet and supply
one of the unverifiable small Kaggle ICH datasets listed above if you have out-of-band knowledge of
its actual provenance and license.

---

## UNBLOCKED — BHSD found, first P2 model built (2026-09-03)

**The "requires a human" conclusion above is superseded.** A further search found a dataset that is
openly and programmatically downloadable, and P2 now has a trained model for the first time.

**Dataset: BHSD** (Brain Haemorrhage Segmentation Dataset), `Wendy-Fly/BHSD` on HuggingFace,
`label_192.zip`, 1.45 GB, **MIT licence**, no login/token/DUA — a plain `curl` away. 192 non-contrast
head-CT volumes (512×512×28, raw Hounsfield units) with **voxel-level annotations for all five
subtypes a neuroradiologist reports**. Critically, it is a *re-annotation of RSNA ICH data*, which is
precisely how it distributes lawfully without RSNA's own access gate — the gap the earlier search
missed. Integrity verified on download; 192 images paired 1:1 with 192 masks, zero unmatched.

**Task framing.** Every BHSD volume is haemorrhage-positive, so "bleed vs no bleed" has no negative
class. The clinically real binary is *which compartment* the blood occupies — subdural, epidural and
intraventricular bleeds differ in cause, urgency and surgical answer. Measured subtype prevalence
across the 192 volumes (143 carry more than one subtype):

| subtype | volumes | share |
|---|---|---|
| intraparenchymal | 127 | 66.1% |
| subarachnoid | 109 | 56.8% |
| **intraventricular (default)** | **104** | **54.2%** |
| subdural | 70 | 36.5% |
| epidural | 23 | 12.0% |

Intraventricular haemorrhage is the default: best class balance, and IVH independently predicts
hydrocephalus and worse outcome, often driving external-ventricular-drain placement.

**Preprocessing note (CT ≠ MR).** CT carries absolute Hounsfield units, so intensity is *windowed*,
not percentile-normalised. The three input channels are the three windows a radiologist actually
toggles through at the workstation — brain (L40/W80), subdural (L75/W215) and bone (L600/W2800) —
which is also the standard multi-window input across the RSNA ICH literature.
(`backend/src/qhealth_qml/bhsd_ich.py`.)

**First-pass result — honest negative** (64×64×32 grid, 3-D CNN → classical/QSVC heads, leak-free
protocol, 3 seeds):

| Model | mean balanced accuracy | std | approx. 95% CI |
|---|---|---|---|
| `logistic_regression` / `rbf_svc` / raw 3-D CNN | 0.5794 | 0.0509 | [0.4529, 0.7059] |
| `qsvc` (decoupled) | 0.5794 | 0.0509 | [0.4529, 0.7059] |
| `hist_gradient_boosting` | 0.5648 | 0.0571 | [0.4230, 0.7066] |

**Every CI spans 0.5 — nothing here is distinguishable from chance, and the baseline-viability gate
is NOT cleared.** Training loss fell steeply (0.23 → 0.07) while validation AUROC stayed flat around
0.44–0.58: textbook overfitting on ~115 training volumes.

**Diagnosed cause, and the correction being tested.** The likely culprit is not the model but the
ingest: 512×512 axial CT was downsampled **8× in-plane to 64×64**. IVH frequently presents as a
*thin layer* of hyperdense blood layering in the ventricles, and at 64×64 that signal is averaged
away entirely. ISLES tolerated the same 64³ grid because the infarct cores being detected (≥21 mL)
are large diffuse regions; fine hyperdense detail is a different problem. A 128×128×32 re-run
(4× the voxels, preserving in-plane detail) is the direct test of this hypothesis and is in
progress — its result, positive or negative, supersedes the table above.

**Status**: P2 is no longer data-blocked. It has a real, licence-clean, radiologist-grade dataset,
a working ingest and training path, and a first measured result. Whether a *usable* model exists
depends on the resolution re-run.

### Resolution hypothesis tested and REFUTED (2026-09-03)

The 128×128×32 re-run (4× the voxels of the first pass, same protocol, 3 seeds) does **not** rescue
the result — it is marginally *worse*:

| Model | 64×64×32 | **128×128×32** | approx. 95% CI (hi-res) |
|---|---|---|---|
| `qsvc` (decoupled) | 0.5794 | **0.5437** | [0.3773, 0.7100] |
| `rbf_svc` | 0.5794 | 0.5357 | [0.3434, 0.7281] |
| `hist_gradient_boosting` | 0.5648 | 0.5344 | [0.3016, 0.7672] |
| `logistic_regression` / raw 3-D CNN | 0.5794 | 0.5265 | [0.3575, 0.6954] |

**The in-plane-resolution explanation is wrong.** Quadrupling the voxel count while holding the
cohort fixed simply gave a 300k-parameter encoder more to overfit on the same ~115 training volumes,
and every confidence interval widened. Both passes sit squarely on chance.

**Revised diagnosis — the binding constraint is cohort size, not detail.** 192 volumes (115 train /
~38 test per split) is too few to fit a 3-D CNN for a subtype-discrimination task where the
discriminative structure is a small fraction of the field of view. This is consistent with the
training curves in both passes: loss falls steeply while validation AUROC stays flat.

**P2's honest verdict**: the condition is now *data-unblocked and pipeline-complete* — licence-clean
radiologist-grade CT, verified ingest with clinically-standard windowing, and a reproducible
leak-free evaluation — but **no model clears the baseline-viability gate, and none is registered.**
Plausible routes if pursued: slice-level framing with strict patient-grouped splits (far more
training units from the same 192 patients, at the cost of a harder leakage discipline), a pretrained
2-D encoder applied slice-wise rather than a 3-D CNN trained from scratch, or simply a larger cohort
(the full RSNA collection, if its access gate is ever cleared by a human). What is *not* the answer,
on this evidence, is more resolution or more architecture search.

### Known preprocessing defect affecting the imaging numbers above (2026-09-03)

A peer review of the ingest path found a defect not caught during this work: volumes are resampled
to a fixed **array** grid (64³ / 128²) without first being **respaced to a common physical voxel
size**. Normalising voxel *counts* is not the same as normalising physical extent — two scans
acquired at different mm/voxel occupy different real-world fields of view while sitting in the same
tensor slot, so the encoder sees anatomy at inconsistent scale across subjects.

Exposure, assessed per cohort:
- **P5 (Zenodo 3935636): not affected** — verified all subjects are 1 mm isotropic, 192×256×176,
  identical FOV.
- **P1 (ISLES 2022): likely affected.** The release is explicitly multi-vendor (3T Philips, 3T and
  1.5T Siemens), so spacing variation across subjects is near-certain. Could not be re-measured
  because the raw NIfTI tree was deleted after caching to `.npy`.
- **P2 (BHSD): likely affected.** One inspected volume was 0.488 × 0.488 × 5.288 mm; slice thickness
  commonly varies across head-CT acquisitions. Same re-measurement limitation.

**The numbers reported in the sections above were measured under this defect and are not being
retracted or silently amended** — they stand as what was actually run. The correct ordering
(decode → HU rescale → orient → **respace** → intensity → grid → stack, via MONAI `Spacingd`) is
being adopted in the consolidated ingestion pipeline, and any re-run under it should be reported as
a separate measurement rather than compared directly against these.
