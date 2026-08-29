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
