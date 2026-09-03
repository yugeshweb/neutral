# `qhealth_qml.pipeline` - build status

Tracks what has actually been built against `ingestion-preprocessing-spec`
(spec.md / data-model.md / design.md / tasks.md), and exactly where this
build knowingly diverges from or simplifies the spec. Read this before
relying on any adapter for something it wasn't verified against.

## What's real and tested (as of this entry)

| Phase | Scope | Status |
|---|---|---|
| Phase 0 | Skeleton, types, spec loading, issue catalogue, QC gate, registry, ledger | Done |
| Phase 1 (tabular slice) | `tabular_csv` adapter, `Recipe` fit/align/transform, split, selection | Done - 52 tests |
| Phase 2 (signal) | `wfdb_ecg`, `edf_eeg`, `gait_txt` adapters | Built, real, tested - see caveats below |
| Phase 3 (imaging, partial) | `nifti_volume`, `dicom_series` adapters | Built, real, tested - see caveats below |
| Not in the spec at all | `image_2d` adapter (`image2d` modality) | Built, real, tested - see caveat #0 below |
| Phase 4 | Consolidating `qc.py` with `platform/execution.py`, wiring into `experiment.py`/`serving.py` | Not started - edits shared files, needs coordination |
| Phase 5 | Frontend preview endpoint | Not started |

`pytest tests/pipeline -q` → 101 passed (0 skipped) as of this writing.
`pytest tests/ -q` (whole backend) → 147 passed, 2 skipped (pre-existing,
OCR-related, unrelated to this package).

## Real-model integration (as opposed to the placeholder extractors)

`origin/main` (not merged to local `main`; see git remote state) carries a
teammate's ("shuvam") real model code under `shuvam/`. Its own
`shuvam/sync.sh` states outright: *"the files under src/ ... here are
copies for handover and review; backend/ is where they actually live and
run"*, and names the exact modules meant to land in
`backend/src/qhealth_qml/`. Pulled in (read-only `git show`, new files
only, nothing committed):

| Module | Status |
|---|---|
| `ecg.py` | **Wired in for real** via `extractors/real_ecg.py`. 290 real deterministic features, T3.4 (train==predict) verified bit-identical, a genuine 12-lead contract mismatch caught and fixed. Reported model performance (teammate's own numbers): PTB-XL cardiovascular ECG, BA 0.862, AUROC 0.937. |
| `angiography.py` | **Wired in for real** via `extractors/real_angiography.py`. 34 real deterministic vesselness/intensity features, T3.4 verified bit-identical. |
| `cardiovascular.py`, `cardiovascular_cli.py`, `prs.py`, `multimodal.py` | Import cleanly against the existing `backend/src/qhealth_qml/`. Higher-level orchestration (not single-record feature extraction), so not wired into `representation.extractor` - would need its own integration work, not attempted here. |
| `hybrid_qnn.py`, `pretrained_encoder.py` | **Broken as pushed** - both `import qhealth_qml.raw_hybrid`, which does not exist anywhere on `origin/main` (confirmed via `git ls-tree`). Not something this build can fix; needs the teammate to push the missing file. |
| `chbmit_preictal.py` (EEG/seizure) | Real and imports cleanly, but is **NOT** in `sync.sh`'s sanctioned module list - it lives in `shuvam/src/` (flat), not `shuvam/src/qhealth_qml/`. Its own team hasn't blessed it as ready for `backend/` yet, so it was deliberately not integrated here the same way. Its reported held-out performance is at chance regardless (LOPO BA 0.505) - see `shuvam/docs/INTEGRATION.md`. |

**Separately, a bigger structural fact worth flagging to your team head
directly**: `shuvam/` also contains a complete, already-deployed parallel
system - `shuvam/src/ingestion.py`, `serving.py`, `serve_api.py`, six
trained model bundles - that does NOT go through this `pipeline` package
at all; it has its own imaging ingestion path. That overlap (this
package vs. `shuvam/ingestion.py`) is a real coordination question, not
something resolved by this integration work.

**0. `image2d` is an addition beyond the spec, not part of it.**
spec.md/design.md define exactly eight modalities
(tabular/ecg/eeg/gait/ct/mr/angio/genomic); `image2d` is a ninth added in
this build to close a concrete, real gap - plain PNG/JPEG datasets with no
scanner metadata at all (no DICOM header, no NIfTI affine), the shape of
e.g. Kaggle's "Brain MRI Images for Brain Tumor Detection". `dicom_series`
and `nifti_volume` correctly refuse such files (they need real scanner
metadata this format doesn't have) - `image_2d` is a separate, explicitly
non-spec adapter for exactly that gap. Flag this to your team head before
relying on it for anything beyond local experimentation: it may not match
whatever they'd have specified for this case. See
`adapters/image_2d.py`'s module docstring for the full design (class-
subdirectory label convention, simplified bounding-box crop instead of the
reference notebook's contour-based one, etc).

## Known simplifications - read before trusting an adapter with real data

**1. Representation extractors are placeholders, not the team's real
feature-extraction model.** `extractors/reference_signal.py` and
`extractors/reference_imaging.py` compute ordinary summary statistics
(mean/std/rms/percentiles/foreground fraction) per channel. spec.md's own
worked example names a specific extractor (`ecg.extract_ecg_features`)
that is expected to already exist in the disease-detection "model
scripts" - those scripts are not in this repository. Swapping in the real
ones is a one-line change to `SourceSpec.representation.extractor` (a
`module:function` dotted path) - no pipeline code changes required. Until
that happens, any model trained on these features is learning from
placeholder statistics, not the team's intended signal/image features.

**2. `representation.kind` only supports `"deterministic"`.** `"learned"`
and `"multimodal"` (design.md §9.2.7, tasks.md T046 - fit-before-encode
ordering, multimodal availability masks) need an actual trainable encoder
and raise `RepresentationError` if declared. Not implemented.

**3. Source contract for multi-file records is directory-based, not
dict-based.** The spec's NIfTI worked example passes
`source={"t1": path, "t2": path, ...}` directly into `Pipeline.read()`.
This build instead expects a directory per case containing files named
`<sequence>.nii[.gz]` (matching `imaging.sequences`), so `Pipeline.read()`
itself needed no new Source kind. Same semantics (recipe dictates channel
order, not the caller), different on-disk layout.

**4. `dicom_series`'s affine is axis-aligned only.** Built from
`PixelSpacing`/`SliceThickness` alone - it does not read
`ImageOrientationPatient`/`ImagePositionPatient` direction cosines. A
series acquired at a genuine gantry tilt or oblique orientation will not
canonicalise correctly. Every fixture here is axis-aligned, so this gap
does not show up in tests; it is real for tilted real-world DICOM.

**5. Labels for wfdb_ecg come from WFDB header comments
(`key: value` lines); edf_eeg/gait_txt/nifti_volume/dicom_series have no
natural header field for a label, so this build's OWN FIXTURES encode it
in the filename/directory name (`<id>__label<0|1>`).** A predict-time
upload with no such suffix correctly yields `label=None` (unaffected).
Real training cohorts for these four modalities will need labels supplied
some other way (a companion tabular file, a real header field, etc.) -
this convention is fixture-only, not a declared part of the spec.

**6. Tabular covariates alongside a signal/imaging record are decoded but
unused.** `wfdb_ecg.harmonize()` parses header comments like `Age: 56`
into `Sample.fields`, matching data-model.md's worked example - but
`Recipe`'s representation path (`_align_representation`) only reads
`Sample.arrays`, never `Sample.fields`, so those covariates currently
never reach the model. Combining per-record covariates with the signal/
imaging feature vector is not implemented.

**7. Column-level cleaning (quarantine, drop-constant, cardinality guard)
has no analogue for signal/imaging.** `_fit_representation_feature_names`
skips all of it - a deterministic extractor's feature set comes from the
declared config, not discovered from column contents, so there is nothing
equivalent to quarantine/drop for it in this build.

**8. Not implemented from tasks.md at all**: `npy_cache` adapter (T057,
on-disk feature caching), the geometry-audit corpus scan across every real
cohort (T062 - meaningful once real cohorts exist, not synthetic
fixtures), streaming/memory-ceiling enforcement (T059), event windowing
with parent-recording propagation (T044), the learned-encoder path (T046,
see #2).

## Fixed while building this: two gaps in the Phase 0/1 wiring

Discovered while wiring the new adapters through `Pipeline.fit()`/`run()`:

- `ModalityAdapter.qc()` was declared on the Protocol and implemented by
  every adapter, but `Pipeline.read()` never called it. Now called right
  after `harmonize()`, per record; its issues are folded into
  `Sample.issues`.
- Reject-severity issues placed on `Sample.issues` by `harmonize()`/`qc()`
  (e.g. `channel_missing`, `sequence_missing`) were never merged into the
  `QCVerdict` for non-excluded records - only `LABEL_EXCLUDE` records got
  their `sample.issues` carried through. Both `fit()` and `run()` now
  merge reject-severity `sample.issues` into the verdict, deduped by
  `(code, field)` against what the universal QC gate and `Recipe.align()`
  already reported.

Both fixes are exercised for real by every "corruption" test in
`test_signal_imaging_adapters.py` (a `channel_missing`/`sequence_missing`/
`modality_mismatch` record would previously have been silently accepted
for anything other than tabular).
