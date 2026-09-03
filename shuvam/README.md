# Cardiovascular hybrid ML frame — handover bundle

Work by Shuvam. A pluggable frame that declares cardiovascular input channels, fits a
detector per channel that has data, and pools their calibrated opinions into one risk
assessment. Built so it can be integrated into another system rather than only run here.

> **Canonical source is `backend/`.** The files under `src/` and `tests/` here are a
> snapshot for review and handover. `cardiovascular.py` imports `.multimodal`,
> `.experiment` and `.ecg`, so it only *runs* from inside the `qhealth_qml` package.
> Refresh this snapshot with `./sync.sh` rather than editing these copies — otherwise
> the two diverge silently.

---

## 1. What actually works

Verified end-to-end against real data on disk. Both ready modalities trained and earned
their fusion weights from their own **validation** splits:

| Modality | Data | Validation BA | Fusion weight | Test AUROC |
|---|---|---|---|---|
| `ecg_12lead` | 6000 × 294, 2960 pos | 0.8486 | **0.6971** | 0.8352 |
| `ehr_tabular` | 888 × 97, 185 pos | 0.6046 | **0.2092** | 0.7896 |

From a `--max-train 600` screening run. At full scale the ECG channel reaches
**BA 0.8620 / AUROC 0.9374**; MUSIC reaches **AUROC 0.768** over a genuine 4-year
horizon. The point of the table is that weights are *earned and differentiated*
(0.70 vs 0.21), not that these are headline numbers.

**103 tests pass.** The library imports with zero heavy dependencies — no torch, qiskit,
wfdb or monai load until you actually fit something — so `readiness_report()` is safe to
call anywhere.

## 2. Modality status

| Modality | State | Framing | Note |
|---|---|---|---|
| `ecg_12lead` | **ready** | detection | PTB-XL; reads a present state, not an early warning |
| `ehr_tabular` | **ready** | prediction | MUSIC; the only genuinely prognostic channel |
| `angiography` | needs_data | detection | Extractor implemented + tested; no frames on disk |
| `cad_prs` | needs_data | screening | Engine implemented; no genotypes anywhere |
| `echocardiography` | stub | characterisation | Declared, not implemented |

`needs_data` means **code-complete and starved**, not broken. `stub` means declared but
unimplemented. The readiness report distinguishes these deliberately so an integrator
can plan against it instead of discovering failures by catching exceptions.

## 3. What's in here

```
README.md                      this file
LICENSE.md                     licensing position + third-party obligations — READ BEFORE SHARING
CARDIOVASCULAR_FRAME.md        integration guide: API, contract, worked examples
sync.sh                        refresh this snapshot from backend/
src/qhealth_qml/
  cardiovascular.py            the frame: registry, fit, pool, readiness
  cardiovascular_cli.py        console entry point (qhealth-cardiovascular)
  prs.py                       polygenic score engine (PGS Catalog -> per-patient risk)
  multimodal.py                skill-weighted combiner
  ecg.py                       12-lead -> 294 deterministic features
  angiography.py               multi-scale Hessian vesselness
  hybrid_qnn.py                dressed quantum circuit (Mari et al.) via TorchConnector
  pretrained_encoder.py        frozen ImageNet ResNet18 encoder
tests/                         103 passing
results/                       measured results (JSON) — no patient-level rows
profiles/                      MUSIC early-detection cohort profile
prepare.py                     MUSIC cohort builder (code only; no data)
run_cardiovascular.py          dev shim for the CLI
```

**No datasets are included.** PTB-XL and MUSIC are third-party licensed and excluded by
`.gitignore`; both are freely obtainable at source. See LICENSE.md.

## 4. Quick start

```bash
pip install ./backend
qhealth-cardiovascular status
qhealth-cardiovascular fit  --artifact-dir runtime/cvd --max-train 600
qhealth-cardiovascular pool --score ecg_12lead=0.81 --score ehr_tabular=0.44 \
                            --weights runtime/cvd/fit-report.json
```

Library form:

```python
from qhealth_qml.cardiovascular import CardiovascularFrame

frame = CardiovascularFrame(root="backend")
frame.readiness_report()["runnable_now"]     # ['ecg_12lead', 'ehr_tabular']
frame.fit_available(artifact_dir="runtime/cvd")
frame.pool({"ecg_12lead": 0.81, "ehr_tabular": 0.44, "angiography": None})
```

## 5. Design rules enforced in code

Not conventions — these are constraints a caller cannot bypass:

1. **Influence must be earned.** Weight is `2·(balanced_accuracy − 0.5)` from a
   modality's own **validation** split, never the test fold, never set by hand. There is
   no API to grant weight directly. A chance-level channel gets ~0 influence.
2. **Absent modalities are omitted, never imputed.** Missing channels drop out and the
   remaining weights renormalise. No zero-filling.
3. **Scores are threshold-aligned before averaging.** 0.42 is positive for a model
   thresholded at 0.30 and negative at 0.60; each score is remapped so its own threshold
   sits at 0.5.
4. **Pooled framing takes the weakest contributor.** A 4-year prediction pooled with a
   present-state ECG detection yields `"detection"`. A combined answer cannot claim more
   reach than its least forward-looking input.
5. **Polygenic scores refuse below coverage.** A PRS over a fraction of its variants is a
   *different* score, not a weaker one; published risk strata do not apply. Raises below
   80% by default.

## 6. Extension points

**Register a modality** — never edit the library:

```python
frame.register(ModalitySpec(
    name="ct_angiography",
    clinical_role="Coronary CT angiography",
    temporal_framing="detection",
    data_requirement="CCTA volumes plus a labels CSV",
    loader=my_loader,                    # (source, **opts) -> LoadedDataset
    default_source="data/ccta",
))
```

**Plug in an external scorer** (medical LLM on the EHR, vendor imaging model). It pools
on identical terms — but its influence is not automatic:

```python
frame.pool({"ehr_tabular": llm_probability, "ecg_12lead": 0.81})
```

A channel with no recorded validation score contributes at **zero weight**. To let it
count, register a `ModalityModel` carrying the balanced accuracy it *demonstrated* on a
held-out split. Measured, not asserted.

## 7. Honest limitations

Read this before presenting the work.

- **The quantum path has no demonstrated advantage.** Across everything measured in this
  project: **zero wins, one parity, six losses** against classical baselines. The frame
  is genuinely useful as a *multimodal pooling architecture with a quantum option* —
  that is the defensible claim. Do not present it as a proven quantum advantage.
- **The cohorts are disjoint.** PTB-XL's ECG patients are not MUSIC's heart-failure
  patients. Joint fusion and learned stacking both need per-patient alignment, so the
  combiner is a fixed rule. The pooled score is a defensible way to combine independent
  evidence; it is **not** a measured improvement over the best single modality, and the
  frame's own report says so.
- **Only one channel is prognostic.** `ehr_tabular` predicts over 4 years; `ecg_12lead`
  reads a present state. Never present a pooled score as early warning unless
  `temporal_framing` says `prediction`.
- **No angiography number exists.** The extractor is unit-tested against synthetic
  vessels and has never seen a real angiogram.
- **No genomics data exists.** The PRS engine is implemented and the published model is
  openly downloadable (PGS000018 metaGRS_CAD, 1,745,180 variants — verified HTTP 200,
  no auth). What is missing is individual-level genotypes.
- **Trained `.pkl` artifacts are deliberately not included.** Pickle executes arbitrary
  code on load, and the artifact schema is mid-migration. Ship the fit script and let
  the receiving side refit; `results/fit-report.json` carries the weights.

## 8. Next steps, in priority order

1. **CADICA angiography** (3.08 GB, openly downloadable) — `angiography.py` already
   consumes this format. Would produce the first cardiovascular *imaging* number.
2. **Extract ISLES-2022** (1.69 GB, already on disk, unzipped) — tests the
   frozen-pretrained-backbone hypothesis that `pretrained_encoder.py` was written for.
3. **Genotypes** for the PRS channel. Publicly, cardiac imaging + genotypes for the same
   people means UK Biobank (application + fee).
