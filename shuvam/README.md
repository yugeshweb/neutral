# Hybrid quantum-classical neurological models — handover package

Self-contained snapshot of the neuro-conditions work: five models over radiologist-grade data,
a deployable serving layer, and the research records behind every number.

**Research use only. Not a medical device. Nothing here is validated for clinical use.**

## Read this first — what actually works

| Model | Modality | Framing | Held-out | Usable? |
|---|---|---|---|---|
| `stroke-core-volume-mri` | MRI: DWI + ADC + FLAIR | characterisation | **BA 0.7765**, AUC 0.879 | Best available |
| `parkinsons-gait-signal` | 18-ch force-plate gait @100 Hz | detection | **BA 0.7980** (subject-grouped) | Best available |
| `ich-intraventricular-ct` | Head CT, 3 clinical windows | detection | BA ~0.54–0.58 | **At chance** |
| `glioma-mgmt-mpmri` | MRI: T1/T1c/T2/FLAIR | characterisation | BA 0.533 | **At chance (n=47)** |
| `alzheimers-t1-mri` | T1 structural MRI | screening | BA 0.43–0.61 | **At/below chance (n=54)** |
| `seizure-preictal-eeg` | 14 EEG band-power features | **prediction**, 5-min lead | **LOPO BA 0.505** | **At chance patient-independently** |

Four of six perform at chance. They are packaged so the integration surface is uniform and
testable, **not** because they work — each carries a `PERFORMS AT CHANCE` string returned in every
prediction response. In all cases the binding constraint is cohort size (47–192 subjects), which was
tested directly against three encoder architectures, not assumed.

## What this contains

```
src/        14 library modules — ingest, encoders, quantum heads, serving
runners/    11 entry points — training, bundling, registration, HTTP service
docs/       INTEGRATION.md (the contract) + pinned requirements
manifests/  model manifests (what each artifact expects and its measured performance)
research/   per-condition research records + result JSONs
test_serving_integration.py   12 acceptance tests
```

Model binaries (`*.pkl`, ~91 MB) are **not** included — they are build outputs, reproducible from
`runners/build_*.py`. The `*.manifest.json` files describe them exactly.

## The headline finding

Two conditions that failed six architecture-search levers on tabular features cleared the viability
gate immediately once the **input modality** changed to what a clinician actually reads — with the
quantum configuration untouched. P1 stroke went 0.549 (failed) → 0.810; P6 Parkinson's 0.561
(failed) → 0.750. Where cohorts were small (47–192), nothing helped: not modality, not resolution,
not architecture. **The lever is data, not circuits.** No quantum advantage is claimed anywhere —
classical heads stay level or ahead and confidence intervals overlap.

## Three bugs worth knowing about

1. **Score-normalisation clipping.** Raw `decision_function` output (unbounded, often negative) was
   clipped to [0,1], flattening negatives to zero and driving the threshold to 0.0 — an all-positive
   predictor. The model was fine (ROC-AUC 0.819); the deployment path destroyed it. Fixing it moved
   held-out balanced accuracy **0.500 → 0.7765** on identical data. Constants are now persisted in
   the artifact and regression-tested.
2. **Missing respacing (D5).** Volumes were resampled to a fixed array grid without first being
   respaced to a common physical voxel size, so scans at different mm/voxel occupied different real
   fields of view in the same tensor. Fixed via MONAI `Spacingd`. Use `src/cohort_audit.py` to check
   whether a cohort is exposed before treating any number measured on it as a reference value.
3. **Batch scoring rebuilt the encoder per sample** — 33M parameters reconstructed per study.

## Reproducing

```bash
pip install -r docs/requirements-serving.txt
python runners/serve_api.py --bundles runtime/bundles --port 8080
curl localhost:8080/models
```

See `docs/INTEGRATION.md` for the full response contract, the four fields a caller must handle,
and the real portability blockers (pickle format, `qhealth_qml` import, runtime weight download).

## Provenance

All datasets are open-licensed and were fetched without any data-use agreement: ISLES 2022
(CC BY 4.0), BHSD (MIT), UPenn-GBM via TCIA (CC BY 4.0), Zenodo 3935636 (CC BY 4.0), PhysioNet
gaitpdb and CHB-MIT (ODC-By). Gated sources (ADNI, PPMI, OASIS image tiers, PhysioNet ct-ich) were
identified and deliberately not used.
