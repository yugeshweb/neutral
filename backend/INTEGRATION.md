# Integrating the hybrid imaging models

What this is: research-grade hybrid quantum-classical models over medical imaging, packaged so
another system can call them. **Not a medical device. Research use only.**

## Honest status first

Read this before planning an integration — it decides whether this is usable for your case.

| | State |
|---|---|
| Deployable artifacts | **5 of 5 conditions** — all callable through the same interface |
| **Only two have demonstrated ability** | `stroke-core-volume-mri` (BA 0.7765) and `parkinsons-gait-signal` (BA 0.7980). The other three **perform at chance** and are deployed for integration purposes only. |
| External validation | **None.** Every number is an internal split of one public cohort. |
| Calibration | **Not assessed.** The score is a thresholded decision value, not a probability of disease. |
| Portability | Artifact is a **Python pickle** — see "Portability limits" below. This is the weakest part. |
| Regulatory | Nothing here approaches SaMD requirements. |

### Per-model status — read this before choosing one

| Model id | Modality | Framing | Held-out | Usable? |
|---|---|---|---|---|
| `stroke-core-volume-mri` | MRI: DWI+ADC+FLAIR | characterisation | **BA 0.7765**, AUC 0.879 | Best available |
| `parkinsons-gait-signal` | 18-ch force-plate gait, 100 Hz | detection | **BA 0.7980** (subject-grouped) | Best available |
| `ich-intraventricular-ct` | Head CT, 3 clinical windows | detection | BA ~0.54–0.58, CI spans 0.5 | **At chance — no discriminative ability** |
| `glioma-mgmt-mpmri` | MRI: T1/T1c/T2/FLAIR | characterisation | BA 0.533, CI [0.224, 0.843] | **At chance — n=47, underpowered** |
| `alzheimers-t1-mri` | T1 structural MRI | screening | BA 0.43–0.61, CIs span 0.5 | **At/below chance — n=54** |
| `seizure-preictal-eeg` | 14 EEG band-power features | **prediction** (5 min lead) | **LOPO BA 0.505 ± 0.257**, AUC 0.468 | **At chance patient-independently** |

The three chance-level models are packaged so the integration surface is uniform and testable, not
because they work. Each carries its own `PERFORMS AT CHANCE` limitation string in the artifact, and
that text is returned in every prediction response. **Do not build a decision on them.** In all
three cases the binding constraint is cohort size (47–192 subjects), not the architecture — that was
tested directly and is documented in the per-condition research records.

`seizure-preictal-eeg` deserves a specific warning: its *threshold* patient scored 0.93, but
leave-one-patient-out is 0.505, and two of four held-out patients scored **below** chance with
inverted ranking. It must not be used to warn or alert anyone.

## Quickest path: HTTP service

```bash
pip install -r requirements-serving.txt
python serve_api.py --bundles runtime/bundles --port 8080
```

```bash
curl localhost:8080/models                      # catalogue + limitations + held-out numbers
curl -X POST localhost:8080/predict/stroke-core-volume-mri \
  -H 'content-type: application/json' \
  -d '{"study_id":"CASE-1","sources":{"dwi":"/data/dwi.nii.gz","adc":"/data/adc.nii.gz","flair":"/data/flair.nii.gz"}}'
```

Or upload the files directly (filenames must contain the channel name):

```bash
curl -X POST localhost:8080/predict/stroke-core-volume-mri/upload \
  -F files=@dwi.nii.gz -F files=@adc.nii.gz -F files=@flair.nii.gz
```

## In-process (Python)

```python
from qhealth_qml.ingestion import ingest_and_predict

result = ingest_and_predict(
    "runtime/bundles/stroke-core-volume-mri.pkl",
    {"dwi": "dwi.nii.gz", "adc": "adc.nii.gz", "flair": "flair.nii.gz"},
    study_id="CASE-1",
)
```

DICOM series work too — pass the *directory* instead of a file; rescale slope/intercept are
applied, so CT arrives in Hounsfield units.

## The response contract

```json
{
  "status": "ok",
  "label": "small_infarct_core",
  "prediction": 0,
  "score": 0.109,
  "threshold": 0.305,
  "decision_margin": 0.196,
  "review_recommended": false,
  "temporal_framing": "characterisation",
  "interpretation": "Model characterises an already-identified finding as ...",
  "limitations": ["..."],
  "research_use_only": true
}
```

Four fields a caller **must** handle rather than ignore:

* **`status`** — `"rejected"` is a normal outcome, returned with HTTP 200 and machine-readable
  `reasons`. It is not a server error. A study with a missing sequence is refused rather than
  scored on a zero-filled channel, because a confident answer from imputed data is worse than
  no answer.
* **`temporal_framing`** — one of `prediction` (early warning, carries lead time), `detection`
  (event present now), `characterisation` (property of an already-identified finding),
  `screening`. The stroke model is **characterisation**: it does not detect whether a stroke is
  present, it describes an infarct already identified. Rendering it as an early-warning score
  would misrepresent it.
* **`review_recommended`** — true when the score sits within the abstention margin of the
  threshold. Treat as "insufficient separation to act on".
* **`research_use_only`** — always true. There is no configuration in which this is false.

`score` is *not* a probability of disease. It is a normalised distance from a
validation-selected operating threshold, and calibration has not been assessed.

## Portability limits (the real blockers)

Be aware of these before depending on it:

1. **The artifact is a Python pickle.** It must be loaded with compatible versions of torch,
   scikit-learn and qiskit — `requirements-serving.txt` pins the exact set it was built with.
   Loading a pickle also executes code, so treat bundles as trusted binaries, not as data:
   only load artifacts you produced or received over a trusted channel.
2. **It imports `qhealth_qml`.** The artifact is not self-contained; the consumer needs this
   package importable. The HTTP service is the way to avoid that — run it here, call it from
   anywhere.
3. **First load fetches weights.** The MedicalNet-backed encoder downloads from HuggingFace on
   first construction. Pre-warm the cache (or pin `HF_HOME`) before running offline.
4. **CPU-only as pinned.** `torch==2.14.0+cpu`. Swap for a CUDA build if you have a GPU;
   nothing in the serving path assumes CPU.

If you need a genuinely self-contained artifact, the encoder should be exported to ONNX and the
head re-expressed without pickling — that work is not done.

## Preprocessing is owned by the artifact, not the caller

Grid size, channel order, orientation, spacing and CT windowing all come from the bundle
manifest. A caller cannot accidentally feed the wrong geometry, and the pipeline is fixed as:

    load → HU rescale → orient (RAS) → respace → intensity → grid → stack

The respace step matters: resampling straight to a grid normalises voxel *counts* but not
physical extent, so scans acquired at different mm/voxel would otherwise occupy different real
fields of view in the same tensor. Use `qhealth_qml.cohort_audit` to check whether a cohort is
spacing-homogeneous before treating any number measured on it as a reference value.

## Verifying an install

```bash
python -m pytest tests/test_serving_integration.py -q   # 12 acceptance tests
python serve_api.py --bundles runtime/bundles --check-only
```

The acceptance tests pin the properties an integration depends on: bundle round-trip, batch
scoring identical to single scoring, preprocessing replayed rather than refitted, junk and
incomplete studies refused with reasons, and temporal framing present in every response.
