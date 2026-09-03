# Integrating the hybrid imaging and signal models

What this is: research-grade hybrid quantum-classical models over medical imaging, biosignals, and clinical tabular data, packaged so another system can call them. **Not a medical device. Research use only.**

## Honest status first

Read this before planning an integration — it decides whether this is usable for your case.

| Dimension | Platform Breakdown |
|---|---|
| **Registered Disease Domains** | **7 total domains** registered across the platform |
| **Deployable Model Bundles** | **5 deployable bundles** (`stroke`, `parkinsons`, `ich`, `glioma`, `alzheimers`) |
| **Active Training Pipelines** | **7 full training pipelines** (including `heart-disease` 12-lead ECG and `breast-cancer` WDBC) |
| **Demonstrated Ability** | `stroke-core-volume-mri` (BA 0.7765) and `parkinsons-gait-signal` (BA 0.7980). |
| **At Chance / Underpowered** | `ich-intraventricular-ct` (BA ~0.54) and `glioma-mgmt-mpmri` (BA 0.533, n=47). Packaged for uniform interface testing with explicit `PERFORMS AT CHANCE` disclaimer strings. |
| **Explicitly Disabled** | `seizure-preictal-eeg` (LOPO BA 0.505 ± 0.257) — **Gated at API level** to prevent clinical alerting. |
| **Score Semantics** | **`normalised_margin_from_threshold`** (NOT a probability of disease). |
| **Artifact Format** | **Joblib serialization** (`.joblib` / `.bundle`) with companion `.manifest.json`. |
| **External validation** | **None.** Every number is an internal split of one public cohort. |

---

### Per-model status & clinical scoping

| Model id | Modality | Framing | Held-out | Usable? |
|---|---|---|---|---|
| `stroke-core-volume-mri` | MRI: DWI+ADC+FLAIR | characterisation | **BA 0.7765**, AUC 0.879 | Best available |
| `parkinsons-gait-signal` | 18-ch force-plate gait, 100 Hz | detection | **BA 0.7980** (subject-grouped) | Best available |
| `ich-intraventricular-ct` | Head CT, 3 clinical windows | detection | BA ~0.54–0.58, CI spans 0.5 | **At chance — no discriminative ability** |
| `glioma-mgmt-mpmri` | MRI: T1/T1c/T2/FLAIR | characterisation | BA 0.533, CI [0.224, 0.843] | **At chance — n=47, underpowered** |
| `alzheimers-oasis-tabular` | Tabular + Volumetric measures | screening | Classical BA 0.823, QSVC BA 0.564 | **Screening / Tabular reference** |
| `seizure-preictal-eeg` | 14 EEG band-power features | **prediction** (5 min lead) | **LOPO BA 0.505 ± 0.257** | **DISABLED AT API LEVEL (Patient Safety)** |

> [!IMPORTANT]
> **Score Semantics**: `score` is a **normalised distance from threshold** (`score_semantics: "normalised_margin_from_threshold"`), NOT a posterior probability of disease. Calibration has not been assessed.

---

## Quickest path: HTTP service

```bash
pip install -r requirements-serving.txt
python serve_api.py --bundles runtime/bundles --port 8080
```

### 1. Model Catalogue
```bash
curl localhost:8080/models
```

### 2. Stroke DWI+ADC+FLAIR MRI Prediction
```bash
curl -X POST localhost:8080/predict/stroke-core-volume-mri \
  -H 'content-type: application/json' \
  -d '{"study_id":"CASE-STROKE-1","sources":{"dwi":"/data/dwi.nii.gz","adc":"/data/adc.nii.gz","flair":"/data/flair.nii.gz"}}'
```

### 3. Alzheimer's Tabular / Volumetric Prediction (`alzheimers-oasis-tabular`)
```bash
curl -X POST localhost:8080/predict/alzheimers-oasis-tabular \
  -H 'content-type: application/json' \
  -d '{"study_id":"CASE-AD-1","sources":{"MMSE":"21","nWBV":"0.695","Age":"82","eTIV":"1450","ASF":"1.21","Educ":"2","SES":"3","M_F":"1"}}'
```

### 4. Direct Multi-Part File Upload
```bash
curl -X POST localhost:8080/predict/stroke-core-volume-mri/upload \
  -F files=@dwi.nii.gz -F files=@adc.nii.gz -F files=@flair.nii.gz
```

---

## In-process Python API

```python
from qhealth_qml.serving import load_bundle, predict
from qhealth_qml.ingestion import ingest_study

bundle = load_bundle("runtime/bundles/stroke-core-volume-mri.joblib")
ingested = ingest_study(bundle.to_manifest(), {"dwi": "dwi.nii.gz", "adc": "adc.nii.gz", "flair": "flair.nii.gz"})
if ingested.ok:
    result = predict(bundle, ingested.volume, study_id="CASE-1")
```

---

## The Response Contract

```json
{
  "status": "ok",
  "label": "small_infarct_core",
  "prediction": 0,
  "score": 0.109,
  "score_semantics": "normalised_margin_from_threshold",
  "threshold": 0.305,
  "decision_margin": 0.196,
  "review_recommended": false,
  "temporal_framing": "characterisation",
  "interpretation": "Model characterises an already-identified finding as small_infarct_core...",
  "limitations": ["Cohort is single-center retrospective..."],
  "research_use_only": true
}
```

### Key Contract Guarantees:
1. **`status`**: Can be `"ok"`, `"rejected"` (missing/incompatible sequence), or `"disabled"` (gated for patient safety).
2. **`temporal_framing`**: One of `prediction` (early warning with lead time), `detection` (present finding), `characterisation` (property of finding), or `screening`.
3. **`score_semantics`**: Explicitly set to `"normalised_margin_from_threshold"`.
4. **`review_recommended`**: `true` when score is near threshold within uncertainty boundary.

---

## Verifying an Install

```bash
python -m pytest tests/test_serving_integration.py -q
python serve_api.py --bundles runtime/bundles --check-only
```
