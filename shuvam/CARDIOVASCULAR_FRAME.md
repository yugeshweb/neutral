# Cardiovascular frame — integration guide

A pluggable frame that declares the cardiovascular input channels, fits a detector for
each one that has data, and pools their calibrated opinions into a single risk
assessment. Built to be embedded: every entry point is a Python call, every output is
JSON, and nothing requires the CLI.

- Library: `src/qhealth_qml/cardiovascular.py`
- Genomics engine: `src/qhealth_qml/prs.py`
- Script: `run_cardiovascular.py`
- Tests: `tests/test_cardiovascular.py`, `tests/test_prs.py` (34 tests)

**Verified end-to-end.** `fit` was run against real data on disk; both ready modalities
trained and earned weights from their own validation splits:

| Modality | Data | Validation BA | Fusion weight | Test BA | Test AUROC |
|---|---|---|---|---|---|
| `ecg_12lead` | 6000 × 294, 2960 pos | 0.8486 | **0.6971** | 0.7380 | 0.8352 |
| `ehr_tabular` | 888 × 97, 185 pos | 0.6046 | **0.2092** | 0.6153 | 0.7896 |

These are from a `--max-train 600` screening run, not the full benchmark — the ECG
channel scores 0.8620 BA / 0.9374 AUROC at full scale. The point of the table is that
the weights are *earned and differentiated* (0.70 vs 0.21), not that these are the
headline numbers.

---

## 1. What it does

```
                 ┌─────────────┐
  12-lead ECG ──▶│  extractor  │──▶ features ──▶ detector ──▶ p, threshold ─┐
                 └─────────────┘                                            │
  EHR / labs ───▶ profile loader ──▶ features ──▶ detector ──▶ p, threshold ─┤
                                                                            ├─▶ skill-weighted
  angiography ──▶ vesselness ─────▶ features ──▶ detector ──▶ p, threshold ─┤    pooling ──▶ risk
                                                                            │
  genotypes ────▶ PGS score ──────▶ 1 feature ─▶ detector ──▶ p, threshold ─┤
                                                                            │
  anything else ──────────────── calibrated probability ────────────────────┘
```

The last row is the point of the design. Any external component that emits a calibrated
probability — a medical LLM reading the record, a vendor imaging model, a hospital's own
risk engine — participates on identical terms with the built-in modalities.

---

## 2. Current readiness

Run `python run_cardiovascular.py status` for the live answer. As of writing:

| Modality | State | Framing | Notes |
|---|---|---|---|
| `ecg_12lead` | **ready** | detection | PTB-XL on disk. Measured BA 0.8620 / AUROC 0.9374 |
| `ehr_tabular` | **ready** | prediction | MUSIC, 4-year cardiac death. Measured AUROC 0.768 |
| `angiography` | needs_data | detection | Extractor implemented + tested; no frames on disk |
| `cad_prs` | needs_data | screening | Engine implemented; no genotypes on disk |
| `echocardiography` | stub | characterisation | Declared, not implemented |

`needs_data` means **code-complete and starved**, not broken. `stub` means declared but
unimplemented. The report distinguishes these deliberately so an integrator can plan
against it rather than discovering failures by catching exceptions.

---

## 3. Embedding it

```python
from qhealth_qml.cardiovascular import CardiovascularFrame

frame = CardiovascularFrame(root="backend")

# 1. What can run right now? Touches no models, imports nothing heavy.
print(frame.readiness_report()["runnable_now"])
# ['ecg_12lead', 'ehr_tabular']

# 2. Fit everything with data. Starved modalities are skipped with a stated reason.
frame.fit_available(artifact_dir="runtime/cvd", max_train=600, max_test=300)

# 3. Pool scores for one patient. `None` = this patient has no such data.
frame.pool({
    "ecg_12lead":  0.81,
    "ehr_tabular": 0.44,
    "angiography": None,
})
```

Returns:

```json
{
  "probability": 0.63,
  "prediction": 1,
  "threshold": 0.5,
  "contributing_modalities": ["ecg_12lead", "ehr_tabular"],
  "missing_modalities": ["angiography"],
  "per_modality": [
    {"modality": "ecg_12lead", "score": 0.81, "weight": 0.72, "aligned_score": 0.81}
  ],
  "temporal_framing": "detection",
  "fusion": "skill-weighted mean of threshold-aligned calibrated probabilities"
}
```

### Adding your own modality

Registration is the supported extension point — you never edit the library:

```python
from qhealth_qml.cardiovascular import ModalitySpec

frame.register(ModalitySpec(
    name="ct_angiography",
    clinical_role="Coronary CT angiography — first-line for stable chest pain",
    temporal_framing="detection",          # prediction | screening | detection | characterisation
    data_requirement="CCTA volumes plus a labels CSV",
    loader=my_loader,                      # (source, **opts) -> LoadedDataset
    default_source="data/ccta",
))
```

### Plugging in an external scorer (e.g. a medical LLM on the EHR)

```python
llm_probability = my_llm.assess(record)          # must be calibrated
frame.pool({"ehr_tabular": llm_probability, "ecg_12lead": 0.81})
```

**Its influence is not automatic.** A modality with no recorded validation score
contributes at zero weight. To let an external scorer count, register a `ModalityModel`
carrying the balanced accuracy it demonstrated on a held-out split:

```python
from qhealth_qml.multimodal import ModalityModel

frame.trained["ehr_tabular"] = ModalityModel(
    modality="ehr_tabular",
    model_id="ehr_tabular:medgemma",
    artifact=None,
    validated_balanced_accuracy=0.71,   # must be measured, not asserted
)
```

---

## 4. Design rules that constrain callers

These are enforced in code, not conventions:

1. **Influence must be earned.** Fusion weight is `2·(balanced_accuracy − 0.5)`, taken
   from a modality's own **validation** split, never the test fold and never set by
   hand. A chance-level channel gets ~0 weight. There is no API to grant weight
   directly.

2. **Absent modalities are omitted, never imputed.** A missing channel drops out and the
   remaining weights renormalise. No zero-filling, no mean-filling.

3. **Scores are threshold-aligned before averaging.** A probability means nothing across
   models without its operating point: 0.42 is positive for a model thresholded at 0.30
   and negative at 0.60. Each score is remapped so its own threshold sits at 0.5.

4. **Pooled framing takes the weakest contributor.** Pooling a 4-year prediction with a
   present-state ECG detection yields `"detection"`, not `"prediction"`. A combined
   answer cannot claim more reach than its least forward-looking input.

5. **Polygenic scores refuse below coverage.** A PRS computed over a fraction of its
   variants is a *different* score, not a weaker one, and published risk strata do not
   apply to it. `apply_scoring_file` raises below `min_coverage` (default 0.80).

---

## 5. Genomics: the model is already published

A polygenic score is not trained here. The PGS Catalog distributes per-variant weights
fitted on cohorts far larger than anything this platform will hold, and they are openly
downloadable with no application:

```bash
curl -O https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000018/ScoringFiles/PGS000018.txt.gz
```

`PGS000018` (metaGRS_CAD, Inouye et al., *J Am Coll Cardiol* 2018) carries **1,745,180
variants**. Verified open: HTTP 200, 15.5 MB, no auth.

```python
from qhealth_qml.prs import read_scoring_file, read_dosage_table, apply_scoring_file

scoring = read_scoring_file("PGS000018.txt.gz")
samples, dosages, alleles = read_dosage_table("genotypes.csv")
result = apply_scoring_file(scoring, dosages, samples, reported_alleles=alleles)
print(result.coverage, result.flipped_variants, result.warnings)
```

Dosage CSV format — column headers may declare the allele pair, which is what makes
strand resolution possible:

```csv
sample_id,rs2843152:G_C,rs35465346:G_A
P001,2,1
P002,,0
```

Blank means **unmeasured**, not homozygous reference.

**What's missing is genotypes, not the model.** No CVD dataset in this repo carries
individual-level genotypes; publicly, the only cohort with both cardiac imaging and
genotypes for the same people is UK Biobank (application + fee).

---

## 6. Honest limitations

- **The cohorts are disjoint.** PTB-XL's ECG patients are not MUSIC's heart-failure
  patients. Joint fusion and learned stacking both need per-patient alignment, so the
  combiner is a fixed rule. The pooled score is a defensible way to combine independent
  evidence; it is **not** a measured improvement over the best single modality, and the
  frame's own report says so.
- **Only one channel is genuinely prognostic.** `ehr_tabular` predicts over a 4-year
  horizon. `ecg_12lead` reads a state already present. Do not present the pooled score
  as early warning unless `temporal_framing` says `prediction`.
- **`cad_prs` is marked not quantum-capable** — one feature per patient gives a quantum
  feature map nothing to entangle.
- **No angiography number exists.** The extractor is unit-tested against synthetic
  vessels; it has never seen a real angiogram.

---

## 7. CLI reference

Installed as a console script (`pip install ./backend`):

```bash
qhealth-cardiovascular status
qhealth-cardiovascular fit  --artifact-dir runtime/cvd --max-train 600 --bootstrap 200
qhealth-cardiovascular pool --score ecg_12lead=0.81 --score ehr_tabular=0.44 \
                            --weights runtime/cvd/fit-report.json
```

From a source checkout without installing, `python run_cardiovascular.py <same args>` is
an equivalent shim. Paths resolve against `--root`, which defaults to the working
directory for the installed script.

Override data locations without touching code:

```bash
python run_cardiovascular.py status --source ecg_12lead=/mnt/data/ptb-xl
python run_cardiovascular.py fit --source angiography=data/cadica \
                                 --option angiography.labels=labels.csv
```

Options are accepted **after** the subcommand.
