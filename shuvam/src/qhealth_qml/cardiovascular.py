"""A pluggable cardiovascular risk frame: declare modalities, fit what has data, pool the rest.

This is the integration surface for cardiovascular disease. The platform already has
the pieces -- an ECG feature extractor, an angiography extractor, a tabular EHR path,
a polygenic score engine, a hybrid quantum head, and a skill-weighted combiner -- but
they were only ever reachable as separate scripts with incompatible entry points.
Nothing outside this repo could call them, and nothing inside it could ask "what can
we actually run right now?".

The frame answers both. A `ModalitySpec` declares one clinical input channel: what it
is, what a doctor uses it for, whether it predicts the future or reads the present,
what data it needs, and how to turn that data into a feature matrix. Specs are
registered, not hard-coded, so an integrator adds a modality by registering one rather
than editing this file.

Three properties make it embeddable rather than merely runnable:

* **Readiness is inspectable without running anything.** `readiness_report()` says which
  modalities have data on disk, which are code-complete but starved, and which are
  stubs -- so a caller can decide what to invoke instead of discovering failures by
  catching exceptions. Most cardiovascular modalities in this repo are currently
  code-complete and starved, and the report says so rather than implying coverage.

* **A modality's influence must be earned.** Fusion weight comes from a modality's own
  *validation* balanced accuracy via the existing combiner, never from the test fold
  and never set by hand. A modality that cannot demonstrate skill cannot acquire a
  voice, and one whose data is absent is omitted rather than imputed.

* **Every output is JSON-serialisable and states its own framing.** A pooled result
  carries which modalities contributed, which were missing, each one's weight, and
  whether the answer is a prediction over a horizon or a detection of a present state.
  A downstream UI cannot render a concurrent detection as an early warning by accident.

The honest limitation, stated here because it shapes what the numbers mean: the
cardiovascular cohorts available to this platform are *disjoint*. PTB-XL's ECG patients
are not MUSIC's heart-failure patients are not any angiography cohort. Joint fusion and
learned stacking both need per-patient alignment, so the combiner is a fixed rule, and
the fused score is a defensible way to pool independent evidence -- not a measured
improvement over the best single modality. `paired` results are only meaningful when a
cohort genuinely carries multiple modalities for the same people.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .multimodal import ModalityEvidence, ModalityModel, fuse_modalities, skill_weight

# What a modality's answer actually means in time. Mirrors serving.TEMPORAL_FRAMINGS
# so a result crossing between the two subsystems keeps its meaning.
TEMPORAL_FRAMINGS = frozenset({"prediction", "detection", "characterisation", "screening"})

READINESS_STATES = frozenset({"ready", "needs_data", "needs_dependency", "stub"})

FRAME_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModalitySpec:
    """One clinical input channel: what it is, what it needs, how to featurise it.

    `loader` takes a source path (whatever that modality's data looks like) and returns
    a `LoadedDataset`. It is deliberately not called during registration or readiness
    checks -- declaring a modality must stay cheap enough that a caller can enumerate
    every one of them without touching disk or importing torch.
    """

    name: str
    clinical_role: str
    temporal_framing: str
    data_requirement: str
    loader: Callable[..., Any] | None = None
    default_source: str | None = None
    default_model: str = "logistic_regression"
    quantum_capable: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.temporal_framing not in TEMPORAL_FRAMINGS:
            raise ValueError(
                f"{self.name}: temporal_framing must be one of "
                f"{sorted(TEMPORAL_FRAMINGS)}, got {self.temporal_framing!r}"
            )
        if not self.name or not self.name.strip():
            raise ValueError("a modality spec needs a name")

    def resolved_source(self, root: Path, override: str | None = None) -> Path | None:
        """Where this modality's data would live, if anywhere."""

        candidate = override if override is not None else self.default_source
        if candidate is None:
            return None
        path = Path(candidate)
        return path if path.is_absolute() else (root / path)

    def readiness(self, root: Path, override: str | None = None) -> dict[str, Any]:
        """Can this modality run right now, and if not, what is missing?"""

        if self.loader is None:
            return {
                "modality": self.name,
                "state": "stub",
                "reason": "no loader registered; this modality is declared but not implemented",
                "source": None,
            }
        source = self.resolved_source(root, override)
        if source is None:
            return {
                "modality": self.name,
                "state": "needs_data",
                "reason": "no data source configured for this modality",
                "source": None,
            }
        if not source.exists():
            return {
                "modality": self.name,
                "state": "needs_data",
                "reason": f"data source does not exist: {source}",
                "source": str(source),
            }
        return {
            "modality": self.name,
            "state": "ready",
            "reason": "loader registered and data source present",
            "source": str(source),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "modality": self.name,
            "clinical_role": self.clinical_role,
            "temporal_framing": self.temporal_framing,
            "data_requirement": self.data_requirement,
            "default_model": self.default_model,
            "quantum_capable": self.quantum_capable,
            "implemented": self.loader is not None,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# Loaders. Each is a thin adapter onto an existing extractor, kept lazy so that importing
# this module never pulls in torch, wfdb or qiskit.
# --------------------------------------------------------------------------------------


def _load_ecg_12lead(source: str | Path, max_records: int = 6000, seed: int = 7, **_: Any) -> Any:
    """PTB-XL 12-lead ECG -> myocardial-infarction detection features."""

    from .ecg import load_ptbxl_ecg_dataset

    root = Path(source)
    cache = root / "cache" / f"ptbxl-mi-{max_records}.npz"
    return load_ptbxl_ecg_dataset(
        root / "ptbxl_database.csv",
        root / "scp_statements.csv",
        root,
        target="mi",
        max_records=max_records,
        seed=seed,
        cache_path=cache,
    )


def _load_ehr_tabular(source: str | Path, **_: Any) -> Any:
    """MUSIC heart-failure cohort -> 4-year cardiac-death risk over a real horizon."""

    from .experiment import load_profile_dataset

    return load_profile_dataset(str(source))


def _load_angiography(source: str | Path, labels: str | Path | None = None, **_: Any) -> Any:
    """Coronary angiography frames -> multi-scale vesselness features.

    `labels` is a CSV of `frame,label` pairs. It is required: an imaging modality with
    no labels can produce features but cannot produce a detector, and silently
    fabricating labels is the failure this refusal prevents.
    """

    import csv

    from .angiography import extract_frames_dataset

    if labels is None:
        raise ValueError(
            "angiography needs a labels CSV (columns: frame,label); frames alone "
            "yield features but no trainable detector"
        )
    root = Path(source)
    frame_paths: list[Path] = []
    values: list[int] = []
    with Path(labels).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = str(row.get("frame", "")).strip()
            label = str(row.get("label", "")).strip()
            if not frame or not label:
                continue
            path = Path(frame)
            frame_paths.append(path if path.is_absolute() else root / path)
            values.append(int(float(label)))
    if not frame_paths:
        raise ValueError(f"no labelled frames read from {labels}")
    return extract_frames_dataset(frame_paths, values)


def _load_cad_prs(
    source: str | Path,
    dosages: str | Path | None = None,
    outcomes: str | Path | None = None,
    min_coverage: float | None = None,
    **_: Any,
) -> Any:
    """PGS Catalog CAD score applied to genotype dosages -> a one-column risk feature.

    `source` is the scoring file, `dosages` the per-sample genotype table, `outcomes`
    a CSV of `sample_id,label`. A polygenic score is a published model, so nothing is
    fitted here; the label is needed only so the combiner can measure how much to
    trust this channel.
    """

    import csv

    from .experiment import LoadedDataset
    from .prs import DEFAULT_MIN_COVERAGE, apply_scoring_file, prs_feature_matrix
    from .prs import read_dosage_table, read_scoring_file

    if dosages is None:
        raise ValueError("cad_prs needs a genotype dosage table; weights alone score nothing")
    if outcomes is None:
        raise ValueError(
            "cad_prs needs an outcomes CSV (sample_id,label) so the combiner can "
            "establish how far to trust genomic evidence"
        )

    scoring = read_scoring_file(source)
    sample_ids, dosage_map, alleles = read_dosage_table(dosages)
    result = apply_scoring_file(
        scoring,
        dosage_map,
        sample_ids,
        reported_alleles=alleles or None,
        min_coverage=DEFAULT_MIN_COVERAGE if min_coverage is None else float(min_coverage),
    )

    labels: dict[str, int] = {}
    with Path(outcomes).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identifier = str(row.get("sample_id", "")).strip()
            label = str(row.get("label", "")).strip()
            if identifier and label:
                labels[identifier] = int(float(label))

    keep = [index for index, sample in enumerate(result.sample_ids) if sample in labels]
    if not keep:
        raise ValueError("no genotyped sample appears in the outcomes CSV")

    X_all, names = prs_feature_matrix(result, standardised=True)
    X = X_all[keep, :]
    y = np.asarray([labels[result.sample_ids[index]] for index in keep], dtype=int)
    row_ids = np.asarray([result.sample_ids[index] for index in keep], dtype=str)

    return LoadedDataset(
        name=f"cad-prs-{scoring.pgs_id}",
        X=X,
        y=y,
        feature_names=list(names),
        positive_label="coronary_artery_disease",
        negative_label="no_coronary_artery_disease",
        provenance={
            "source": "PGS Catalog polygenic score applied to genotype dosages",
            "scoring_file": scoring.as_dict(),
            "application": result.as_dict(),
        },
        row_ids=row_ids,
    )


# --------------------------------------------------------------------------------------
# The cardiovascular registry. Ordered by how directly a clinician relies on the channel.
# --------------------------------------------------------------------------------------

CARDIOVASCULAR_MODALITIES: tuple[ModalitySpec, ...] = (
    ModalitySpec(
        name="ecg_12lead",
        clinical_role=(
            "Resting 12-lead ECG. First-line test for suspected myocardial infarction "
            "and the only modality here that is universally available at the point of care."
        ),
        temporal_framing="detection",
        data_requirement="PTB-XL WFDB records plus ptbxl_database.csv and scp_statements.csv",
        loader=_load_ecg_12lead,
        default_source="data/ptb-xl/1.0.3",
        default_model="logistic_regression",
        notes=(
            "Measured: balanced accuracy 0.8620, AUROC 0.9374. Reads a state already "
            "present; it is not an early warning."
        ),
    ),
    ModalitySpec(
        name="ehr_tabular",
        clinical_role=(
            "Structured clinical record: demographics, NYHA class, vitals, laboratory "
            "biomarkers (Pro-BNP, troponin, creatinine), echocardiographic measurements, "
            "chest-radiograph indices, ECG intervals, Holter/HRV summaries and medications."
        ),
        temporal_framing="prediction",
        data_requirement="A profile JSON whose cohort declares a horizon and index/outcome times",
        loader=_load_ehr_tabular,
        default_source="profiles/music_cardiac_death.json",
        default_model="logistic_regression",
        notes=(
            "Measured: AUROC 0.768 for cardiac death within 4 years. The only genuinely "
            "prognostic channel in this frame."
        ),
    ),
    ModalitySpec(
        name="angiography",
        clinical_role=(
            "X-ray coronary angiography. The clinical reference standard for coronary "
            "stenosis; a lesion's calibre is read directly from the contrast-filled tree."
        ),
        temporal_framing="detection",
        data_requirement="A directory of angiography frames plus a labels CSV (frame,label)",
        loader=_load_angiography,
        default_source=None,
        default_model="logistic_regression",
        notes=(
            "Extractor implemented and unit-tested (multi-scale Hessian vesselness); no "
            "frames on disk, so this modality has never produced a number. CADICA is "
            "openly downloadable and matches this loader's expectations."
        ),
    ),
    ModalitySpec(
        name="cad_prs",
        clinical_role=(
            "Genomic risk. A published coronary-artery-disease polygenic score, which is "
            "fixed at birth and therefore contributes risk context no measurement can date."
        ),
        temporal_framing="screening",
        data_requirement=(
            "A PGS Catalog scoring file, a per-sample genotype dosage table, and outcomes"
        ),
        loader=_load_cad_prs,
        default_source=None,
        default_model="logistic_regression",
        quantum_capable=False,
        notes=(
            "The model is published and openly downloadable (PGS000018 metaGRS_CAD, "
            "1,745,180 variants) so nothing is fitted; individual-level genotypes are "
            "what this platform lacks. One feature per patient, so a quantum feature map "
            "has nothing to entangle -- marked not quantum-capable deliberately."
        ),
    ),
    ModalitySpec(
        name="echocardiography",
        clinical_role=(
            "Transthoracic echocardiography. Source of LVEF, the single strongest "
            "prognostic measurement in heart failure, plus chamber and valve assessment."
        ),
        temporal_framing="characterisation",
        data_requirement="Echo video volumes; no loader implemented in this repo",
        loader=None,
        default_source=None,
        notes=(
            "Declared, not implemented. EchoNet-Dynamic publishes both data and pretrained "
            "weights (LVEF mean absolute error 4.1%), so this is the cheapest imaging "
            "modality to add next; MUSIC already carries LVEF as a tabular feature."
        ),
    ),
)


def modality_registry(
    extra: Iterable[ModalitySpec] = (),
) -> dict[str, ModalitySpec]:
    """The cardiovascular modality registry, plus any caller-supplied specs.

    Registration rather than editing this module is the supported extension point: an
    integrator with their own CT-angiography or wearable channel supplies a
    `ModalitySpec` and the rest of the frame treats it identically to a built-in.
    """

    registry: dict[str, ModalitySpec] = {spec.name: spec for spec in CARDIOVASCULAR_MODALITIES}
    for spec in extra:
        if not isinstance(spec, ModalitySpec):
            raise TypeError("extra modalities must be ModalitySpec instances")
        registry[spec.name] = spec
    return registry


@dataclass
class CardiovascularFrame:
    """Assembles cardiovascular modalities into one poolable risk assessment."""

    root: Path
    registry: dict[str, ModalitySpec] = field(default_factory=modality_registry)
    sources: dict[str, str] = field(default_factory=dict)
    loader_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    trained: dict[str, ModalityModel] = field(default_factory=dict)
    training_reports: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- inspection ---------------------------------------------------------------

    def readiness_report(self) -> dict[str, Any]:
        """What can run right now, without importing heavy deps or touching a model."""

        entries = [
            {
                **spec.as_dict(),
                **spec.readiness(self.root, self.sources.get(name)),
            }
            for name, spec in self.registry.items()
        ]
        by_state: dict[str, list[str]] = {}
        for entry in entries:
            by_state.setdefault(entry["state"], []).append(entry["modality"])
        return {
            "schema_version": FRAME_SCHEMA_VERSION,
            "condition": "cardiovascular",
            "root": str(self.root),
            "modalities": entries,
            "summary": {state: sorted(names) for state, names in sorted(by_state.items())},
            "runnable_now": sorted(
                entry["modality"] for entry in entries if entry["state"] == "ready"
            ),
            "note": (
                "A modality reported 'needs_data' is code-complete and starved, not "
                "broken. 'stub' means declared but unimplemented."
            ),
        }

    def available(self) -> list[str]:
        return list(self.readiness_report()["runnable_now"])

    # -- fitting ------------------------------------------------------------------

    def load(self, modality: str) -> Any:
        """Load one modality's dataset, raising with a specific reason if it cannot."""

        spec = self.registry.get(modality)
        if spec is None:
            raise KeyError(f"unknown modality {modality!r}; known: {sorted(self.registry)}")
        if spec.loader is None:
            raise ValueError(f"{modality} is declared but has no loader implemented")
        source = spec.resolved_source(self.root, self.sources.get(modality))
        if source is None:
            raise ValueError(f"{modality} has no data source configured")
        if not source.exists():
            raise FileNotFoundError(f"{modality} data source does not exist: {source}")
        options = dict(self.loader_options.get(modality, {}))
        return spec.loader(source, **options)

    def fit_modality(
        self,
        modality: str,
        artifact_dir: str | Path,
        model: str | None = None,
        **run_kwargs: Any,
    ) -> ModalityModel:
        """Train one modality's detector and record the weight it earned.

        The weight comes from the combiner's own `train_modality_model`, which reads
        *validation* balanced accuracy. Nothing here can hand a modality influence.
        """

        from .multimodal import train_modality_model

        spec = self.registry[modality]
        dataset = self.load(modality)
        artifact_path = Path(artifact_dir) / f"cvd-{modality}.pkl"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        model_name = model or spec.default_model
        defaults: dict[str, Any] = {
            "validation_size": 0.2,
            "threshold_policy": "max_balanced_accuracy",
            "seed": 7,
        }
        defaults.update(run_kwargs)

        trained, report = train_modality_model(
            modality=modality,
            model_id=f"{modality}:{model_name}",
            dataset=dataset,
            artifact_path=artifact_path,
            models=(model_name,),
            **defaults,
        )
        self.trained[modality] = trained
        self.training_reports[modality] = {
            "modality": modality,
            "model": model_name,
            "temporal_framing": spec.temporal_framing,
            "validated_balanced_accuracy": trained.validated_balanced_accuracy,
            "fusion_weight": trained.weight,
            "artifact": str(artifact_path),
            "dataset": {
                "name": getattr(dataset, "name", modality),
                "rows": int(np.asarray(dataset.X).shape[0]),
                "features": int(np.asarray(dataset.X).shape[1]),
                "positives": int(np.asarray(dataset.y).sum()),
            },
            "test_metrics": report["models"][model_name].get("metrics", {}),
        }
        return trained

    def fit_available(
        self,
        artifact_dir: str | Path,
        models: Mapping[str, str] | None = None,
        skip_errors: bool = True,
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        """Fit every modality that has data, reporting rather than raising on failures."""

        chosen = dict(models or {})
        outcomes: dict[str, Any] = {"fitted": [], "skipped": []}
        for modality in self.available():
            try:
                self.fit_modality(
                    modality,
                    artifact_dir,
                    model=chosen.get(modality),
                    **run_kwargs,
                )
                outcomes["fitted"].append(modality)
            except Exception as exc:  # a starved or broken modality must not sink the frame
                if not skip_errors:
                    raise
                outcomes["skipped"].append({"modality": modality, "reason": str(exc)[:300]})
        return outcomes

    # -- scoring ------------------------------------------------------------------

    def pool(
        self,
        scores: Mapping[str, float | None],
        threshold: float = 0.5,
        thresholds: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Pool per-modality scores into one cardiovascular risk assessment.

        `scores` maps a modality to its calibrated probability, or to `None` when that
        modality is absent for this patient. Absent modalities are dropped and the
        remaining weights renormalised -- never imputed. Scores may come from this
        frame's own models or from anything else that emits a calibrated probability,
        which is what lets an external component (a medical LLM reading the record, a
        vendor imaging model) participate on equal terms.
        """

        per_modality_thresholds = dict(thresholds or {})
        evidence: list[ModalityEvidence] = []
        unknown = [name for name in scores if name not in self.registry]
        if unknown:
            raise KeyError(f"scores reference unknown modalities: {sorted(unknown)}")

        for modality, score in scores.items():
            trained = self.trained.get(modality)
            if trained is not None:
                weight = trained.weight
                model_id = trained.model_id
            else:
                # An external contributor must still declare demonstrated skill; with
                # none recorded it contributes at zero weight rather than by assertion.
                weight = 0.0
                model_id = f"{modality}:external"
            evidence.append(
                ModalityEvidence(
                    modality=modality,
                    model_id=model_id,
                    score=None if score is None else float(score),
                    threshold=float(per_modality_thresholds.get(modality, 0.5)),
                    weight=weight,
                )
            )

        fused = fuse_modalities(evidence, threshold=threshold)
        payload = fused.as_dict()
        payload["temporal_framing"] = self._pooled_framing(fused.contributing)
        payload["schema_version"] = FRAME_SCHEMA_VERSION
        return payload

    def _pooled_framing(self, contributing: Sequence[str]) -> str:
        """The weakest framing among contributors, so nothing is over-claimed.

        Pooling a 4-year risk prediction with a present-state ECG detection does not
        yield a prediction: the combined answer can only claim what its weakest
        contributor supports. Ordering runs prediction -> screening -> detection ->
        characterisation, and the pooled result takes the least forward-looking one.
        """

        order = ["prediction", "screening", "detection", "characterisation"]
        present = [
            self.registry[name].temporal_framing
            for name in contributing
            if name in self.registry
        ]
        if not present:
            return "detection"
        return max(present, key=lambda framing: order.index(framing))

    def register(self, spec: ModalitySpec) -> None:
        """Add or replace a modality. The supported way to extend the frame."""

        if not isinstance(spec, ModalitySpec):
            raise TypeError("register expects a ModalitySpec")
        self.registry[spec.name] = spec

    # -- reporting ----------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Everything an integrator needs: readiness, what was fitted, what it earned."""

        weights = {
            name: {
                "validated_balanced_accuracy": model.validated_balanced_accuracy,
                "fusion_weight": model.weight,
                "model_id": model.model_id,
            }
            for name, model in self.trained.items()
        }
        return {
            "schema_version": FRAME_SCHEMA_VERSION,
            "condition": "cardiovascular",
            "readiness": self.readiness_report(),
            "trained_modalities": sorted(self.trained),
            "weights": weights,
            "training_reports": self.training_reports,
            "fusion": {
                "rule": "skill-weighted mean of threshold-aligned calibrated probabilities",
                "weight_source": "each modality's own validation balanced accuracy",
                "limitation": (
                    "The cardiovascular cohorts here are disjoint, so a fused score pools "
                    "independent evidence but its improvement over the best single "
                    "modality is not measurable on this data."
                ),
            },
        }

    def save_report(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.report(), indent=2, default=str) + "\n", encoding="utf-8")
        return output
