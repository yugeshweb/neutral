"""Platform schema: canonical dataclasses, enums, and serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


# Platform version constant
PLATFORM_SCHEMA_VERSION = 1


# Custom exception for schema validation
class SchemaError(Exception):
    """Raised when schema validation fails."""

    def __init__(self, field_path: str, message: str):
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")

    def __str__(self) -> str:
        return f"{self.field_path}: {self.message}"


class _AsDict:
    """Shared to_dict() for every schema dataclass. dataclasses.asdict() already
    recurses into nested dataclasses and lists of them, and Literal-typed fields
    are plain strings at runtime — verified byte-identical to hand-written
    per-class to_dict() implementations this replaces."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]


# Enumerations (section 2)

ModelStatus = Literal[
    "compatible",
    "incompatible",
    "not available",
    "insufficient data",
    "ready",
    "running",
    "completed",
    "abstained",
    "failed",
]

FindingStatus = Literal[
    "not evaluated",
    "not available",
    "insufficient data",
    "negative",
    "positive",
    "abstained",
]

Modality = Literal[
    "structured_clinical",
    "imaging",
    "signal",
    "biomarker",
    "genomic",
    "document",
    "derived_features",
]

AssetFormat = Literal["csv", "json", "fhir", "dicom", "nifti", "edf", "npz"]

TaskType = Literal[
    "binary_classification",
    "multilabel_classification",
    "segmentation",
    "event_detection",
    "regression",
    "progression_risk",
]

ReadinessTier = Literal["high_reference", "research_only"]
ReferenceTier = Literal["high", "moderate", "low"]
Availability = Literal["available", "not available"]
Lifecycle = Literal["experimental", "operational_reference", "deprecated"]
ScoreType = Literal[
    "calibrated_probability",
    "probability",
    "decision_function",
    "regression_value",
    "mask",
    "intervals",
    "none",
]
CalibrationStatus = Literal["calibrated", "uncalibrated", "not_assessed", "failed"]
UncertaintyKind = Literal["calibrated_probability", "decision_margin", "bootstrap_ci", "none"]
ExplanationStatus = Literal["available", "unavailable", "not_applicable", "surrogate"]
AssetValidationStatus = Literal["accepted", "rejected", "missing", "quality_failed"]
RunStatus = Literal["ready", "running", "completed", "failed"]
RunMode = Literal["research", "demo"]
ReuseDecision = Literal["reused", "adapted", "reproduced", "newly_authored"]
RealGainDecision = Literal["passed", "failed", "not_assessed"]
EvidenceKind = Literal[
    "feature_contribution",
    "image_region",
    "signal_interval",
    "biomarker",
    "quality_flag",
    "reference_link",
]
Severity = Literal["info", "warning", "error"]


# Supporting structs (section 3.1)

@dataclass(frozen=True)
class Reference(_AsDict):
    label: str
    url: str
    citation: str | None = None


def Reference_from_dict(raw: dict) -> Reference:
    """Deserialize a Reference from a dict."""
    try:
        return Reference(
            label=str(raw["label"]),
            url=str(raw["url"]),
            citation=str(raw["citation"]) if raw.get("citation") is not None else None,
        )
    except KeyError as e:
        raise SchemaError(f"Reference", f"missing required field: {e}")


@dataclass(frozen=True)
class ValidationIssue(_AsDict):
    code: str
    severity: Severity
    message: str
    field: str | None = None
    asset_id: str | None = None


def ValidationIssue_from_dict(raw: dict) -> ValidationIssue:
    """Deserialize a ValidationIssue from a dict."""
    try:
        severity = raw["severity"]
        if severity not in ("info", "warning", "error"):
            raise SchemaError("ValidationIssue.severity", f"invalid value: {severity}")
        return ValidationIssue(
            code=str(raw["code"]),
            severity=severity,
            message=str(raw["message"]),
            field=str(raw["field"]) if raw.get("field") is not None else None,
            asset_id=str(raw["asset_id"]) if raw.get("asset_id") is not None else None,
        )
    except KeyError as e:
        raise SchemaError("ValidationIssue", f"missing required field: {e}")


@dataclass(frozen=True)
class FieldMapping(_AsDict):
    canonical_field: str
    source_field: str
    source_value: str | None
    code_system: str | None
    unit: str | None
    timestamp: str | None
    transform: str
    note: str | None = None


def FieldMapping_from_dict(raw: dict) -> FieldMapping:
    """Deserialize a FieldMapping from a dict."""
    try:
        return FieldMapping(
            canonical_field=str(raw["canonical_field"]),
            source_field=str(raw["source_field"]),
            source_value=str(raw["source_value"]) if raw.get("source_value") is not None else None,
            code_system=str(raw["code_system"]) if raw.get("code_system") is not None else None,
            unit=str(raw["unit"]) if raw.get("unit") is not None else None,
            timestamp=str(raw["timestamp"]) if raw.get("timestamp") is not None else None,
            transform=str(raw["transform"]),
            note=str(raw["note"]) if raw.get("note") is not None else None,
        )
    except KeyError as e:
        raise SchemaError("FieldMapping", f"missing required field: {e}")


@dataclass(frozen=True)
class Visit(_AsDict):
    visit_id: str
    timestamp: str | None = None
    note: str | None = None


def Visit_from_dict(raw: dict) -> Visit:
    """Deserialize a Visit from a dict."""
    try:
        return Visit(
            visit_id=str(raw["visit_id"]),
            timestamp=str(raw["timestamp"]) if raw.get("timestamp") is not None else None,
            note=str(raw["note"]) if raw.get("note") is not None else None,
        )
    except KeyError as e:
        raise SchemaError("Visit", f"missing required field: {e}")


@dataclass(frozen=True)
class OptionalModalitySpec(_AsDict):
    modality: Modality
    trained_for_absence: bool


def OptionalModalitySpec_from_dict(raw: dict) -> OptionalModalitySpec:
    """Deserialize an OptionalModalitySpec from a dict."""
    try:
        modality = raw["modality"]
        valid_modalities = (
            "structured_clinical",
            "imaging",
            "signal",
            "biomarker",
            "genomic",
            "document",
            "derived_features",
        )
        if modality not in valid_modalities:
            raise SchemaError("OptionalModalitySpec.modality", f"invalid value: {modality}")
        return OptionalModalitySpec(
            modality=modality,
            trained_for_absence=bool(raw["trained_for_absence"]),
        )
    except KeyError as e:
        raise SchemaError("OptionalModalitySpec", f"missing required field: {e}")


@dataclass(frozen=True)
class InputContract(_AsDict):
    required_modalities: list[Modality]
    optional_modalities: list[OptionalModalitySpec]
    required_fields: list[str]
    min_rows: int
    population: str
    population_filters: dict[str, str]
    quality_constraints: dict[str, str]


def InputContract_from_dict(raw: dict) -> InputContract:
    """Deserialize an InputContract from a dict."""
    try:
        return InputContract(
            required_modalities=list(raw["required_modalities"]),
            optional_modalities=[OptionalModalitySpec_from_dict(om) for om in raw["optional_modalities"]],
            required_fields=list(raw["required_fields"]),
            min_rows=int(raw["min_rows"]),
            population=str(raw["population"]),
            population_filters=dict(raw["population_filters"]),
            quality_constraints=dict(raw["quality_constraints"]),
        )
    except KeyError as e:
        raise SchemaError("InputContract", f"missing required field: {e}")


@dataclass(frozen=True)
class PreprocessingSpec(_AsDict):
    imputation: str
    scaling: str
    reduction: str
    n_components: int | None
    angle_scaling: str | None
    fitted_on: str


def PreprocessingSpec_from_dict(raw: dict) -> PreprocessingSpec:
    """Deserialize a PreprocessingSpec from a dict."""
    try:
        return PreprocessingSpec(
            imputation=str(raw["imputation"]),
            scaling=str(raw["scaling"]),
            reduction=str(raw["reduction"]),
            n_components=int(raw["n_components"]) if raw.get("n_components") is not None else None,
            angle_scaling=str(raw["angle_scaling"]) if raw.get("angle_scaling") is not None else None,
            fitted_on=str(raw["fitted_on"]),
        )
    except KeyError as e:
        raise SchemaError("PreprocessingSpec", f"missing required field: {e}")


@dataclass(frozen=True)
class QuantumSpec(_AsDict):
    framework: str
    encoding: str
    ansatz: str | None
    circuit_version: str
    backend_mode: str
    n_qubits: int
    shots: int | None


def QuantumSpec_from_dict(raw: dict) -> QuantumSpec:
    """Deserialize a QuantumSpec from a dict."""
    try:
        return QuantumSpec(
            framework=str(raw["framework"]),
            encoding=str(raw["encoding"]),
            ansatz=str(raw["ansatz"]) if raw.get("ansatz") is not None else None,
            circuit_version=str(raw["circuit_version"]),
            backend_mode=str(raw["backend_mode"]),
            n_qubits=int(raw["n_qubits"]),
            shots=int(raw["shots"]) if raw.get("shots") is not None else None,
        )
    except KeyError as e:
        raise SchemaError("QuantumSpec", f"missing required field: {e}")


@dataclass(frozen=True)
class ArtifactRef(_AsDict):
    kind: str
    path: str | None
    manifest_path: str | None
    sha256: str | None
    schema_version: int | None


def ArtifactRef_from_dict(raw: dict) -> ArtifactRef:
    """Deserialize an ArtifactRef from a dict."""
    try:
        return ArtifactRef(
            kind=str(raw["kind"]),
            path=str(raw["path"]) if raw.get("path") is not None else None,
            manifest_path=str(raw["manifest_path"]) if raw.get("manifest_path") is not None else None,
            sha256=str(raw["sha256"]) if raw.get("sha256") is not None else None,
            schema_version=int(raw["schema_version"]) if raw.get("schema_version") is not None else None,
        )
    except KeyError as e:
        raise SchemaError("ArtifactRef", f"missing required field: {e}")


@dataclass(frozen=True)
class CalibrationSpec(_AsDict):
    method: str
    status: CalibrationStatus
    note: str | None = None


def CalibrationSpec_from_dict(raw: dict) -> CalibrationSpec:
    """Deserialize a CalibrationSpec from a dict."""
    try:
        status = raw["status"]
        valid_statuses = ("calibrated", "uncalibrated", "not_assessed", "failed")
        if status not in valid_statuses:
            raise SchemaError("CalibrationSpec.status", f"invalid value: {status}")
        return CalibrationSpec(
            method=str(raw["method"]),
            status=status,
            note=str(raw["note"]) if raw.get("note") is not None else None,
        )
    except KeyError as e:
        raise SchemaError("CalibrationSpec", f"missing required field: {e}")


@dataclass(frozen=True)
class ExplainabilitySpec(_AsDict):
    method: str
    scope: str
    surrogate: bool


def ExplainabilitySpec_from_dict(raw: dict) -> ExplainabilitySpec:
    """Deserialize an ExplainabilitySpec from a dict."""
    try:
        return ExplainabilitySpec(
            method=str(raw["method"]),
            scope=str(raw["scope"]),
            surrogate=bool(raw["surrogate"]),
        )
    except KeyError as e:
        raise SchemaError("ExplainabilitySpec", f"missing required field: {e}")


@dataclass(frozen=True)
class SafetySpec(_AsDict):
    allows_negative_finding: bool
    abstain_margin: float | None
    requires_full_coverage: bool
    disclaimer: str
    limitations: list[str]


def SafetySpec_from_dict(raw: dict) -> SafetySpec:
    """Deserialize a SafetySpec from a dict."""
    try:
        return SafetySpec(
            allows_negative_finding=bool(raw["allows_negative_finding"]),
            abstain_margin=float(raw["abstain_margin"]) if raw.get("abstain_margin") is not None else None,
            requires_full_coverage=bool(raw["requires_full_coverage"]),
            disclaimer=str(raw["disclaimer"]),
            limitations=list(raw["limitations"]),
        )
    except KeyError as e:
        raise SchemaError("SafetySpec", f"missing required field: {e}")


@dataclass(frozen=True)
class ReuseManifestEntry(_AsDict):
    component: str
    decision: ReuseDecision
    source_url: str | None
    release_or_commit: str | None
    paper_citation: str | None
    license: str | None
    license_verified: bool
    weight_source: str | None
    weight_sha256: str | None
    preprocessing_assumptions: str | None
    io_contract: str | None
    local_modifications: str | None


def ReuseManifestEntry_from_dict(raw: dict) -> ReuseManifestEntry:
    """Deserialize a ReuseManifestEntry from a dict."""
    try:
        decision = raw["decision"]
        valid_decisions = ("reused", "adapted", "reproduced", "newly_authored")
        if decision not in valid_decisions:
            raise SchemaError("ReuseManifestEntry.decision", f"invalid value: {decision}")
        return ReuseManifestEntry(
            component=str(raw["component"]),
            decision=decision,
            source_url=str(raw["source_url"]) if raw.get("source_url") is not None else None,
            release_or_commit=str(raw["release_or_commit"]) if raw.get("release_or_commit") is not None else None,
            paper_citation=str(raw["paper_citation"]) if raw.get("paper_citation") is not None else None,
            license=str(raw["license"]) if raw.get("license") is not None else None,
            license_verified=bool(raw["license_verified"]),
            weight_source=str(raw["weight_source"]) if raw.get("weight_source") is not None else None,
            weight_sha256=str(raw["weight_sha256"]) if raw.get("weight_sha256") is not None else None,
            preprocessing_assumptions=str(raw["preprocessing_assumptions"]) if raw.get("preprocessing_assumptions") is not None else None,
            io_contract=str(raw["io_contract"]) if raw.get("io_contract") is not None else None,
            local_modifications=str(raw["local_modifications"]) if raw.get("local_modifications") is not None else None,
        )
    except KeyError as e:
        raise SchemaError("ReuseManifestEntry", f"missing required field: {e}")


@dataclass(frozen=True)
class ModelCard(_AsDict):
    intended_use: str
    excluded_use: str
    training_population: str
    evaluation_population: str
    label_policy: str
    data_sources: list[Reference]
    data_licenses: list[str]
    limitations: list[str]
    calibration_status: CalibrationStatus
    explanation_method: str
    maintainer: str
    reuse_manifest: list[ReuseManifestEntry]
    research_record: str


def ModelCard_from_dict(raw: dict) -> ModelCard:
    """Deserialize a ModelCard from a dict."""
    try:
        calibration_status = raw["calibration_status"]
        valid_statuses = ("calibrated", "uncalibrated", "not_assessed", "failed")
        if calibration_status not in valid_statuses:
            raise SchemaError("ModelCard.calibration_status", f"invalid value: {calibration_status}")
        return ModelCard(
            intended_use=str(raw["intended_use"]),
            excluded_use=str(raw["excluded_use"]),
            training_population=str(raw["training_population"]),
            evaluation_population=str(raw["evaluation_population"]),
            label_policy=str(raw["label_policy"]),
            data_sources=[Reference_from_dict(ref) for ref in raw["data_sources"]],
            data_licenses=list(raw["data_licenses"]),
            limitations=list(raw["limitations"]),
            calibration_status=calibration_status,
            explanation_method=str(raw["explanation_method"]),
            maintainer=str(raw["maintainer"]),
            reuse_manifest=[ReuseManifestEntry_from_dict(entry) for entry in raw["reuse_manifest"]],
            research_record=str(raw["research_record"]),
        )
    except KeyError as e:
        raise SchemaError("ModelCard", f"missing required field: {e}")


@dataclass(frozen=True)
class Uncertainty(_AsDict):
    kind: UncertaintyKind
    value: float | None
    lower: float | None
    upper: float | None
    calibration_status: CalibrationStatus


def Uncertainty_from_dict(raw: dict) -> Uncertainty:
    """Deserialize an Uncertainty from a dict."""
    try:
        kind = raw["kind"]
        valid_kinds = ("calibrated_probability", "decision_margin", "bootstrap_ci", "none")
        if kind not in valid_kinds:
            raise SchemaError("Uncertainty.kind", f"invalid value: {kind}")
        calibration_status = raw["calibration_status"]
        valid_statuses = ("calibrated", "uncalibrated", "not_assessed", "failed")
        if calibration_status not in valid_statuses:
            raise SchemaError("Uncertainty.calibration_status", f"invalid value: {calibration_status}")
        return Uncertainty(
            kind=kind,
            value=float(raw["value"]) if raw.get("value") is not None else None,
            lower=float(raw["lower"]) if raw.get("lower") is not None else None,
            upper=float(raw["upper"]) if raw.get("upper") is not None else None,
            calibration_status=calibration_status,
        )
    except KeyError as e:
        raise SchemaError("Uncertainty", f"missing required field: {e}")


@dataclass(frozen=True)
class CoverageReport(_AsDict):
    required_present: int
    required_total: int
    optional_present: int
    optional_total: int
    coverage_ratio: float
    missing: list[str]
    quality_failed: list[str]


def CoverageReport_from_dict(raw: dict) -> CoverageReport:
    """Deserialize a CoverageReport from a dict."""
    try:
        return CoverageReport(
            required_present=int(raw["required_present"]),
            required_total=int(raw["required_total"]),
            optional_present=int(raw["optional_present"]),
            optional_total=int(raw["optional_total"]),
            coverage_ratio=float(raw["coverage_ratio"]),
            missing=list(raw["missing"]),
            quality_failed=list(raw["quality_failed"]),
        )
    except KeyError as e:
        raise SchemaError("CoverageReport", f"missing required field: {e}")


@dataclass(frozen=True)
class LeakageCheck(_AsDict):
    name: str
    status: str
    detail: str


def LeakageCheck_from_dict(raw: dict) -> LeakageCheck:
    """Deserialize a LeakageCheck from a dict."""
    try:
        return LeakageCheck(
            name=str(raw["name"]),
            status=str(raw["status"]),
            detail=str(raw["detail"]),
        )
    except KeyError as e:
        raise SchemaError("LeakageCheck", f"missing required field: {e}")


@dataclass(frozen=True)
class StageEvent(_AsDict):
    at: str
    stage: str
    model_id: str | None
    level: Severity
    message: str


def StageEvent_from_dict(raw: dict) -> StageEvent:
    """Deserialize a StageEvent from a dict."""
    try:
        level = raw["level"]
        valid_levels = ("info", "warning", "error")
        if level not in valid_levels:
            raise SchemaError("StageEvent.level", f"invalid value: {level}")
        return StageEvent(
            at=str(raw["at"]),
            stage=str(raw["stage"]),
            model_id=str(raw["model_id"]) if raw.get("model_id") is not None else None,
            level=level,
            message=str(raw["message"]),
        )
    except KeyError as e:
        raise SchemaError("StageEvent", f"missing required field: {e}")


@dataclass(frozen=True)
class ResourceUsage(_AsDict):
    wall_seconds: float
    training_rows: int | None
    test_rows: int | None
    feature_count: int | None
    qubits: int | None
    shots: int | None
    backend: str | None
    estimated_kernel_pairs: int | None


def ResourceUsage_from_dict(raw: dict) -> ResourceUsage:
    """Deserialize a ResourceUsage from a dict."""
    try:
        return ResourceUsage(
            wall_seconds=float(raw["wall_seconds"]),
            training_rows=int(raw["training_rows"]) if raw.get("training_rows") is not None else None,
            test_rows=int(raw["test_rows"]) if raw.get("test_rows") is not None else None,
            feature_count=int(raw["feature_count"]) if raw.get("feature_count") is not None else None,
            qubits=int(raw["qubits"]) if raw.get("qubits") is not None else None,
            shots=int(raw["shots"]) if raw.get("shots") is not None else None,
            backend=str(raw["backend"]) if raw.get("backend") is not None else None,
            estimated_kernel_pairs=int(raw["estimated_kernel_pairs"]) if raw.get("estimated_kernel_pairs") is not None else None,
        )
    except KeyError as e:
        raise SchemaError("ResourceUsage", f"missing required field: {e}")


# Key entities (section 3.2+)

@dataclass(frozen=True)
class ConditionDefinition(_AsDict):
    condition_id: str
    name: str
    domain: str
    priority: str
    task_type: TaskType
    readiness_tier: ReadinessTier
    reference_label_tier: ReferenceTier
    target_definition: str
    population: str
    required_modalities: list[Modality]
    optional_modalities: list[Modality]
    expected_output: str
    reference_datasets: list[Reference]
    evidence_links: list[Reference]
    limitations: list[str]


def ConditionDefinition_from_dict(raw: dict) -> ConditionDefinition:
    """Deserialize a ConditionDefinition from a dict."""
    try:
        task_type = raw["task_type"]
        valid_task_types = (
            "binary_classification",
            "multilabel_classification",
            "segmentation",
            "event_detection",
            "regression",
            "progression_risk",
        )
        if task_type not in valid_task_types:
            raise SchemaError("ConditionDefinition.task_type", f"invalid value: {task_type}")

        readiness_tier = raw["readiness_tier"]
        if readiness_tier not in ("high_reference", "research_only"):
            raise SchemaError("ConditionDefinition.readiness_tier", f"invalid value: {readiness_tier}")

        reference_label_tier = raw["reference_label_tier"]
        if reference_label_tier not in ("high", "moderate", "low"):
            raise SchemaError("ConditionDefinition.reference_label_tier", f"invalid value: {reference_label_tier}")

        return ConditionDefinition(
            condition_id=str(raw["condition_id"]),
            name=str(raw["name"]),
            domain=str(raw["domain"]),
            priority=str(raw["priority"]),
            task_type=task_type,
            readiness_tier=readiness_tier,
            reference_label_tier=reference_label_tier,
            target_definition=str(raw["target_definition"]),
            population=str(raw["population"]),
            required_modalities=list(raw["required_modalities"]),
            optional_modalities=list(raw["optional_modalities"]),
            expected_output=str(raw["expected_output"]),
            reference_datasets=[Reference_from_dict(ref) for ref in raw["reference_datasets"]],
            evidence_links=[Reference_from_dict(ref) for ref in raw["evidence_links"]],
            limitations=list(raw["limitations"]),
        )
    except KeyError as e:
        raise SchemaError("ConditionDefinition", f"missing required field: {e}")


@dataclass(frozen=True)
class ModelDefinition(_AsDict):
    model_id: str
    condition_id: str
    version: str
    display_name: str
    task_type: TaskType
    availability: Availability
    lifecycle: Lifecycle
    executor: str
    input_contract: InputContract
    dataset_profile_path: str | None
    temporal_validation: str
    preprocessing: PreprocessingSpec
    quantum: QuantumSpec | None
    classical_baseline_model_id: str | None
    artifact: ArtifactRef
    calibration: CalibrationSpec
    explainability: ExplainabilitySpec
    output_score_type: ScoreType
    evaluation_record_ids: list[str]
    safety: SafetySpec
    model_card: ModelCard


def ModelDefinition_from_dict(raw: dict) -> ModelDefinition:
    """Deserialize a ModelDefinition from a dict."""
    try:
        task_type = raw["task_type"]
        valid_task_types = (
            "binary_classification",
            "multilabel_classification",
            "segmentation",
            "event_detection",
            "regression",
            "progression_risk",
        )
        if task_type not in valid_task_types:
            raise SchemaError("ModelDefinition.task_type", f"invalid value: {task_type}")

        availability = raw["availability"]
        if availability not in ("available", "not available"):
            raise SchemaError("ModelDefinition.availability", f"invalid value: {availability}")

        lifecycle = raw["lifecycle"]
        if lifecycle not in ("experimental", "operational_reference", "deprecated"):
            raise SchemaError("ModelDefinition.lifecycle", f"invalid value: {lifecycle}")

        output_score_type = raw["output_score_type"]
        valid_score_types = (
            "calibrated_probability",
            "probability",
            "decision_function",
            "regression_value",
            "mask",
            "intervals",
            "none",
        )
        if output_score_type not in valid_score_types:
            raise SchemaError("ModelDefinition.output_score_type", f"invalid value: {output_score_type}")

        return ModelDefinition(
            model_id=str(raw["model_id"]),
            condition_id=str(raw["condition_id"]),
            version=str(raw["version"]),
            display_name=str(raw["display_name"]),
            task_type=task_type,
            availability=availability,
            lifecycle=lifecycle,
            executor=str(raw["executor"]),
            input_contract=InputContract_from_dict(raw["input_contract"]),
            dataset_profile_path=str(raw["dataset_profile_path"]) if raw.get("dataset_profile_path") is not None else None,
            temporal_validation=str(raw["temporal_validation"]),
            preprocessing=PreprocessingSpec_from_dict(raw["preprocessing"]),
            quantum=QuantumSpec_from_dict(raw["quantum"]) if raw.get("quantum") is not None else None,
            classical_baseline_model_id=str(raw["classical_baseline_model_id"]) if raw.get("classical_baseline_model_id") is not None else None,
            artifact=ArtifactRef_from_dict(raw["artifact"]),
            calibration=CalibrationSpec_from_dict(raw["calibration"]),
            explainability=ExplainabilitySpec_from_dict(raw["explainability"]),
            output_score_type=output_score_type,
            evaluation_record_ids=list(raw["evaluation_record_ids"]),
            safety=SafetySpec_from_dict(raw["safety"]),
            model_card=ModelCard_from_dict(raw["model_card"]),
        )
    except KeyError as e:
        raise SchemaError("ModelDefinition", f"missing required field: {e}")


@dataclass(frozen=True)
class DataBundle(_AsDict):
    bundle_id: str
    case_id: str
    case_id_source: str
    created_at: str
    source: str
    synthetic: bool
    visits: list[Visit]
    assets: list  # ModalityAsset
    validation: list[ValidationIssue]
    content_hashes: dict[str, str]
    provenance: dict[str, str]


@dataclass(frozen=True)
class ModalityAsset(_AsDict):
    asset_id: str
    bundle_id: str
    modality: Modality
    format: AssetFormat
    role: str
    visit_id: str | None
    uri: str
    content_hash: str
    byte_size: int
    rows: int | None
    dimensions: list[int] | None
    units: dict[str, str]
    acquired_at: str | None
    validation_status: AssetValidationStatus
    validation_issues: list[ValidationIssue]
    field_mappings: list[FieldMapping]
    derived_from: list[str]


@dataclass(frozen=True)
class RoutingDecision(_AsDict):
    model_id: str
    model_version: str
    condition_id: str
    status: ModelStatus
    reason: str
    satisfied_modalities: list[Modality]
    missing_required: list[Modality]
    missing_optional: list[Modality]
    unmet_constraints: list[str]


@dataclass(frozen=True)
class EvidenceItem(_AsDict):
    evidence_id: str
    finding_id: str
    kind: EvidenceKind
    label: str
    value: float | None
    unit: str | None
    region: dict[str, float] | None
    interval: dict[str, float] | None
    source_asset_id: str | None
    confidence: float | None
    note: str | None


@dataclass(frozen=True)
class Finding(_AsDict):
    finding_id: str
    run_id: str
    model_id: str
    model_version: str
    condition_id: str
    condition_name: str
    task_type: TaskType
    status: FindingStatus
    reason: str
    score: float | None
    score_type: ScoreType
    output: dict[str, Any]
    threshold: float | None
    threshold_policy: str | None
    uncertainty: Uncertainty
    abstained: bool
    reference_label_tier: ReferenceTier
    input_coverage: CoverageReport
    modalities_used: list[Modality]
    evidence: list[EvidenceItem]
    explanation_status: ExplanationStatus
    limitations: list[str]
    disclaimer: str
    synthetic: bool


@dataclass(frozen=True)
class AssessmentRun(_AsDict):
    run_id: str
    bundle_id: str
    domain: str
    started_at: str
    completed_at: str | None
    status: RunStatus
    mode: RunMode
    synthetic: bool
    schema_version: int
    routing: list[RoutingDecision]
    findings: list[Finding]
    stage_events: list[StageEvent]
    resource: ResourceUsage
    fingerprints: dict[str, str]
    disclaimer: str
    errors: list[str]


@dataclass(frozen=True)
class EvaluationRecord(_AsDict):
    evaluation_id: str
    model_id: str
    model_version: str
    condition_id: str
    created_at: str
    dataset_profile: dict[str, Any]
    dataset_fingerprint: str
    split_strategy: str
    split_summary: dict[str, int]
    preprocessing: PreprocessingSpec
    leakage_checks: list[LeakageCheck]
    metrics: dict[str, float | None]
    segmentation_metrics: dict[str, float | None]
    event_metrics: dict[str, float | None]
    calibration: dict[str, Any]
    abstention: dict[str, Any]
    resource: ResourceUsage
    confidence_intervals: dict[str, dict[str, float]]
    baseline_model_id: str | None
    baseline_metrics: dict[str, float | None]
    paired_comparison: dict[str, Any]
    real_gain_decision: RealGainDecision
    real_gain_reason: str
    software: dict[str, str]
    source_result_path: str


def EvaluationRecord_from_dict(raw: dict) -> EvaluationRecord:
    """Deserialize an EvaluationRecord from a dict."""
    try:
        real_gain_decision = raw["real_gain_decision"]
        if real_gain_decision not in ("passed", "failed", "not_assessed"):
            raise SchemaError("EvaluationRecord.real_gain_decision", f"invalid value: {real_gain_decision}")

        return EvaluationRecord(
            evaluation_id=str(raw["evaluation_id"]),
            model_id=str(raw["model_id"]),
            model_version=str(raw["model_version"]),
            condition_id=str(raw["condition_id"]),
            created_at=str(raw["created_at"]),
            dataset_profile=dict(raw["dataset_profile"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            split_strategy=str(raw["split_strategy"]),
            split_summary=dict(raw["split_summary"]),
            preprocessing=PreprocessingSpec_from_dict(raw["preprocessing"]),
            leakage_checks=[LeakageCheck_from_dict(lc) for lc in raw["leakage_checks"]],
            metrics=dict(raw["metrics"]),
            segmentation_metrics=dict(raw["segmentation_metrics"]),
            event_metrics=dict(raw["event_metrics"]),
            calibration=dict(raw["calibration"]),
            abstention=dict(raw["abstention"]),
            resource=ResourceUsage_from_dict(raw["resource"]),
            confidence_intervals={k: dict(v) for k, v in raw["confidence_intervals"].items()},
            baseline_model_id=str(raw["baseline_model_id"]) if raw.get("baseline_model_id") is not None else None,
            baseline_metrics=dict(raw["baseline_metrics"]),
            paired_comparison=dict(raw["paired_comparison"]),
            real_gain_decision=real_gain_decision,
            real_gain_reason=str(raw["real_gain_reason"]),
            software=dict(raw["software"]),
            source_result_path=str(raw["source_result_path"]),
        )
    except KeyError as e:
        raise SchemaError("EvaluationRecord", f"missing required field: {e}")
