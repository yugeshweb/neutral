"""Unified ingestion & preprocessing pipeline and training orchestration layer."""

# --- Ingestion & Preprocessing Pipeline (spec 002-ingestion-preprocessing) ---
from .pipeline import Pipeline, PipelineError as StandardizerPipelineError
from .recipe import ManifestMismatchError, Recipe, RecipePredatesRecipeFormatError, RecipeVersionError
from .registry import AdapterRegistry, AdapterUnavailableError, DispatchError, default_registry
from .spec import SourceSpec, SpecValidationError
from .types import (
    Batch,
    FitResult,
    Issue,
    IssueCode,
    LABEL_EXCLUDE,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    Ledger,
    PreparedArrays,
    QCVerdict,
    RunResult,
    Sample,
    Source,
)

# --- Training Orchestration Layer ---
from qhealth_qml.pipeline.exceptions import (
    StandardizationError,
    UnknownDiseaseError,
    UnsupportedFormatError,
    EmptyDatasetError,
    SchemaMismatchError,
    SchemaNotFittedError,
    PipelineError,
    ModelTrainingError,
    BenchmarkNotFoundError,
)
from qhealth_qml.pipeline.interfaces import (
    ModelOutput,
    TrainDiseaseModelsFn,
    DiseaseModelTrainer,
)
from qhealth_qml.pipeline.model_registry import (
    register_disease_models,
    unregister_disease_models,
    get_disease_models_trainer,
    has_disease_models_trainer,
    list_registered_disease_trainers,
    clear_model_registry,
)
from qhealth_qml.pipeline.disease_registry import (
    DiseaseConfig,
    DiseaseRegistry,
    disease_registry,
)
from qhealth_qml.pipeline.preprocessor import (
    LeakageSafePreprocessor,
    stratified_split,
    create_and_fit_preprocessor,
)
from qhealth_qml.pipeline.evaluator import (
    ModelMetrics,
    evaluate_model_predictions,
    find_optimal_threshold_youden_j,
)
from qhealth_qml.pipeline.benchmarks import (
    load_benchmark_reference,
    compute_metric_deltas,
    compare_with_benchmarks,
)
from qhealth_qml.pipeline.persistence import (
    persist_training_artifacts,
    load_training_artifacts,
    compute_dataset_hash,
)
from qhealth_qml.pipeline.dispatcher import (
    run_training_pipeline,
)
from qhealth_qml.pipeline.fakes import (
    fake_standardize,
    fake_train_disease_models,
    fake_list_supported_diseases,
    MockQuantumModel,
)
from qhealth_qml.pipeline.api import app

__all__ = [
    # Ingestion pipeline
    "Pipeline",
    "Recipe",
    "RecipeVersionError",
    "RecipePredatesRecipeFormatError",
    "ManifestMismatchError",
    "AdapterRegistry",
    "AdapterUnavailableError",
    "DispatchError",
    "default_registry",
    "SourceSpec",
    "SpecValidationError",
    "Batch",
    "Sample",
    "Source",
    "Issue",
    "IssueCode",
    "QCVerdict",
    "PreparedArrays",
    "Ledger",
    "FitResult",
    "RunResult",
    "LABEL_POSITIVE",
    "LABEL_NEGATIVE",
    "LABEL_EXCLUDE",
    # Training orchestration
    "StandardizationError",
    "UnknownDiseaseError",
    "UnsupportedFormatError",
    "EmptyDatasetError",
    "SchemaMismatchError",
    "SchemaNotFittedError",
    "PipelineError",
    "ModelTrainingError",
    "BenchmarkNotFoundError",
    "ModelOutput",
    "TrainDiseaseModelsFn",
    "DiseaseModelTrainer",
    "register_disease_models",
    "unregister_disease_models",
    "get_disease_models_trainer",
    "has_disease_models_trainer",
    "list_registered_disease_trainers",
    "clear_model_registry",
    "DiseaseConfig",
    "DiseaseRegistry",
    "disease_registry",
    "LeakageSafePreprocessor",
    "stratified_split",
    "create_and_fit_preprocessor",
    "ModelMetrics",
    "evaluate_model_predictions",
    "find_optimal_threshold_youden_j",
    "load_benchmark_reference",
    "compute_metric_deltas",
    "compare_with_benchmarks",
    "persist_training_artifacts",
    "load_training_artifacts",
    "compute_dataset_hash",
    "run_training_pipeline",
    "fake_standardize",
    "fake_train_disease_models",
    "fake_list_supported_diseases",
    "MockQuantumModel",
    "app",
]
