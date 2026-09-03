"""FastAPI application for training pipeline orchestration and disease benchmarks."""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from qhealth_qml.pipeline.benchmarks import load_benchmark_reference
from qhealth_qml.pipeline.disease_registry import disease_registry
from qhealth_qml.pipeline.dispatcher import run_training_pipeline
from qhealth_qml.pipeline.exceptions import (
    BenchmarkNotFoundError,
    EmptyDatasetError,
    ModelTrainingError,
    PipelineError,
    SchemaMismatchError,
    SchemaNotFittedError,
    StandardizationError,
    UnknownDiseaseError,
    UnsupportedFormatError,
)

app = FastAPI(
    title="Quantum-Classical Disease Detection Training API",
    description="Backend orchestration layer for training, evaluation, benchmark comparison, and artifact persistence.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---


class ErrorDetail(BaseModel):
    status: str = "error"
    error_type: str
    disease_id: Optional[str] = None
    message: str
    missing_fields: Optional[List[str]] = None


class DiseaseSummary(BaseModel):
    disease_id: str
    name: str
    short_name: str
    standardizer_disease_id: str
    benchmark_file: str
    default_qubits: int
    modality: str
    positive_label: str
    negative_label: str


class ConfusionMatrixSchema(BaseModel):
    tp: int
    fn: int
    tn: int
    fp: int


class RocPointSchema(BaseModel):
    fpr: float
    tpr: float


class ModelMetricSchema(BaseModel):
    accuracy: float
    precision: float
    sensitivity: float
    specificity: float
    f1: float
    roc_auc: float
    confusion_matrix: ConfusionMatrixSchema
    roc_points: List[RocPointSchema]
    threshold: float
    train_time: Any
    infer_time: Any


class StandardizationSummary(BaseModel):
    status: str
    total_rows: int
    raw_features: int
    train_rows: int
    test_rows: int
    preprocessed_features: int
    data_hash: str


class TrainResponse(BaseModel):
    status: str = "success"
    run_id: str
    disease_id: str
    disease_name: str
    standardization: StandardizationSummary
    models: Dict[str, ModelMetricSchema]
    benchmark_comparison: Dict[str, Any]
    artifacts: Dict[str, str]


# --- Exception Handlers ---


@app.exception_handler(UnknownDiseaseError)
async def handle_unknown_disease(request: Request, exc: UnknownDiseaseError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorDetail(
            error_type="UnknownDiseaseError",
            disease_id=getattr(exc, "disease_id", None),
            message=getattr(exc, "message", str(exc)),
        ).model_dump(),
    )


@app.exception_handler(UnsupportedFormatError)
async def handle_unsupported_format(request: Request, exc: UnsupportedFormatError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorDetail(
            error_type="UnsupportedFormatError",
            disease_id=getattr(exc, "disease_id", None),
            message=getattr(exc, "message", str(exc)),
        ).model_dump(),
    )


@app.exception_handler(EmptyDatasetError)
async def handle_empty_dataset(request: Request, exc: EmptyDatasetError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorDetail(
            error_type="EmptyDatasetError",
            disease_id=getattr(exc, "disease_id", None),
            message=getattr(exc, "message", str(exc)),
        ).model_dump(),
    )


@app.exception_handler(SchemaMismatchError)
async def handle_schema_mismatch(request: Request, exc: SchemaMismatchError):
    return JSONResponse(
        status_code=422,
        content=ErrorDetail(
            error_type="SchemaMismatchError",
            disease_id=getattr(exc, "disease_id", None),
            message=getattr(exc, "message", str(exc)),
            missing_fields=getattr(exc, "missing_fields", None),
        ).model_dump(),
    )


@app.exception_handler(SchemaNotFittedError)
async def handle_schema_not_fitted(request: Request, exc: SchemaNotFittedError):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=ErrorDetail(
            error_type="SchemaNotFittedError",
            disease_id=getattr(exc, "disease_id", None),
            message=getattr(exc, "message", str(exc)),
        ).model_dump(),
    )


@app.exception_handler(BenchmarkNotFoundError)
async def handle_benchmark_not_found(request: Request, exc: BenchmarkNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorDetail(
            error_type="BenchmarkNotFoundError",
            disease_id=getattr(exc, "disease_id", None),
            message=getattr(exc, "message", str(exc)),
        ).model_dump(),
    )


@app.exception_handler(ModelTrainingError)
async def handle_model_training_error(request: Request, exc: ModelTrainingError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorDetail(
            error_type="ModelTrainingError",
            disease_id=getattr(exc, "disease_id", None),
            message=getattr(exc, "message", str(exc)),
        ).model_dump(),
    )


@app.exception_handler(StandardizationError)
async def handle_generic_standardization_error(request: Request, exc: StandardizationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorDetail(
            error_type="StandardizationError",
            disease_id=getattr(exc, "disease_id", None),
            message=getattr(exc, "message", str(exc)),
        ).model_dump(),
    )


@app.exception_handler(PipelineError)
async def handle_generic_pipeline_error(request: Request, exc: PipelineError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorDetail(
            error_type="PipelineError",
            disease_id=getattr(exc, "disease_id", None),
            message=getattr(exc, "message", str(exc)),
        ).model_dump(),
    )


# --- Endpoints ---


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "quantum-classical-training-pipeline"}


@app.get("/api/diseases", response_model=List[DiseaseSummary])
async def list_diseases():
    """List the 4 supported diseases and their pipeline metadata."""
    configs = disease_registry.list_diseases()
    return [
        DiseaseSummary(
            disease_id=cfg.disease_id,
            name=cfg.name,
            short_name=cfg.short_name,
            standardizer_disease_id=cfg.standardizer_disease_id,
            benchmark_file=cfg.benchmark_file,
            default_qubits=cfg.default_qubits,
            modality=cfg.modality,
            positive_label=cfg.positive_label,
            negative_label=cfg.negative_label,
        )
        for cfg in configs
    ]


@app.get("/api/benchmarks/{disease_id}")
async def get_benchmark(disease_id: str):
    """Retrieve the static reference benchmark for a disease."""
    return load_benchmark_reference(disease_id)


@app.post("/api/train", response_model=TrainResponse)
async def train_model_endpoint(
    file: UploadFile = File(..., description="Raw tabular dataset CSV file"),
    disease_id: str = Form(..., description="Target disease identifier"),
):
    """Train classical and quantum models for early disease detection.

    Orchestrates: standardize -> split & preprocess -> dispatch -> evaluate -> benchmark -> persist -> return.
    """
    # Read file content safely into memory buffer
    file_bytes = await file.read()

    result = run_training_pipeline(
        raw_file=file_bytes,
        disease_id=disease_id,
    )
    return result
