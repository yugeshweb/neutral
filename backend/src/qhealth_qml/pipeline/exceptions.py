"""Typed exceptions for standardization and training pipeline orchestration."""

from typing import List, Optional

try:
    from qhealth_qml.standardize import (
        EmptyDatasetError,
        SchemaMismatchError,
        SchemaNotFittedError,
        StandardizationError,
        UnknownDiseaseError,
        UnsupportedFormatError,
    )
except ImportError:
    class StandardizationError(Exception):
        """Base exception for all standardizer-related failures."""

        def __init__(self, message: str, disease_id: Optional[str] = None):
            super().__init__(message)
            self.message = message
            self.disease_id = disease_id

    class UnknownDiseaseError(StandardizationError):
        """Raised when the requested disease_id is not registered in the standardizer."""

        def __init__(self, disease_id: str, message: Optional[str] = None):
            msg = message or f"Unknown disease identifier: '{disease_id}'"
            super().__init__(msg, disease_id=disease_id)

    class UnsupportedFormatError(StandardizationError):
        """Raised when the raw uploaded file is not parseable as valid CSV or valid UTF-8."""

        def __init__(self, message: str = "Uploaded file is not a supported CSV format or valid UTF-8 encoding", disease_id: Optional[str] = None):
            super().__init__(message, disease_id=disease_id)

    class EmptyDatasetError(StandardizationError):
        """Raised when the uploaded file contains headers but zero data rows."""

        def __init__(self, message: str = "Dataset contains no data rows", disease_id: Optional[str] = None):
            super().__init__(message, disease_id=disease_id)

    class SchemaMismatchError(StandardizationError):
        """Raised when required columns for the disease schema are missing."""

        def __init__(self, missing_fields: Optional[List[str]] = None, message: Optional[str] = None, disease_id: Optional[str] = None):
            self.missing_fields = missing_fields or []
            msg = message or f"Dataset schema mismatch: missing required columns {self.missing_fields}"
            super().__init__(msg, disease_id=disease_id)

    class SchemaNotFittedError(StandardizationError):
        """Raised when an unlabeled predict upload arrives before any training upload ever fitted this disease's schema."""

        def __init__(self, disease_id: str, message: Optional[str] = None):
            msg = message or f"Schema not fitted for disease '{disease_id}'. A labeled training dataset must be ingested first."
            super().__init__(msg, disease_id=disease_id)


class PipelineError(Exception):
    """Base exception for pipeline orchestration failures."""

    def __init__(self, message: str, disease_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.disease_id = disease_id


class ModelTrainingError(PipelineError):
    """Raised when registered model training fails."""
    pass


class BenchmarkNotFoundError(PipelineError):
    """Raised when static benchmark reference file cannot be located."""
    pass
