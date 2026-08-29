"""Source-format to canonical asset adapters."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .schema import (
    AssetFormat,
    FieldMapping,
    ModalityAsset,
    Modality,
    ValidationIssue,
)


# Registry of format adapters: AssetFormat -> callable(path, role, modality, bundle_id) -> ModalityAsset
_adapter_registry: dict[AssetFormat, Callable] = {}


def register_adapter(format: AssetFormat, fn: Callable) -> None:
    """Register an adapter function for a specific asset format.

    The adapter function should have signature:
        fn(path: str | Path, role: str, modality: Modality, bundle_id: str) -> ModalityAsset
    """
    _adapter_registry[format] = fn


def _compute_hash(path: str | Path) -> str:
    """Compute SHA256 hash of file contents."""
    path = Path(path)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_byte_size(path: str | Path) -> int:
    """Get file size in bytes."""
    return Path(path).stat().st_size


def adapt_csv(
    path: str | Path,
    role: str,
    modality: Modality = "structured_clinical",
    bundle_id: str = None,
) -> ModalityAsset:
    """Adapt a CSV file to a ModalityAsset.

    Args:
        path: Path to the CSV file.
        role: Role of the asset (e.g., "clinical_table").
        modality: Modality type (default: "structured_clinical").
        bundle_id: Bundle ID for the asset (generated if None).

    Returns:
        A ModalityAsset with field mappings and validation status.
    """
    if bundle_id is None:
        bundle_id = f"bundle-{uuid4()}"

    path = Path(path)
    asset_id = f"asset-{uuid4()}"

    try:
        # Compute hash and size
        content_hash = _compute_hash(path)
        byte_size = _get_byte_size(path)

        # Read CSV and count rows, create field mappings
        field_mappings = []
        row_count = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Create field mapping for each column
            if reader.fieldnames:
                for field_name in reader.fieldnames:
                    field_mappings.append(
                        FieldMapping(
                            canonical_field=field_name,
                            source_field=field_name,
                            source_value=None,
                            code_system=None,
                            unit=None,
                            timestamp=None,
                            transform="identity",
                            note=None,
                        )
                    )

            # Count data rows
            for row in reader:
                row_count += 1

        # Build the asset
        return ModalityAsset(
            asset_id=asset_id,
            bundle_id=bundle_id,
            modality=modality,
            format="csv",
            role=role,
            visit_id=None,
            uri=str(path),
            content_hash=content_hash,
            byte_size=byte_size,
            rows=row_count,
            dimensions=None,
            units={},
            acquired_at=None,
            validation_status="accepted",
            validation_issues=[],
            field_mappings=field_mappings,
            derived_from=[],
        )

    except Exception as e:
        # Return a rejected asset on any error
        return ModalityAsset(
            asset_id=asset_id,
            bundle_id=bundle_id,
            modality=modality,
            format="csv",
            role=role,
            visit_id=None,
            uri=str(path),
            content_hash="",
            byte_size=0,
            rows=0,
            dimensions=None,
            units={},
            acquired_at=None,
            validation_status="rejected",
            validation_issues=[
                ValidationIssue(
                    code="csv_read_error",
                    severity="error",
                    message=f"Failed to read CSV file: {str(e)}",
                    field=None,
                    asset_id=asset_id,
                )
            ],
            field_mappings=[],
            derived_from=[],
        )


def adapt_json(
    path: str | Path,
    role: str,
    modality: Modality = "structured_clinical",
    bundle_id: str = None,
) -> ModalityAsset:
    """Adapt a JSON file to a ModalityAsset.

    If the JSON is a flat single-record dict, each key becomes a field with transform="identity".

    Args:
        path: Path to the JSON file.
        role: Role of the asset.
        modality: Modality type (default: "structured_clinical").
        bundle_id: Bundle ID for the asset (generated if None).

    Returns:
        A ModalityAsset with field mappings and validation status.
    """
    if bundle_id is None:
        bundle_id = f"bundle-{uuid4()}"

    path = Path(path)
    asset_id = f"asset-{uuid4()}"

    try:
        # Compute hash and size
        content_hash = _compute_hash(path)
        byte_size = _get_byte_size(path)

        # Load JSON
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create field mappings from JSON keys
        field_mappings = []

        if isinstance(data, dict):
            # Single-record dict: each key is a field
            for key in data.keys():
                field_mappings.append(
                    FieldMapping(
                        canonical_field=key,
                        source_field=key,
                        source_value=None,
                        code_system=None,
                        unit=None,
                        timestamp=None,
                        transform="identity",
                        note=None,
                    )
                )
            row_count = 1
        elif isinstance(data, list):
            # List of records: use keys from first record
            if data and isinstance(data[0], dict):
                for key in data[0].keys():
                    field_mappings.append(
                        FieldMapping(
                            canonical_field=key,
                            source_field=key,
                            source_value=None,
                            code_system=None,
                            unit=None,
                            timestamp=None,
                            transform="identity",
                            note=None,
                        )
                    )
                row_count = len(data)
            else:
                row_count = len(data)
        else:
            row_count = 0

        return ModalityAsset(
            asset_id=asset_id,
            bundle_id=bundle_id,
            modality=modality,
            format="json",
            role=role,
            visit_id=None,
            uri=str(path),
            content_hash=content_hash,
            byte_size=byte_size,
            rows=row_count,
            dimensions=None,
            units={},
            acquired_at=None,
            validation_status="accepted",
            validation_issues=[],
            field_mappings=field_mappings,
            derived_from=[],
        )

    except Exception as e:
        return ModalityAsset(
            asset_id=asset_id,
            bundle_id=bundle_id,
            modality=modality,
            format="json",
            role=role,
            visit_id=None,
            uri=str(path),
            content_hash="",
            byte_size=0,
            rows=0,
            dimensions=None,
            units={},
            acquired_at=None,
            validation_status="rejected",
            validation_issues=[
                ValidationIssue(
                    code="json_read_error",
                    severity="error",
                    message=f"Failed to read JSON file: {str(e)}",
                    field=None,
                    asset_id=asset_id,
                )
            ],
            field_mappings=[],
            derived_from=[],
        )


def adapt_asset(
    path: str | Path,
    role: str,
    modality: Modality,
    format: AssetFormat,
    bundle_id: str = None,
) -> ModalityAsset:
    """Dispatch to the appropriate adapter based on format.

    Args:
        path: Path to the asset file.
        role: Role of the asset.
        modality: Modality type.
        format: Asset format (determines which adapter to use).
        bundle_id: Bundle ID for the asset.

    Returns:
        A ModalityAsset with validation status set appropriately.
        For unsupported formats, returns a rejected asset.
    """
    if bundle_id is None:
        bundle_id = f"bundle-{uuid4()}"

    asset_id = f"asset-{uuid4()}"

    # Check built-in adapters first
    if format == "csv":
        return adapt_csv(path, role, modality, bundle_id)
    elif format == "json":
        return adapt_json(path, role, modality, bundle_id)

    # Check registered adapters
    if format in _adapter_registry:
        return _adapter_registry[format](path, role, modality, bundle_id)

    # Unsupported format: return rejected asset
    return ModalityAsset(
        asset_id=asset_id,
        bundle_id=bundle_id,
        modality=modality,
        format=format,
        role=role,
        visit_id=None,
        uri=str(path),
        content_hash="",
        byte_size=0,
        rows=None,
        dimensions=None,
        units={},
        acquired_at=None,
        validation_status="rejected",
        validation_issues=[
            ValidationIssue(
                code="unsupported_format",
                severity="error",
                message=f"Format '{format}' is not supported. Supported formats: csv, json, and registered adapters.",
                field=None,
                asset_id=asset_id,
            )
        ],
        field_mappings=[],
        derived_from=[],
    )
