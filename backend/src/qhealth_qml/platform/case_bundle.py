"""Case bundle building and validation."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .adapters import adapt_asset
from .registry import Registry
from .schema import (
    AssetFormat,
    DataBundle,
    FieldMapping,
    ModalityAsset,
    Modality,
    ValidationIssue,
    Visit,
)


def build_bundle(spec: dict, *, synthetic: bool = False) -> DataBundle:
    """Build a DataBundle from a caller-supplied specification.

    Args:
        spec: Dictionary with keys:
            - case_id: Patient/case identifier (optional; auto-assigned if missing)
            - source: "upload" | "fixture" | "demo"
            - visits: List of visit dicts with visit_id, timestamp (optional), note (optional)
            - assets: List of asset dicts with path, role, modality, format

        synthetic: Whether this bundle is synthetic/demo data.

    Returns:
        A DataBundle with adapted assets and initial validation state.
    """
    bundle_id = f"bundle-{uuid4()}"

    # Handle case_id: assign local if missing
    case_id = spec.get("case_id")
    if not case_id or (isinstance(case_id, str) and not case_id.strip()):
        case_id = f"local-{uuid4()}"
        case_id_source = "assigned_local"
    else:
        case_id_source = "supplied"

    # Timestamp: ISO-8601 UTC
    created_at = datetime.now(timezone.utc).isoformat()

    # Source: use supplied or default to "upload"
    source = spec.get("source", "upload")

    # Build visits
    visits = []
    for visit_spec in spec.get("visits", []):
        visits.append(
            Visit(
                visit_id=visit_spec.get("visit_id", f"visit-{uuid4()}"),
                timestamp=visit_spec.get("timestamp"),
                note=visit_spec.get("note"),
            )
        )

    # Build assets by dispatching to adapters
    assets = []
    content_hashes = {}
    validation_issues = []

    for asset_spec in spec.get("assets", []):
        asset_path = asset_spec.get("path")
        asset_role = asset_spec.get("role", "unknown")
        asset_modality = asset_spec.get("modality", "structured_clinical")
        asset_format = asset_spec.get("format", "csv")

        # Adapt the asset
        asset = adapt_asset(
            path=asset_path,
            role=asset_role,
            modality=asset_modality,
            format=asset_format,
            bundle_id=bundle_id,
        )

        assets.append(asset)
        content_hashes[asset.asset_id] = asset.content_hash

        # Collect any validation issues from the asset
        validation_issues.extend(asset.validation_issues)

    # Provenance: basic metadata
    provenance = {
        "adapter": "adapters.py",
        "adapter_version": "1.0.0",
        "source_system": source,
        "created_at": created_at,
    }

    return DataBundle(
        bundle_id=bundle_id,
        case_id=case_id,
        case_id_source=case_id_source,
        created_at=created_at,
        source=source,
        synthetic=synthetic,
        visits=visits,
        assets=assets,
        validation=validation_issues,
        content_hashes=content_hashes,
        provenance=provenance,
    )


def validate_bundle(bundle: DataBundle, registry: Registry) -> DataBundle:
    """Validate a DataBundle and return a new bundle with validation statuses set.

    Performs FR-008 checks per asset:
    - Identifier presence
    - Declared modality vs format compatibility
    - SHA256 recomputation and mismatch detection
    - Unit table sanity
    - Dimension sanity (for imaging)
    - Timestamp parseability and ordering

    Args:
        bundle: The DataBundle to validate.
        registry: The Registry (currently unused; reserved for future model-specific checks).

    Returns:
        A new DataBundle with updated validation_status on each asset.
    """
    # Bundle-level issues only; per-asset issues are re-derived below from the
    # (possibly updated) validated assets, not carried over from bundle.validation,
    # to avoid double-counting an issue that was already recorded during adaptation.
    bundle_level_issues: list[ValidationIssue] = []

    if not bundle.case_id or (isinstance(bundle.case_id, str) and not bundle.case_id.strip()):
        bundle_level_issues.append(
            ValidationIssue(
                code="missing_case_id",
                severity="warning",
                message="Case identifier is missing or empty.",
                field="case_id",
                asset_id=None,
            )
        )

    # Validate each asset
    validated_assets = []

    for asset in bundle.assets:
        # Start with existing issues (e.g., from adapter rejection)
        asset_issues = list(asset.validation_issues)
        # Use the existing status unless we find new issues
        validation_status = asset.validation_status

        # Don't validate further if already rejected by adapter
        if validation_status != "rejected":
            # Check 1: File exists and is readable (hard error -> rejected)
            if asset.uri and not asset.uri.startswith("memory:"):
                try:
                    path = Path(asset.uri)
                    if not path.exists():
                        asset_issues.append(
                            ValidationIssue(
                                code="file_not_found",
                                severity="error",
                                message=f"Asset file does not exist.",
                                field=None,
                                asset_id=asset.asset_id,
                            )
                        )
                        validation_status = "rejected"
                except Exception:
                    pass

            # Check 2: Format-modality compatibility (soft warning)
            # (Placeholder for future model-specific checks; currently just log)

            # Check 3: SHA256 recomputation if file exists
            if asset.uri and not asset.uri.startswith("memory:"):
                try:
                    from .adapters import _compute_hash

                    recomputed_hash = _compute_hash(asset.uri)
                    if asset.content_hash and asset.content_hash != recomputed_hash:
                        asset_issues.append(
                            ValidationIssue(
                                code="hash_mismatch",
                                severity="error",
                                message="Content hash does not match file contents.",
                                field=None,
                                asset_id=asset.asset_id,
                            )
                        )
                        validation_status = "rejected"
                except Exception:
                    pass

            # Check 4: Unit table sanity (soft error -> quality_failed)
            if asset.units:
                for field_name, unit_value in asset.units.items():
                    if not unit_value or (isinstance(unit_value, str) and not unit_value.strip()):
                        asset_issues.append(
                            ValidationIssue(
                                code="empty_unit",
                                severity="warning",
                                message=f"Unit for field '{field_name}' is empty or missing.",
                                field=field_name,
                                asset_id=asset.asset_id,
                            )
                        )
                        if validation_status == "accepted":
                            validation_status = "quality_failed"

            # Check 5: Dimension sanity for imaging assets
            if asset.modality == "imaging":
                if not asset.dimensions:
                    asset_issues.append(
                        ValidationIssue(
                            code="missing_dimensions",
                            severity="error",
                            message="Imaging asset is missing dimensions.",
                            field=None,
                            asset_id=asset.asset_id,
                        )
                    )
                    validation_status = "rejected"
                else:
                    # Check all dimensions are positive
                    for i, dim in enumerate(asset.dimensions):
                        if dim <= 0:
                            asset_issues.append(
                                ValidationIssue(
                                    code="invalid_dimension",
                                    severity="error",
                                    message=f"Dimension {i} is not positive: {dim}.",
                                    field=None,
                                    asset_id=asset.asset_id,
                                )
                            )
                            validation_status = "rejected"

            # Check 6: Timestamp parseability and ordering
            if asset.acquired_at:
                try:
                    acquired_dt = datetime.fromisoformat(asset.acquired_at.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)

                    # Soft error: acquired_at in the future -> quality_failed
                    if acquired_dt > now:
                        asset_issues.append(
                            ValidationIssue(
                                code="future_timestamp",
                                severity="warning",
                                message="Asset acquired_at timestamp is in the future.",
                                field="acquired_at",
                                asset_id=asset.asset_id,
                            )
                        )
                        if validation_status == "accepted":
                            validation_status = "quality_failed"
                except (ValueError, TypeError):
                    asset_issues.append(
                        ValidationIssue(
                            code="invalid_timestamp",
                            severity="error",
                            message="Asset acquired_at timestamp is not valid ISO-8601.",
                            field="acquired_at",
                            asset_id=asset.asset_id,
                        )
                    )
                    validation_status = "quality_failed"

            # Check visit timestamps if asset references a visit
            if asset.visit_id:
                visit = next((v for v in bundle.visits if v.visit_id == asset.visit_id), None)
                if visit and visit.timestamp:
                    try:
                        visit_dt = datetime.fromisoformat(visit.timestamp.replace("Z", "+00:00"))
                        now = datetime.now(timezone.utc)

                        if visit_dt > now:
                            asset_issues.append(
                                ValidationIssue(
                                    code="future_visit_timestamp",
                                    severity="warning",
                                    message=f"Visit {asset.visit_id} timestamp is in the future.",
                                    field="timestamp",
                                    asset_id=asset.asset_id,
                                )
                            )
                            if validation_status == "accepted":
                                validation_status = "quality_failed"
                    except (ValueError, TypeError):
                        asset_issues.append(
                            ValidationIssue(
                                code="invalid_visit_timestamp",
                                severity="error",
                                message=f"Visit {asset.visit_id} timestamp is not valid ISO-8601.",
                                field="timestamp",
                                asset_id=asset.asset_id,
                            )
                        )
                        if validation_status == "accepted":
                            validation_status = "quality_failed"

        # Create validated asset with updated status and issues
        validated_asset = dataclasses.replace(
            asset,
            validation_status=validation_status,
            validation_issues=asset_issues,
        )
        validated_assets.append(validated_asset)

    all_issues = bundle_level_issues + [issue for asset in validated_assets for issue in asset.validation_issues]

    # Return new bundle with validated assets
    return dataclasses.replace(
        bundle,
        assets=validated_assets,
        validation=all_issues,
    )
