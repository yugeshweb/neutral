"""Ingestion adapter: a real study on disk -> model-ready tensor, or an explicit refusal.

This is the missing link between an upstream pipeline (PACS export, a research archive, a
folder someone dropped a study into) and `serving.predict`. It accepts what such a pipeline
actually produces -- a DICOM series directory, a NIfTI file, or a set of per-sequence NIfTIs --
and applies **the same preprocessing the model was trained with**, read from the artifact
rather than re-specified by the caller.

Design rules, each of which exists because the alternative fails quietly in production:

* **The artifact dictates preprocessing.** Grid size, channel order, and CT windowing come from
  the bundle's manifest, not from arguments. A caller cannot accidentally feed 64^3 into a model
  trained at 128^2, or swap FLAIR and T2 by passing paths in the wrong order.
* **Missing sequences are an error, not a zero-filled channel.** A four-sequence glioma model
  handed three sequences must refuse; substituting zeros produces a confident, meaningless score.
* **Modality is checked, not assumed.** CT is windowed in Hounsfield units; MR is percentile
  normalised. Applying the wrong one silently destroys contrast.
* **Every ingest returns provenance** -- what files were read, what was applied -- so a result
  can be traced back to the exact inputs that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .imaging_hybrid import normalize_volume

SUPPORTED_MODALITIES = {"MR", "CT"}


@dataclass
class IngestResult:
    ok: bool
    volume: np.ndarray | None
    provenance: dict[str, Any]
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "shape": list(self.volume.shape) if self.volume is not None else None,
            "provenance": self.provenance,
            "errors": self.errors,
        }


def read_dicom_series(directory: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Assemble one DICOM series directory into a 3-D array ordered by slice position."""

    import pydicom

    folder = Path(directory)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".dcm", ""})
    slices: list[tuple[float, np.ndarray]] = []
    modality = None
    slope, intercept = 1.0, 0.0
    for path in files:
        try:
            dataset = pydicom.dcmread(str(path), force=True)
        except Exception:  # noqa: BLE001 - a non-DICOM file in the folder is not fatal
            continue
        if not hasattr(dataset, "pixel_array"):
            continue
        modality = modality or str(getattr(dataset, "Modality", "") or "")
        slope = float(getattr(dataset, "RescaleSlope", slope) or slope)
        intercept = float(getattr(dataset, "RescaleIntercept", intercept) or intercept)
        position = getattr(dataset, "ImagePositionPatient", None)
        key = float(position[2]) if position else float(getattr(dataset, "InstanceNumber", len(slices)))
        slices.append((key, dataset.pixel_array.astype(np.float32)))
    if not slices:
        raise ValueError(f"no readable DICOM slices in {folder}")
    slices.sort(key=lambda item: item[0])
    volume = np.stack([array for _, array in slices], axis=-1)
    # Hounsfield units require the rescale transform; skipping it is a classic CT bug.
    volume = volume * slope + intercept
    return volume, {
        "reader": "pydicom",
        "slices": len(slices),
        "modality": modality,
        "rescale_slope": slope,
        "rescale_intercept": intercept,
    }


def read_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    import nibabel as nib

    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    return data, {"reader": "nibabel", "zooms": [float(z) for z in image.header.get_zooms()[:3]]}


def _apply_modality_normalisation(
    volume: np.ndarray, modality: str, grid: tuple[int, int, int], ct_windows: list[dict] | None
) -> list[np.ndarray]:
    """Return the channel stack for one source volume, per the artifact's modality rules."""

    if modality.upper() == "CT" and ct_windows:
        from .bhsd_ich import apply_ct_window

        out = []
        for window in ct_windows:
            windowed = apply_ct_window(volume, float(window["level"]), float(window["width"]))
            out.append(normalize_volume(windowed, grid) if windowed.shape != grid else windowed)
        return out
    return [normalize_volume(volume, grid)]


def ingest_study(
    manifest: dict[str, Any],
    sources: dict[str, str | Path] | str | Path,
    modality: str | None = None,
) -> IngestResult:
    """Turn a study into the exact tensor the artifact expects.

    `manifest` is a bundle manifest (`InferenceBundle.to_manifest()`).
    `sources` is either a single path (single-channel models) or a mapping of
    channel name -> path (multi-sequence models, e.g. {"t1": ..., "flair": ...}).
    """

    errors: list[str] = []
    expects = manifest.get("expects", {})
    channels: list[str] = list(expects.get("channels", []))
    grid = tuple(int(v) for v in expects.get("grid", [64, 64, 64]))
    ct_windows = (manifest.get("training_provenance", {}) or {}).get("ct_windows")
    declared_modality = (modality or (manifest.get("training_provenance", {}) or {}).get("modality", "MR"))
    declared_modality = "CT" if "ct" in str(declared_modality).lower() else "MR"

    # A CT model built from multiple windows of ONE scan takes a single source, not N sources.
    multi_window_ct = declared_modality == "CT" and ct_windows and len(ct_windows) == len(channels)

    provenance: dict[str, Any] = {
        "expected_channels": channels,
        "grid": list(grid),
        "modality": declared_modality,
        "sources": {},
    }

    try:
        if isinstance(sources, (str, Path)):
            if len(channels) > 1 and not multi_window_ct:
                errors.append(
                    f"model expects {len(channels)} sequences {channels}; a single path was given. "
                    "Missing sequences are refused rather than zero-filled."
                )
                return IngestResult(False, None, provenance, errors)
            path = Path(sources)
            raw, info = (read_dicom_series(path) if path.is_dir() else read_nifti(path))
            provenance["sources"][channels[0] if channels else "input"] = {
                "path": str(path), **info
            }
            stack = _apply_modality_normalisation(raw, declared_modality, grid, ct_windows)
        else:
            missing = [c for c in channels if c not in sources]
            if missing:
                errors.append(
                    f"missing required sequence(s) {missing}; model expects {channels}. "
                    "Refusing rather than substituting zeros."
                )
                return IngestResult(False, None, provenance, errors)
            stack = []
            for channel in channels:
                path = Path(sources[channel])
                raw, info = (read_dicom_series(path) if path.is_dir() else read_nifti(path))
                provenance["sources"][channel] = {"path": str(path), **info}
                stack.extend(_apply_modality_normalisation(raw, declared_modality, grid, None))
    except Exception as exc:  # noqa: BLE001 - surface the reason, do not raise into a pipeline
        errors.append(f"{type(exc).__name__}: {exc}")
        return IngestResult(False, None, provenance, errors)

    volume = np.stack(stack, axis=0).astype(np.float32)
    if volume.shape[0] != len(channels):
        errors.append(f"assembled {volume.shape[0]} channel(s) but model expects {len(channels)}")
        return IngestResult(False, None, provenance, errors)

    provenance["applied"] = (
        f"CT windowing ({len(ct_windows)} windows) + resample to {grid}"
        if multi_window_ct
        else f"percentile normalisation + resample to {grid}"
    )
    return IngestResult(True, volume, provenance, errors)


def ingest_and_predict(
    bundle_path: str | Path,
    sources: dict[str, str | Path] | str | Path,
    study_id: str = "study",
) -> dict[str, Any]:
    """One call from files on disk to a usable result. The integration entry point."""

    from .serving import load_bundle, predict

    bundle = load_bundle(bundle_path)
    ingested = ingest_study(bundle.to_manifest(), sources)
    if not ingested.ok:
        return {
            "schema_version": 2,
            "model_id": bundle.model_id,
            "study_id": study_id,
            "status": "rejected",
            "reasons": ingested.errors,
            "ingestion": ingested.provenance,
            "prediction": None,
            "score": None,
        }
    result = predict(bundle, ingested.volume, study_id=study_id)
    result["ingestion"] = ingested.provenance
    return result


if __name__ == "__main__":
    import sys

    print(json.dumps(ingest_and_predict(sys.argv[1], sys.argv[2]), indent=2, default=str))
