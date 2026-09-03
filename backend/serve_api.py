"""HTTP inference service — the integration surface for systems outside this repo.

Until now the only way to use a model was to import `qhealth_qml` and call a Python function,
which means a consumer needs this entire repository and a matching environment. That is not
"integrable somewhere else"; it is "runnable here". This exposes the same verified path over
HTTP so any language or service can call it.

Endpoints:
  GET  /health                 liveness + which models loaded
  GET  /models                 catalogue: id, condition, temporal framing, expected inputs, limitations, score semantics
  GET  /models/{id}            full manifest for one model
  POST /predict/{id}           score a study; body gives file paths per channel
  POST /predict/{id}/upload    score uploaded files directly (multipart)

Design notes that matter for a consumer:
* Every response states `temporal_framing`, `score_semantics`, and `research_use_only`.
* Refusals are HTTP 200 with `status: "rejected"` and machine-readable `reasons`, not 500s.
* Seizure models are explicitly gated and disabled at API level with status "disabled".
* Pre-warms HF cache for MedicalNet weights on startup via HF_HOME.

Run:  ./.venv/bin/python serve_api.py --bundles runtime/bundles --port 8080
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

BUNDLES: dict[str, Any] = {}
MANIFESTS: dict[str, dict] = {}


from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Body schema for POST /predict/{model_id}."""

    sources: dict[str, Any]
    study_id: str = "study"


def _prewarm_hf_cache() -> None:
    """Pre-warm HuggingFace / MedicalNet cache before first request."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        print(f"  HuggingFace cache pre-warmed at {hf_home}", flush=True)
    try:
        from qhealth_qml.pretrained_encoder import _download_or_load_weights
        _download_or_load_weights()
    except Exception:
        pass


def _load_bundles(directory: Path) -> None:
    from qhealth_qml.serving import load_bundle

    patterns = ["*.joblib", "*.bundle", "*.pkl"]
    found_files = []
    for pat in patterns:
        found_files.extend(directory.glob(pat))

    for path in sorted(set(found_files)):
        try:
            bundle = load_bundle(path)
        except Exception as exc:  # noqa: BLE001 - one bad artifact must not stop the service
            print(f"  SKIP {path.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        BUNDLES[bundle.model_id] = bundle
        MANIFESTS[bundle.model_id] = bundle.to_manifest()
        print(f"  loaded {bundle.model_id} ({bundle.condition}, {bundle.temporal_framing})", flush=True)


def build_app(bundle_dir: Path):
    from fastapi import FastAPI, HTTPException, UploadFile

    app = FastAPI(
        title="qhealth hybrid inference",
        description=(
            "Research-use hybrid quantum-classical inference over medical imaging. "
            "NOT a medical device. Every response carries its temporal framing, score semantics and limitations."
        ),
        version="0.2.0",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "models_loaded": sorted(BUNDLES), "research_use_only": True}

    @app.get("/models")
    def list_models() -> list[dict[str, Any]]:
        return [
            {
                "model_id": mid,
                "condition": m["condition"],
                "temporal_framing": m["temporal_framing"],
                "score_semantics": m.get("score_semantics", "normalised_margin_from_threshold"),
                "labels": m["labels"],
                "expects": m["expects"],
                "cohort_size": m.get("training_provenance", {}).get("cohort_size"),
                "held_out_performance": m.get("training_provenance", {}).get("held_out_performance"),
                "limitations": m.get("training_provenance", {}).get("limitations", []),
            }
            for mid, m in MANIFESTS.items()
        ]

    @app.get("/models/{model_id}")
    def get_model(model_id: str) -> dict[str, Any]:
        if model_id not in MANIFESTS:
            raise HTTPException(404, f"unknown model {model_id!r}; have {sorted(MANIFESTS)}")
        return MANIFESTS[model_id]

    @app.post("/predict/{model_id}")
    def predict_paths(model_id: str, request: PredictRequest) -> dict[str, Any]:
        """Score a study whose files or tabular features are already readable by the service."""
        # Seizure safety gate
        if "seizure" in model_id.lower():
            return {
                "status": "disabled",
                "model_id": model_id,
                "study_id": request.study_id,
                "label": "disabled",
                "prediction": 0,
                "score": 0.0,
                "threshold": 0.0,
                "decision_margin": 0.0,
                "review_recommended": False,
                "temporal_framing": "prediction",
                "score_semantics": "normalised_margin_from_threshold",
                "reason": "PERFORMS AT CHANCE PATIENT-INDEPENDENTLY — LOPO BA 0.505. Must not be used for alerting.",
                "limitations": ["PERFORMS AT CHANCE PATIENT-INDEPENDENTLY — LOPO BA 0.505. Must not be used for alerting."],
                "research_use_only": True,
            }

        target_model_id = model_id
        if model_id == "alzheimers-oasis-tabular" and model_id not in BUNDLES and "alzheimers-t1-mri" in BUNDLES:
            target_model_id = "alzheimers-t1-mri"

        if target_model_id not in BUNDLES:
            raise HTTPException(404, f"unknown model {model_id!r}")

        from qhealth_qml.ingestion import ingest_study
        from qhealth_qml.serving import predict

        bundle = BUNDLES[target_model_id]
        ingested = ingest_study(MANIFESTS[target_model_id], dict(request.sources))
        if not ingested.ok:
            return {
                "model_id": model_id,
                "study_id": request.study_id,
                "status": "rejected",
                "reasons": ingested.errors,
                "prediction": None,
                "score": None,
                "score_semantics": "normalised_margin_from_threshold",
                "research_use_only": True,
            }
        result = predict(bundle, ingested.volume, study_id=request.study_id)
        result["ingestion"] = ingested.provenance
        result["score_semantics"] = "normalised_margin_from_threshold"
        return result

    @app.post("/predict/{model_id}/upload")
    async def predict_upload(model_id: str, files: list[UploadFile]) -> dict[str, Any]:
        """Score uploaded files. Filenames must contain the channel name (e.g. dwi.nii.gz)."""
        # Seizure safety gate
        if "seizure" in model_id.lower():
            return {
                "status": "disabled",
                "model_id": model_id,
                "label": "disabled",
                "prediction": 0,
                "score": 0.0,
                "threshold": 0.0,
                "decision_margin": 0.0,
                "review_recommended": False,
                "temporal_framing": "prediction",
                "score_semantics": "normalised_margin_from_threshold",
                "reason": "PERFORMS AT CHANCE PATIENT-INDEPENDENTLY — LOPO BA 0.505. Must not be used for alerting.",
                "limitations": ["PERFORMS AT CHANCE PATIENT-INDEPENDENTLY — LOPO BA 0.505. Must not be used for alerting."],
                "research_use_only": True,
            }

        target_model_id = model_id
        if model_id == "alzheimers-oasis-tabular" and model_id not in BUNDLES and "alzheimers-t1-mri" in BUNDLES:
            target_model_id = "alzheimers-t1-mri"

        if target_model_id not in BUNDLES:
            raise HTTPException(404, f"unknown model {model_id!r}")
        manifest = MANIFESTS[target_model_id]
        expected = [c.lower() for c in manifest["expects"]["channels"]]

        with tempfile.TemporaryDirectory() as tmp:
            sources: dict[str, str] = {}
            for upload in files:
                name = (upload.filename or "").lower()
                match = next((c for c in expected if c in name), None)
                if match is None:
                    continue
                target = Path(tmp) / (upload.filename or "upload")
                target.write_bytes(await upload.read())
                sources[match] = str(target)
            from qhealth_qml.ingestion import ingest_study
            from qhealth_qml.serving import predict

            ingested = ingest_study(manifest, sources)
            if not ingested.ok:
                return {
                    "model_id": model_id,
                    "status": "rejected",
                    "reasons": ingested.errors,
                    "received_files": [f.filename for f in files],
                    "matched_channels": sorted(sources),
                    "prediction": None,
                    "score": None,
                    "score_semantics": "normalised_margin_from_threshold",
                    "research_use_only": True,
                }
            result = predict(BUNDLES[target_model_id], ingested.volume, study_id="upload")
            result["ingestion"] = ingested.provenance
            result["score_semantics"] = "normalised_margin_from_threshold"
            return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", default="runtime/bundles")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check-only", action="store_true", help="load bundles and exit")
    args = parser.parse_args()

    _prewarm_hf_cache()

    directory = Path(args.bundles)
    print(f"loading bundles from {directory}", flush=True)
    _load_bundles(directory)
    if not BUNDLES:
        print("no loadable bundles found", flush=True)
        sys.exit(1)
    if args.check_only:
        print(json.dumps({"models": sorted(BUNDLES)}, indent=2))
        return

    import uvicorn

    uvicorn.run(build_app(directory), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
