"""Local-only dashboard server for research artifacts and saved models."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .experiment import (
    dataset_fingerprint,
    load_breast_cancer_dataset,
    load_csv_dataset,
    load_model_artifact,
    load_prediction_csv,
    predict_with_model_artifact,
    run_repeated_experiment,
    write_results,
)
from .platform.case_bundle import build_bundle, validate_bundle
from .platform.execution import run_assessment
from .platform.registry import load_registry
from .platform.routing import route


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _prediction_rows(path: Path, feature_names: list[str]) -> list[dict[str, Any]]:
    X, _ = load_prediction_csv(path, feature_names)
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            features = {
                name: (float(row[name]) if str(row.get(name, "")).strip() else None)
                for name in feature_names
            }
            rows.append({"row_id": str(row.get("record_id") or row.get("id") or f"row-{index + 1}"), "features": features})
    if len(rows) != len(X):
        raise ValueError("prediction CSV row count changed while it was being read")
    return rows


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve static dashboard assets and narrowly scoped local JSON APIs."""

    config: dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, value: Any, status: int = 200) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(
            "Access-Control-Allow-Origin",
            str(self.config.get("cors_origin", "*")),
        )
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(payload)

    def _request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _train(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run one local experiment and refresh the in-memory dashboard state."""

        lock = self.config["train_lock"]
        if not lock.acquire(blocking=False):
            raise ValueError("a training run is already in progress")

        temporary_path: Path | None = None
        try:
            source = str(body.get("source", "wdbc"))
            dataset_name = str(body.get("dataset_name", "uploaded.csv"))
            models_value = body.get(
                "models", ["logistic_regression", "rbf_svc", "qsvc"]
            )
            if isinstance(models_value, str):
                models = [name.strip() for name in models_value.split(",") if name.strip()]
            elif isinstance(models_value, list):
                models = [str(name).strip() for name in models_value if str(name).strip()]
            else:
                raise ValueError("models must be a list or comma-separated string")
            if not models:
                raise ValueError("at least one model is required")
            if source == "wdbc":
                dataset = load_breast_cancer_dataset()
            elif source == "csv":
                csv_text = body.get("csv_text")
                target = str(body.get("target", "")).strip()
                if not isinstance(csv_text, str) or not csv_text.strip():
                    raise ValueError("csv_text is required for CSV training")
                if not target:
                    raise ValueError("target is required for CSV training")
                if len(csv_text.encode("utf-8")) > 10 * 1024 * 1024:
                    raise ValueError("uploaded CSV exceeds the 10 MB local training limit")
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".csv",
                    prefix="neutral-upload-",
                    delete=False,
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    handle.write(csv_text)
                    temporary_path = Path(handle.name)
                dataset = load_csv_dataset(
                    temporary_path,
                    target=target,
                    positive_label=(str(body.get("positive_label", "")).strip() or None),
                    group_column=(str(body.get("group_column", "")).strip() or None),
                    time_column=(str(body.get("time_column", "")).strip() or None),
                    id_column=(str(body.get("id_column", "")).strip() or None),
                    site_column=(str(body.get("site_column", "")).strip() or None),
                    subgroup_columns=[
                        str(value).strip()
                        for value in body.get("subgroup_columns", [])
                        if str(value).strip()
                    ],
                    outcome_time_column=(
                        str(body.get("outcome_time_column", "")).strip() or None
                    ),
                    leakage_columns=[
                        str(value).strip()
                        for value in body.get("leakage_columns", [])
                        if str(value).strip()
                    ],
                )
                from dataclasses import replace

                dataset = replace(dataset, name=dataset_name)
            else:
                raise ValueError("source must be 'wdbc' or 'csv'")

            backend = str(body.get("backend", "statevector"))
            if backend not in {"statevector", "aer", "fake", "ibm"}:
                raise ValueError("backend must be statevector, aer, fake, or ibm")
            n_qubits = int(body.get("n_qubits", 4))
            shots = int(body.get("shots", 256))
            max_train = int(body.get("max_train", 80))
            max_test = int(body.get("max_test", 40))
            if not 2 <= n_qubits <= 12:
                raise ValueError("n_qubits must be between 2 and 12")
            if shots < 1 or max_train < 0 or max_test < 0:
                raise ValueError("shots, max_train, and max_test must be non-negative; shots must be positive")
            selection_repeats = int(body.get("selection_repeats", 1))
            if not 1 <= selection_repeats <= 10:
                raise ValueError("selection_repeats must be between 1 and 10")

            runtime_dir = Path(self.config["runtime_dir"])
            runtime_dir.mkdir(parents=True, exist_ok=True)
            result_path = Path(self.config["result_path"])
            model_path = Path(self.config["default_model_path"])
            result = run_repeated_experiment(
                dataset=dataset,
                repeats=selection_repeats,
                models=models,
                backend=backend,
                n_qubits=n_qubits,
                shots=shots,
                test_size=float(body.get("test_size", 0.2)),
                seed=int(body.get("seed", 7)),
                max_train=max_train,
                max_test=max_test,
                aer_noise=str(body.get("aer_noise", "none")),
                pegasos_steps=int(body.get("pegasos_steps", 50)),
                vqc_maxiter=int(body.get("vqc_maxiter", 25)),
                calibrate=bool(body.get("calibrate", False)),
                validation_size=(
                    float(body["validation_size"])
                    if body.get("validation_size") is not None
                    else None
                ),
                threshold_policy=str(body.get("threshold_policy", "default")),
                target_sensitivity=(
                    float(body["target_sensitivity"])
                    if body.get("target_sensitivity") is not None
                    else None
                ),
                abstain_margin=(
                    float(body["abstain_margin"])
                    if body.get("abstain_margin") is not None
                    else None
                ),
                bootstrap_samples=int(body.get("bootstrap_samples", 0)),
                explain=False,
                reduction=str(body.get("reduction", "anova")),
                holdout_site=(str(body.get("holdout_site", "")).strip() or None),
            )

            model_results = result.get("models", {})
            if not model_results:
                raise ValueError("training returned no model results")
            repeated_summary = result.get("repeated_evaluation", {}).get(
                "metric_summary", {}
            )
            selection_metric = str(body.get("selection_metric", "balanced_accuracy"))

            def selection_score(model_name: str) -> float:
                summary = repeated_summary.get(model_name, {})
                metric = summary.get(selection_metric, {})
                value = metric.get("mean")
                if value is None:
                    value = summary.get("balanced_accuracy", {}).get("mean")
                return float(value) if value is not None else float("-inf")

            best_model = max(
                model_results,
                key=selection_score
                if repeated_summary
                else lambda name: float(
                    model_results[name]["metrics"].get(selection_metric)
                    or model_results[name]["metrics"].get("balanced_accuracy")
                    or 0.0
                ),
            )
            saved = run_repeated_experiment(
                dataset=dataset,
                repeats=1,
                models=[best_model],
                backend=backend,
                n_qubits=n_qubits,
                shots=shots,
                test_size=float(body.get("test_size", 0.2)),
                seed=int(body.get("seed", 7)),
                max_train=max_train,
                max_test=max_test,
                aer_noise=str(body.get("aer_noise", "none")),
                pegasos_steps=int(body.get("pegasos_steps", 50)),
                vqc_maxiter=int(body.get("vqc_maxiter", 25)),
                calibrate=bool(body.get("calibrate", False)),
                validation_size=(
                    float(body["validation_size"])
                    if body.get("validation_size") is not None
                    else None
                ),
                threshold_policy=str(body.get("threshold_policy", "default")),
                target_sensitivity=(
                    float(body["target_sensitivity"])
                    if body.get("target_sensitivity") is not None
                    else None
                ),
                abstain_margin=(
                    float(body["abstain_margin"])
                    if body.get("abstain_margin") is not None
                    else None
                ),
                bootstrap_samples=int(body.get("bootstrap_samples", 0)),
                model_artifact_path=model_path,
                explain=False,
                reduction=str(body.get("reduction", "anova")),
                holdout_site=(str(body.get("holdout_site", "")).strip() or None),
            )
            result["model_artifact"] = saved["model_artifact"]
            result["model_artifact"]["selected_for_inference"] = best_model
            result["study"] = {
                "type": "local_training_run",
                "selected_for_inference": best_model,
                "models_compared": list(model_results),
                "selection_metric": selection_metric,
                "selection_repeats": selection_repeats,
                "selection_policy": (
                    "repeated_holdout_mean"
                    if repeated_summary
                    else "single_holdout_legacy"
                ),
            }
            write_results(result, result_path)
            self.config["result"] = result
            self.config["artifact"] = load_model_artifact(model_path)
            return result
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            lock.release()

    def _registry(self):
        registry = self.config.get("registry")
        if registry is None:
            registry = load_registry()
            self.config["registry"] = registry
        return registry

    def _bundle_from_upload(
        self,
        body: dict[str, Any],
        *,
        upload_dir: Path | None = None,
    ):
        """Build+validate a DataBundle from an upload spec with inline asset content.

        Each asset in `body["assets"]` carries `content` (text) rather than a
        server-local `path`, since the caller is a browser upload, not a fixture
        file. Content is written to a temp file under the runtime directory by
        default, where a later `POST /api/assess` call can read the same asset
        `uri` again for single-case inference. Callers that do not need a later
        assessment can provide an ephemeral directory instead.
        """

        upload_dir = upload_dir or Path(self.config["runtime_dir"]) / "bundle_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        spec = dict(body)
        assets = []
        for asset_spec in body.get("assets", []):
            asset_spec = dict(asset_spec)
            content = asset_spec.pop("content", None)
            if content is not None:
                fmt = str(asset_spec.get("format", "csv"))
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=f".{fmt}", prefix="bundle-upload-",
                    delete=False, dir=upload_dir, encoding="utf-8", newline="",
                ) as handle:
                    handle.write(content)
                    asset_spec["path"] = handle.name
            assets.append(asset_spec)
        spec["assets"] = assets
        bundle = build_bundle(spec, synthetic=bool(body.get("synthetic", False)))
        return validate_bundle(bundle, self._registry())

    def _validate_ehr_cohort(self, body: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="ehr-validation-") as directory:
            return self._validate_ehr_cohort_in_directory(body, Path(directory))

    def _validate_ehr_cohort_in_directory(
        self,
        body: dict[str, Any],
        upload_dir: Path,
    ) -> dict[str, Any]:
        """Validate one adapter-produced EHR cohort without running inference."""

        csv_text = body.get("csv_text")
        if not isinstance(csv_text, str) or not csv_text.strip():
            raise ValueError("csv_text is required; send the adapter's canonical CSV")
        if len(csv_text.encode("utf-8")) > 10 * 1024 * 1024:
            raise ValueError("canonical EHR CSV exceeds the 10 MB local validation limit")

        target = str(body.get("target", "label")).strip()
        if not target:
            raise ValueError("target is required")

        bundle = self._bundle_from_upload(
            {
                "case_id": str(body.get("dataset_name", "ehr-validation")),
                "source": "ehr-validation",
                "synthetic": bool(body.get("synthetic", False)),
                "assets": [
                    {
                        "content": csv_text,
                        "format": "csv",
                        "modality": "structured_clinical",
                        "role": "canonical_ehr_cohort",
                    }
                ],
            },
            upload_dir=upload_dir,
        )
        if len(bundle.assets) != 1:
            raise ValueError("EHR validation expects exactly one canonical cohort asset")

        asset = bundle.assets[0]
        if asset.validation_status != "accepted":
            messages = "; ".join(issue.message for issue in asset.validation_issues)
            raise ValueError(messages or "canonical cohort failed asset validation")

        dataset = load_csv_dataset(
            asset.uri,
            target=target,
            positive_label=(str(body.get("positive_label", "")).strip() or None),
            group_column=(str(body.get("group_column", "")).strip() or None),
            time_column=(str(body.get("time_column", "")).strip() or None),
            id_column=(str(body.get("id_column", "")).strip() or None),
            site_column=(str(body.get("site_column", "")).strip() or None),
            outcome_time_column=(str(body.get("outcome_time_column", "")).strip() or None),
            subgroup_columns=[
                str(value).strip()
                for value in body.get("subgroup_columns", [])
                if str(value).strip()
            ],
            leakage_columns=[
                str(value).strip()
                for value in body.get("leakage_columns", [])
                if str(value).strip()
            ],
        )
        routing = route(bundle, self._registry(), preview=True)
        labels, counts = np.unique(dataset.y, return_counts=True)
        label_counts = {str(int(label)): int(count) for label, count in zip(labels, counts)}
        missing_cells = int(np.isnan(dataset.X).sum())
        compatible = sum(decision.status == "compatible" for decision in routing)
        bundle_payload = bundle.to_dict()
        for asset_payload in bundle_payload.get("assets", []):
            asset_payload["uri"] = "ephemeral://ehr-validation"

        return {
            "status": "passed",
            "endpoint": "/api/validation/ehr",
            "source_format": str(body.get("source_format", "canonical_csv")),
            "dataset": {
                "name": str(body.get("dataset_name") or dataset.name),
                "rows": int(dataset.X.shape[0]),
                "features": int(dataset.X.shape[1]),
                "feature_names": dataset.feature_names,
                "positive_label": dataset.positive_label,
                "negative_label": dataset.negative_label,
                "label_counts": label_counts,
                "missing_cells": missing_cells,
                "fingerprint": dataset_fingerprint(dataset),
            },
            "checks": [
                {
                    "name": "canonical_matrix",
                    "status": "passed",
                    "detail": f"loaded {dataset.X.shape[0]} rows x {dataset.X.shape[1]} numeric features",
                },
                {
                    "name": "binary_label",
                    "status": "passed",
                    "detail": f"found {len(labels)} label classes",
                },
                {
                    "name": "missing_values",
                    "status": "warning" if missing_cells else "passed",
                    "detail": f"{missing_cells} missing feature cell(s); the training pipeline will fit imputation on its training partition",
                },
                {
                    "name": "model_routing",
                    "status": "passed" if compatible else "warning",
                    "detail": f"{compatible} registered neurological model(s) match this cohort contract",
                },
            ],
            "bundle": bundle_payload,
            "routing": [decision.to_dict() for decision in routing],
            "disclaimer": (
                "Cohort contract validation only. No model was trained and no patient-level "
                "diagnosis or inference was produced."
            ),
        }

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", str(self.config.get("cors_origin", "*")))
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/result":
            self._send_json(self.config.get("result", {}))
            return
        if route == "/api/model":
            artifact = self.config.get("artifact")
            if artifact is None:
                self._send_json({"available": False})
            else:
                self._send_json(
                    {
                        "available": True,
                        "model_name": artifact.model_name,
                        "feature_names": artifact.preprocessor.feature_names,
                        "selected_features": artifact.selected_features,
                        "feature_space": artifact.feature_space,
                        "threshold": artifact.threshold,
                        "threshold_policy": artifact.threshold_policy,
                        "abstain_margin": artifact.abstain_margin,
                        "hardware_probe": artifact.hardware_probe,
                        "dataset": artifact.dataset,
                    }
                )
            return
        if route == "/api/samples":
            self._send_json({"rows": self.config.get("sample_rows", [])})
            return
        if route == "/api/catalog":
            try:
                catalog = self._registry().catalog()
                self._send_json(
                    [
                        {
                            "condition": entry.condition.to_dict(),
                            "models": [m.to_dict() for m in entry.models],
                            "availability": entry.availability,
                        }
                        for entry in catalog
                    ]
                )
            except (OSError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if route == "/api/assessment":
            query_params = parse_qs(urlparse(self.path).query)
            run_id = (query_params.get("run_id") or [""])[0]
            run = self.config.get("assessment_runs", {}).get(run_id)
            if run is None:
                self._send_json({"error": "unknown run_id"}, status=404)
            else:
                self._send_json(run.to_dict())
            return
        if route == "/ehr-validation" and self.config.get("ui_dir"):
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/validation/ehr":
            try:
                self._send_json(self._validate_ehr_cohort(self._request_json()))
            except (
                OSError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if route == "/api/bundle":
            try:
                bundle = self._bundle_from_upload(self._request_json())
                self.config.setdefault("bundles", {})[bundle.bundle_id] = bundle
                self._send_json(bundle.to_dict())
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if route == "/api/assess":
            try:
                body = self._request_json()
                bundle_id = str(body.get("bundle_id", ""))
                bundle = self.config.get("bundles", {}).get(bundle_id)
                if bundle is None:
                    raise ValueError(f"unknown bundle_id {bundle_id!r}; call POST /api/bundle first")
                run = run_assessment(
                    bundle, self._registry(),
                    mode=str(body.get("mode", "research")),
                    runtime_dir=Path(self.config["runtime_dir"]),
                )
                self.config.setdefault("assessment_runs", {})[run.run_id] = run
                self._send_json(run.to_dict())
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if route == "/api/train":
            try:
                self._send_json(self._train(self._request_json()))
            except (
                FileNotFoundError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if route != "/api/predict":
            self._send_json({"error": "route not found"}, status=404)
            return
        artifact = self.config.get("artifact")
        if artifact is None:
            self._send_json({"error": "start the dashboard with --model to enable prediction"}, status=400)
            return
        try:
            body = self._request_json()
            features = body.get("features")
            expected = artifact.preprocessor.feature_names
            if isinstance(features, dict):
                missing = [name for name in expected if name not in features]
                if missing:
                    raise ValueError(f"missing model features: {', '.join(missing)}")
                values = [features[name] for name in expected]
            elif isinstance(features, list):
                values = features
            else:
                raise ValueError("features must be an object or ordered list")
            numeric = np.asarray(
                [[float(value) if value is not None else np.nan for value in values]],
                dtype=float,
            )
            result = predict_with_model_artifact(
                artifact,
                numeric,
                expected,
                dataset_name="dashboard_input",
                row_ids=[str(body.get("row_id", "dashboard-row"))],
                explain=bool(body.get("explain", True)),
            )
            self._send_json(result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)


def _handler(config: dict[str, Any], directory: Path):
    class ConfiguredDashboardHandler(DashboardHandler):
        pass

    ConfiguredDashboardHandler.config = config
    return partial(ConfiguredDashboardHandler, directory=str(directory))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("artifacts/demo.json"),
        help="JSON experiment or study artifact",
    )
    parser.add_argument("--model", type=Path, help="saved .pkl model for local prediction")
    parser.add_argument("--prediction-csv", type=Path, help="sample feature rows for the prediction panel")
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("runs"),
        help="local directory for models and results created by /api/train",
    )
    parser.add_argument(
        "--ui-dir",
        type=Path,
        help="serve a built frontend directory instead of the bundled dashboard",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_path = args.result.resolve()
    artifact = load_model_artifact(args.model) if args.model else None
    samples = (
        _prediction_rows(args.prediction_csv, artifact.preprocessor.feature_names)
        if args.prediction_csv and artifact is not None
        else []
    )
    assets = args.ui_dir.resolve() if args.ui_dir else Path(__file__).with_name("dashboard")
    runtime_dir = args.runtime_dir.resolve()
    default_model_path = args.model.resolve() if args.model else runtime_dir / "neutral-model.pkl"
    config = {
        "result": _read_json(result_path),
        "artifact": artifact,
        "sample_rows": samples,
        "runtime_dir": runtime_dir,
        "result_path": result_path,
        "default_model_path": default_model_path,
        "ui_dir": args.ui_dir is not None,
        "train_lock": threading.Lock(),
        "cors_origin": "*",
    }
    server = ThreadingHTTPServer((args.host, args.port), _handler(config, assets))
    print(f"dashboard=http://{args.host}:{server.server_port}")
    print("local-only server; press Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
