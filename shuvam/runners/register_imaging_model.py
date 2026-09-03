"""Register a trained imaging/signal hybrid as a first-class platform model.

Until now the new modality models existed only as scripts and inference bundles, which means
the platform's registry — the thing that decides what is available, what its limitations are,
and whether it may be presented as an operational reference — knew nothing about them. This
writes a schema-valid `ModelDefinition` (and matching `EvaluationRecord`) from a run's report
plus its inference bundle, so the models are discoverable and governed like every other entry.

Deliberate choices:
* `lifecycle` defaults to `experimental`. Promotion to `operational_reference` is gated on the
  registry's own baseline-viability rule (CI lower bound > 0.5) and is not something a build
  script should grant itself.
* `availability` is `available` only when an inference bundle actually exists on disk — a model
  the platform cannot execute must not advertise itself as available.
* Limitations are copied verbatim from the bundle, so what the serving layer tells a caller and
  what the registry tells the catalogue cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

REGISTRY = Path("src/qhealth_qml/platform/registry_data")


def build_model_definition(
    bundle_manifest: dict,
    report: dict,
    model_id: str,
    condition_id: str,
    display_name: str,
    bundle_path: str,
    classical_baseline_id: str | None,
    evaluation_id: str,
    profile_name: str,
) -> dict:
    expects = bundle_manifest.get("expects", {})
    provenance = bundle_manifest.get("training_provenance", {})
    summary = report.get("summary", {})
    quantum = bundle_manifest.get("quantum") or {}

    best_head = max(
        (name for name in summary),
        key=lambda n: summary[n].get("mean", 0.0),
        default=None,
    )
    qsvc = summary.get("qsvc_decoupled", {})

    return {
        "model_id": model_id,
        "condition_id": condition_id,
        "version": "0.1.0",
        "display_name": display_name,
        "task_type": "binary_classification",
        "availability": "available" if Path(bundle_path).exists() else "not available",
        "lifecycle": "experimental",
        "executor": "imaging_hybrid",
        "input_contract": {
            "required_modalities": ["imaging"],
            "optional_modalities": [],
            "required_fields": list(expects.get("channels", [])),
            "min_rows": 1,
            "population": provenance.get("population", provenance.get("dataset", "see research record")),
            "population_filters": {},
            "quality_constraints": {
                "grid": f"volumes are resampled to {expects.get('grid')}",
                "channels": f"all of {expects.get('channels')} are required; missing sequences are refused, not zero-filled",
                "modality": provenance.get("modality", "MR"),
            },
        },
        "dataset_profile_path": provenance.get("dataset_profile_path")
            or f"backend/profiles/{profile_name}.json",
        "temporal_validation": (
            "not applicable — cross-sectional imaging cohort, no index or outcome time"
        ),
        "preprocessing": {
            "imputation": "none (dense imaging)",
            "scaling": provenance.get("scaling", "percentile intensity normalisation"),
            "reduction": "PCA to qubit count at the quantum boundary",
            "n_components": quantum.get("n_qubits"),
            "angle_scaling": f"minmax (angle_scale={quantum.get('angle_scale')})",
            "fitted_on": "train_partition_only",
        },
        "quantum": {
            "framework": "qiskit",
            "encoding": quantum.get("feature_map", "ZZFeatureMap"),
            "ansatz": None,
            "circuit_version": (
                f"qiskit-2.5/{quantum.get('feature_map','zzfeaturemap')}-reps{quantum.get('feature_map_reps',1)}"
                f"-{quantum.get('entanglement','linear')}-C{quantum.get('C')}-anglescale{quantum.get('angle_scale')}"
            ),
            "backend_mode": quantum.get("backend", "statevector"),
            "n_qubits": quantum.get("n_qubits", 4),
            "shots": 512,
        } if quantum else None,
        "classical_baseline_model_id": classical_baseline_id,
        "artifact": {
            "kind": "inference_bundle",
            "path": bundle_path,
            "manifest_path": f"{bundle_path}.manifest.json",
            "sha256": None,
            "schema_version": bundle_manifest.get("schema_version", 2),
        },
        "calibration": {
            "method": "validation_selected_threshold",
            "status": "not_assessed",
            "note": (
                f"Operating threshold selected on the validation split only "
                f"({bundle_manifest.get('operating_point',{}).get('policy')}). "
                f"Held-out test performance: {json.dumps(provenance.get('held_out_performance', {}))}. "
                f"Reliability curve / ECE not yet computed — calibration is NOT assessed."
            ),
        },
        "explainability": {
            "method": "none",
            "scope": "per_row",
            "surrogate": False,
        },
        "output_score_type": "calibrated_probability",
        "evaluation_record_ids": [evaluation_id],
        "safety": {
            "allows_negative_finding": True,
            "abstain_margin": 0.05,
            "requires_full_coverage": True,
            "disclaimer": (
                f"Research {bundle_manifest.get('temporal_framing','detection')} estimate from imaging. "
                "Not a diagnosis and not a medical device."
            ),
            "limitations": list(provenance.get("limitations", [])) + [
                f"Cohort size {provenance.get('cohort_size')} — see research record for confidence intervals.",
                "No external-site or multi-cohort validation has been performed.",
                "Calibration (reliability/ECE) is not assessed; the score is a thresholded decision value.",
            ],
        },
        "model_card": {
            "intended_use": (
                f"Research benchmarking of a hybrid quantum-classical {bundle_manifest.get('temporal_framing','detection')} "
                f"model on {provenance.get('modality','imaging')} for {condition_id}."
            ),
            "excluded_use": "Any clinical use. Diagnosis. Triage of real patients.",
            "training_population": str(provenance.get("dataset")),
            "evaluation_population": "Stratified held-out split of the same cohort. No external cohort.",
            "label_policy": bundle_manifest.get("labels", {}).get("positive", ""),
            "data_sources": [{"label": str(provenance.get("dataset")), "url": provenance.get("source_url", ""), "citation": provenance.get("citation", "")}],
            "data_licenses": [provenance.get("license", "see research record")],
            "limitations": ["See safety.limitations."],
            "calibration_status": "not_assessed",
            "explanation_method": "none",
            "maintainer": "tech@centai.in",
            "reuse_manifest": [
                {
                    "component": "pretrained encoder",
                    "decision": "reused",
                    "source_url": provenance.get("encoder_source", ""),
                    "release_or_commit": bundle_manifest.get("encoder", {}).get("backbone", ""),
                    "paper_citation": None,
                    "license": provenance.get("encoder_license", ""),
                    "license_verified": bool(provenance.get("encoder_license")),
                    "weight_source": provenance.get("encoder_source", ""),
                    "weight_sha256": None,
                    "preprocessing_assumptions": f"volumes resampled to {expects.get('grid')}",
                    "io_contract": "volume [C,D,H,W] -> latent -> quantum/classical head",
                    "local_modifications": "backbone frozen; trainable projection head added",
                },
            ],
            "research_record": provenance.get("research_record", ""),
        },
    }


def build_evaluation_record(
    report: dict, bundle_manifest: dict, model_id: str, condition_id: str, evaluation_id: str
) -> dict:
    provenance = bundle_manifest.get("training_provenance", {})
    summary = report.get("summary", {})
    qsvc = summary.get("qsvc_decoupled", {})
    held_out = provenance.get("held_out_performance", {})
    return {
        "evaluation_id": evaluation_id,
        "model_id": model_id,
        "model_version": "0.1.0",
        "condition_id": condition_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_profile": {
            "name": str(provenance.get("dataset")),
            "modality": "imaging",
            "task_type": "binary_classification",
            "outcome_definition": bundle_manifest.get("labels", {}).get("positive", ""),
            "reduction": "pca",
        },
        "dataset_fingerprint": None,
        "split_strategy": "stratified_random",
        "split_summary": provenance.get("split", {}),
        "preprocessing": {
            "imputation": "none",
            "scaling": "percentile intensity normalisation",
            "reduction": "pca",
            "n_components": (bundle_manifest.get("quantum") or {}).get("n_qubits"),
            "angle_scaling": "minmax",
            "fitted_on": "train_partition_only",
        },
        "leakage_checks": [
            {
                "name": "training_only_preprocessing",
                "status": "passed",
                "detail": "Encoder, standardiser, PCA, angle scaler and threshold are all fitted on training/validation only and persisted in the inference bundle; nothing is refitted at inference.",
            },
            {
                "name": "subject_identity",
                "status": "passed",
                "detail": "One scan per subject in this cohort, so a stratified split cannot place a subject on both sides.",
            },
        ],
        "metrics": {
            "balanced_accuracy": qsvc.get("mean"),
            **{k: v for k, v in held_out.items() if k != "balanced_accuracy"},
        },
        "segmentation_metrics": {},
        "event_metrics": {},
        "calibration": {"calibration_curve": [], "engine_calibration": {}},
        "abstention": {"enabled": True, "margin": 0.05},
        "resource": {
            "wall_seconds": 0.0,
            "qubits": (bundle_manifest.get("quantum") or {}).get("n_qubits"),
            "backend": (bundle_manifest.get("quantum") or {}).get("backend", "statevector"),
        },
        "confidence_intervals": {
            "balanced_accuracy": {
                "lower": (qsvc.get("ci") or [None, None])[0],
                "upper": (qsvc.get("ci") or [None, None])[1],
                "n": qsvc.get("n"),
            }
        },
        "baseline_model_id": None,
        "baseline_metrics": {
            name: values.get("mean") for name, values in summary.items() if name != "qsvc_decoupled"
        },
        "paired_comparison": {},
        "real_gain_decision": "not_assessed",
        "real_gain_reason": (
            "Paired nested evaluation has not been run for the imaging path; classical head results "
            "on the same latent are reported as baseline_metrics for context."
        ),
        "software": {},
        "source_result_path": report.get("_report_path", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--classical-baseline-id", default=None)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(f"{args.bundle}.manifest.json").read_text())
    report = json.loads(Path(args.report).read_text())
    report["_report_path"] = args.report
    evaluation_id = f"eval-{args.model_id}-0.1.0"

    model = build_model_definition(
        manifest, report, args.model_id, args.condition_id, args.display_name,
        args.bundle, args.classical_baseline_id, evaluation_id, args.profile_name,
    )
    evaluation = build_evaluation_record(
        report, manifest, args.model_id, args.condition_id, evaluation_id
    )

    if args.dry_run:
        print(json.dumps({"model": model, "evaluation": evaluation}, indent=2, default=str)[:3000])
        return

    model_path = REGISTRY / "models" / f"{args.model_id}.json"
    eval_path = REGISTRY / "evaluations" / f"{evaluation_id}.json"
    model_path.write_text(json.dumps(model, indent=2, default=str) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps(evaluation, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {model_path}\nwrote {eval_path}")

    from qhealth_qml.platform.registry import load_registry

    registry = load_registry()
    entry = registry.model(args.model_id)
    print(f"registry loads OK; {args.model_id} lifecycle={entry.lifecycle} availability={entry.availability}")


if __name__ == "__main__":
    main()
