"""Command-line entry point for the benchmark."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from .experiment import (
    load_breast_cancer_dataset,
    load_csv_dataset,
    load_model_artifact,
    load_prediction_csv,
    load_profile_dataset,
    predict_with_model_artifact,
    run_repeated_experiment,
    write_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", help="numeric CSV; omit to use the public breast-cancer benchmark")
    source.add_argument(
        "--profile",
        type=Path,
        help="JSON early-detection profile; resolves its dataset path relative to the profile",
    )
    parser.add_argument("--target", help="CSV target column")
    parser.add_argument("--positive-label", help="CSV label to treat as the positive class")
    parser.add_argument(
        "--group-column",
        help="optional CSV patient/cohort identifier; keeps groups out of both split sides",
    )
    parser.add_argument(
        "--time-column",
        help="optional CSV timestamp/order column; trains on earlier rows and tests later rows",
    )
    parser.add_argument("--id-column", help="optional CSV row or patient-event identifier")
    parser.add_argument("--site-column", help="optional CSV site/cohort column")
    parser.add_argument(
        "--outcome-time-column",
        help="optional CSV event time used by an early-detection profile",
    )
    parser.add_argument(
        "--subgroup-column",
        action="append",
        default=[],
        help="optional subgroup column; repeat for subgroup/fairness reports",
    )
    parser.add_argument(
        "--models",
        default="classical,qsvc",
        help="comma-separated: classical, qsvc, pegasos_qsvc, vqc (or explicit names)",
    )
    parser.add_argument(
        "--backend",
        choices=["statevector", "aer", "fake", "ibm"],
        default="statevector",
    )
    parser.add_argument("--aer-noise", choices=["none", "fake"], default="none")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=1, help="repeat with consecutive seeds")
    parser.add_argument("--max-train", type=int, default=80, help="0 means no cap")
    parser.add_argument("--max-test", type=int, default=40, help="0 means no cap")
    parser.add_argument("--pegasos-steps", type=int, default=50)
    parser.add_argument("--vqc-maxiter", type=int, default=25)
    parser.add_argument(
        "--reduction",
        choices=["anova", "pca"],
        default=None,
        help="training-only reduction before model fitting; profiles default to PCA",
    )
    parser.add_argument(
        "--holdout-site",
        help="evaluate on exactly one site while training on all other sites",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=None,
        help="validation fraction for threshold selection; custom policies default to 0.2",
    )
    parser.add_argument(
        "--threshold-policy",
        choices=["default", "max_balanced_accuracy", "target_sensitivity"],
        default="default",
    )
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        help="minimum validation sensitivity for target_sensitivity thresholding",
    )
    parser.add_argument(
        "--abstain-margin",
        type=float,
        help="abstain when probability is this close to the selected threshold",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="held-out bootstrap samples for 95%% metric intervals; 0 disables",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="fit sigmoid probabilities without using outer test labels",
    )
    parser.add_argument("--allow-remote-calibration", action="store_true")
    parser.add_argument("--no-explain", action="store_true")
    parser.add_argument("--allow-remote-explanations", action="store_true")
    parser.add_argument(
        "--save-model",
        type=Path,
        help="persist one trained model and its preprocessing policy for inference",
    )
    parser.add_argument(
        "--predict-model",
        type=Path,
        help="load a saved model artifact instead of training",
    )
    parser.add_argument(
        "--predict-csv",
        type=Path,
        help="feature CSV for --predict-model; target column is optional and ignored",
    )
    parser.add_argument("--out", type=Path, help="write the JSON result to this path")
    return parser


def _format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.predict_model:
        if not args.predict_csv:
            parser.error("--predict-model requires --predict-csv")
        if any(
            (
                args.csv,
                args.target,
                args.positive_label,
                args.group_column,
                args.time_column,
                args.id_column,
                args.site_column,
                args.outcome_time_column,
                args.subgroup_column,
                args.save_model,
                args.profile,
            )
        ):
            parser.error("prediction mode cannot be combined with training data options")
        try:
            artifact = load_model_artifact(args.predict_model)
            X, feature_names = load_prediction_csv(
                args.predict_csv,
                artifact.preprocessor.feature_names,
            )
            results = predict_with_model_artifact(
                artifact,
                X,
                feature_names,
                dataset_name=args.predict_csv.name,
                explain=not args.no_explain,
            )
        except (
            FileNotFoundError,
            ImportError,
            RuntimeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            parser.error(str(exc))
        print(
            f"model={results['model_name']} rows={len(results['predictions'])} "
            f"threshold={results['threshold']!s} "
            f"coverage={_format_metric(results['abstention']['coverage'])}"
        )
        if args.out:
            write_results(results, args.out)
            print(f"wrote {args.out}")
        return 0

    if args.profile and any(
        (
            args.target,
            args.positive_label,
            args.group_column,
            args.time_column,
            args.id_column,
            args.site_column,
            args.outcome_time_column,
            args.subgroup_column,
        )
    ):
        parser.error("profile mode gets data columns from the profile JSON")
    if bool(args.csv) != bool(args.target):
        parser.error("--csv and --target must be supplied together")
    if any(
        (
            args.positive_label,
            args.group_column,
            args.time_column,
            args.id_column,
            args.site_column,
            args.outcome_time_column,
            args.subgroup_column,
        )
    ) and not args.csv:
        parser.error("CSV column options require --csv")
    if args.predict_csv:
        parser.error("--predict-csv requires --predict-model")
    if args.save_model and args.repeats != 1:
        parser.error("--save-model supports exactly one run; omit --repeats")

    try:
        dataset = (
            load_profile_dataset(args.profile)
            if args.profile
            else load_csv_dataset(
                args.csv,
                args.target,
                args.positive_label,
                group_column=args.group_column,
                time_column=args.time_column,
                id_column=args.id_column,
                site_column=args.site_column,
                outcome_time_column=args.outcome_time_column,
                subgroup_columns=args.subgroup_column,
            )
            if args.csv
            else load_breast_cancer_dataset()
        )
        reduction = args.reduction or (
            str(dataset.task_profile.get("reduction", "anova"))
            if dataset.task_profile
            else "anova"
        )
        results = run_repeated_experiment(
            dataset=dataset,
            repeats=args.repeats,
            models=args.models.split(","),
            backend=args.backend,
            n_qubits=args.n_qubits,
            shots=args.shots,
            test_size=args.test_size,
            seed=args.seed,
            max_train=args.max_train,
            max_test=args.max_test,
            aer_noise=args.aer_noise,
            pegasos_steps=args.pegasos_steps,
            vqc_maxiter=args.vqc_maxiter,
            calibrate=args.calibrate,
            allow_remote_calibration=args.allow_remote_calibration,
            validation_size=args.validation_size,
            threshold_policy=args.threshold_policy,
            target_sensitivity=args.target_sensitivity,
            abstain_margin=args.abstain_margin,
            bootstrap_samples=args.bootstrap_samples,
            model_artifact_path=args.save_model,
            explain=not args.no_explain,
            allow_remote_explanations=args.allow_remote_explanations,
            reduction=reduction,
            holdout_site=args.holdout_site,
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, pickle.PicklingError) as exc:
        parser.error(str(exc))

    print(
        f"dataset={results['dataset']['name']} "
        f"train={results['split']['train_rows']} "
        f"validation={results['split']['validation_rows']} "
        f"test={results['split']['test_rows']}"
    )
    print(
        f"backend={results['execution'].get('resolved_backend', args.backend)} "
        f"selected={', '.join(results['preprocessing']['selected_features'])}"
    )
    repeated = results.get("repeated_evaluation")
    if repeated:
        for name, summary in repeated["metric_summary"].items():
            print(
                f"{name:24s} accuracy={_format_metric(summary['accuracy']['mean'])}"
                f"±{_format_metric(summary['accuracy']['std'])} "
                f"balanced={_format_metric(summary['balanced_accuracy']['mean'])}"
                f"±{_format_metric(summary['balanced_accuracy']['std'])} "
                f"sensitivity={_format_metric(summary['sensitivity']['mean']):>5} "
                f"specificity={_format_metric(summary['specificity']['mean']):>5} "
                f"auc={_format_metric(summary['roc_auc']['mean']):>5} "
                f"pr_auc={_format_metric(summary['pr_auc']['mean']):>5} "
                f"brier={_format_metric(summary['brier_score']['mean']):>5} "
                f"coverage={_format_metric(summary['coverage']['mean']):>5}"
            )
    else:
        for name, result in results["models"].items():
            metrics = result["metrics"]
            print(
                f"{name:24s} accuracy={_format_metric(metrics['accuracy'])} "
                f"balanced={_format_metric(metrics['balanced_accuracy'])} "
                f"sensitivity={_format_metric(metrics['sensitivity']):>5} "
                f"specificity={_format_metric(metrics['specificity']):>5} "
                f"auc={_format_metric(metrics['roc_auc']):>5} "
                f"pr_auc={_format_metric(metrics['pr_auc']):>5} "
                f"brier={_format_metric(metrics['brier_score']):>5} "
                f"seconds={result['elapsed_seconds']:.2f}"
            )
    if args.out:
        write_results(results, args.out)
        print(f"wrote {args.out}")
    return 0
