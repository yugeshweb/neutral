"""Command-line entry point for nested studies and resource sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import load_breast_cancer_dataset, load_csv_dataset, load_profile_dataset, write_results
from .study import run_nested_evaluation, run_resource_sweep, write_study_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["nested", "sweep"], required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", help="numeric CSV; omit to use the public benchmark")
    source.add_argument("--profile", type=Path, help="JSON early-detection profile")
    parser.add_argument("--target", help="CSV target column")
    parser.add_argument("--positive-label", help="CSV positive label")
    parser.add_argument("--group-column", help="CSV patient/cohort identifier")
    parser.add_argument("--time-column", help="CSV index time/order column")
    parser.add_argument("--id-column", help="CSV row identifier")
    parser.add_argument("--site-column", help="CSV site/cohort column")
    parser.add_argument("--outcome-time-column", help="CSV outcome time column")
    parser.add_argument("--subgroup-column", action="append", default=[])
    parser.add_argument("--models", default="classical,qsvc")
    parser.add_argument("--backend", default="statevector", choices=["statevector", "aer", "fake", "ibm"])
    parser.add_argument("--aer-noise", default="none", choices=["none", "fake"])
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--inner-test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--inner-repeats", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=80)
    parser.add_argument("--max-test", type=int, default=40)
    parser.add_argument("--pegasos-steps", type=int, default=50)
    parser.add_argument("--vqc-maxiter", type=int, default=25)
    parser.add_argument("--reduction", choices=["anova", "pca"], default=None)
    parser.add_argument("--holdout-site")
    parser.add_argument(
        "--threshold-policy",
        choices=["default", "max_balanced_accuracy", "target_sensitivity"],
        default="default",
    )
    parser.add_argument("--target-sensitivity", type=float)
    parser.add_argument("--validation-size", type=float)
    parser.add_argument("--abstain-margin", type=float)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--no-tune", action="store_true", help="use default model parameters")
    parser.add_argument("--sweep-qubits", default="2,4,6")
    parser.add_argument("--sweep-backends", default="statevector,aer")
    parser.add_argument("--sweep-models", default="rbf_svc,qsvc")
    parser.add_argument("--out", type=Path, default=Path("artifacts/study.json"))
    parser.add_argument("--report", type=Path, help="optional reviewer-friendly HTML report")
    return parser


def _load_dataset(args: argparse.Namespace):
    if args.profile:
        return load_profile_dataset(args.profile)
    if args.csv:
        if not args.target:
            raise ValueError("--target is required with --csv")
        return load_csv_dataset(
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
    if any(
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
        raise ValueError("CSV column options require --csv")
    return load_breast_cancer_dataset()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = _load_dataset(args)
        reduction = args.reduction or (
            str(dataset.task_profile.get("reduction", "anova"))
            if dataset.task_profile
            else "anova"
        )
        if args.mode == "nested":
            results = run_nested_evaluation(
                dataset,
                models=args.models.split(","),
                outer_repeats=args.outer_repeats,
                inner_repeats=args.inner_repeats,
                test_size=args.test_size,
                inner_test_size=args.inner_test_size,
                seed=args.seed,
                backend=args.backend,
                n_qubits=args.n_qubits,
                shots=args.shots,
                aer_noise=args.aer_noise,
                max_train=args.max_train,
                max_test=args.max_test,
                pegasos_steps=args.pegasos_steps,
                vqc_maxiter=args.vqc_maxiter,
                reduction=reduction,
                validation_size=args.validation_size,
                threshold_policy=args.threshold_policy,
                target_sensitivity=args.target_sensitivity,
                abstain_margin=args.abstain_margin,
                bootstrap_samples=args.bootstrap_samples,
                holdout_site=args.holdout_site,
                tune=not args.no_tune,
            )
        else:
            results = run_resource_sweep(
                dataset,
                qubits=[int(value.strip()) for value in args.sweep_qubits.split(",") if value.strip()],
                backends=[value.strip() for value in args.sweep_backends.split(",") if value.strip()],
                models=args.sweep_models.split(","),
                seed=args.seed,
                shots=args.shots,
                test_size=args.test_size,
                max_train=args.max_train,
                max_test=args.max_test,
                aer_noise=args.aer_noise,
                reduction=reduction,
                repeats=args.repeats,
            )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        build_parser().error(str(exc))
    write_results(results, args.out)
    if args.report:
        write_study_report(results, args.report)
    print(f"wrote {args.out}")
    if args.report:
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
