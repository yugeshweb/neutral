#!/usr/bin/env python
"""Console entry point for the cardiovascular frame: inspect, fit, pool.

Installed as `qhealth-cardiovascular`. It exists so the frame can be exercised from a
shell without writing Python, and so an integrator can see a complete worked
invocation before embedding the library form. Lives inside the package rather than as
a loose script so that installing the distribution is all an integrator needs.

Three subcommands, in the order you would actually use them:

  status   What can run right now, and why the rest cannot. Touches no models and
           imports nothing heavy, so it is safe to call on any machine.

  fit      Train a detector for every modality that has data, recording the fusion
           weight each one earned from its own validation split. Modalities without
           data are skipped with a stated reason rather than failing the run.

  pool     Combine per-modality scores for one patient into a single assessment.
           Scores may come from this frame's models or from any external component
           that emits a calibrated probability, which is how a medical LLM reading the
           record or a vendor imaging model participates on equal terms.

Examples:

  qhealth-cardiovascular status
  qhealth-cardiovascular fit --artifact-dir runtime/cvd --max-train 400 --max-test 200
  qhealth-cardiovascular pool --score ecg_12lead=0.81 --score ehr_tabular=0.44
  qhealth-cardiovascular pool --score ecg_12lead=0.81 --score angiography=none
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cardiovascular import CardiovascularFrame


def _emit(payload: object, out: Path | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {out}", file=sys.stderr)


def _build_frame(args: argparse.Namespace) -> CardiovascularFrame:
    sources: dict[str, str] = {}
    for item in getattr(args, "source", None) or []:
        modality, _, path = item.partition("=")
        if not modality or not path:
            raise SystemExit(f"--source expects MODALITY=PATH, got {item!r}")
        sources[modality.strip()] = path.strip()

    options: dict[str, dict[str, object]] = {}
    for item in getattr(args, "option", None) or []:
        modality, _, rest = item.partition(".")
        key, _, value = rest.partition("=")
        if not modality or not key or value == "":
            raise SystemExit(f"--option expects MODALITY.KEY=VALUE, got {item!r}")
        try:
            parsed: object = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        options.setdefault(modality.strip(), {})[key.strip()] = parsed

    return CardiovascularFrame(root=Path(args.root), sources=sources, loader_options=options)


def command_status(args: argparse.Namespace) -> int:
    frame = _build_frame(args)
    _emit(frame.readiness_report(), args.out)
    return 0


def command_fit(args: argparse.Namespace) -> int:
    frame = _build_frame(args)

    models: dict[str, str] = {}
    for item in args.model or []:
        modality, _, name = item.partition("=")
        if not modality or not name:
            raise SystemExit(f"--model expects MODALITY=ESTIMATOR, got {item!r}")
        models[modality.strip()] = name.strip()

    run_kwargs: dict[str, object] = {"seed": args.seed}
    if args.max_train is not None:
        run_kwargs["max_train"] = args.max_train
    if args.max_test is not None:
        run_kwargs["max_test"] = args.max_test
    if args.bootstrap:
        run_kwargs["bootstrap_samples"] = args.bootstrap
    if args.n_qubits is not None:
        run_kwargs["n_qubits"] = args.n_qubits

    outcome = frame.fit_available(
        artifact_dir=args.artifact_dir,
        models=models,
        skip_errors=not args.strict,
        **run_kwargs,
    )

    report = frame.report()
    report["fit_outcome"] = outcome
    _emit(report, args.out)

    if not outcome["fitted"]:
        print(
            "\nNo modality was fitted. Run `status` to see what data each one needs.",
            file=sys.stderr,
        )
        return 1
    return 0


def command_pool(args: argparse.Namespace) -> int:
    frame = _build_frame(args)

    scores: dict[str, float | None] = {}
    for item in args.score or []:
        modality, _, value = item.partition("=")
        modality = modality.strip()
        value = value.strip().lower()
        if not modality or not value:
            raise SystemExit(f"--score expects MODALITY=VALUE, got {item!r}")
        scores[modality] = None if value in {"none", "na", "absent", ""} else float(value)
    if not scores:
        raise SystemExit("pool needs at least one --score MODALITY=VALUE")

    thresholds: dict[str, float] = {}
    for item in args.threshold or []:
        modality, _, value = item.partition("=")
        thresholds[modality.strip()] = float(value)

    weights_loaded = False
    if args.weights is not None:
        weights_loaded = _load_weights(frame, Path(args.weights))

    try:
        payload = frame.pool(scores, threshold=args.decision_threshold, thresholds=thresholds)
    except ValueError as exc:
        # Refusing to pool unweighted evidence is correct, but an integration surface
        # should say why in its own output format rather than raising a traceback.
        _emit(
            {
                "status": "refused",
                "reason": str(exc),
                "remedy": (
                    "Every contributing modality scored zero weight. Fusion weight is "
                    "earned from a modality's validated balanced accuracy, so run "
                    "`fit` first and pass its report via --weights."
                    if not weights_loaded
                    else "No supplied modality had a usable score."
                ),
                "supplied_scores": {name: value for name, value in scores.items()},
                "weights_loaded": weights_loaded,
            },
            args.out,
        )
        return 2

    if not weights_loaded:
        payload["warning"] = (
            "No trained weights were supplied (--weights), so pooled influence rests on "
            "whatever weights this frame already held. Run `fit` and pass its report to "
            "weight each modality by demonstrated skill."
        )
    _emit(payload, args.out)
    return 0


def _load_weights(frame: CardiovascularFrame, path: Path) -> bool:
    """Rehydrate fusion weights from a previous `fit` report.

    Only the weight and model id are restored, not the estimators: pooling externally
    supplied scores needs to know how far to trust each channel, not how to recompute
    it. A report that records no weights leaves the frame untrusted rather than
    defaulting to equal influence.
    """

    from .multimodal import ModalityModel

    if not path.exists():
        raise SystemExit(f"weights report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    weights = report.get("weights") or {}
    if not weights:
        return False
    for modality, entry in weights.items():
        validated = entry.get("validated_balanced_accuracy")
        if validated is None:
            continue
        frame.trained[modality] = ModalityModel(
            modality=modality,
            model_id=entry.get("model_id", f"{modality}:restored"),
            artifact=None,
            validated_balanced_accuracy=float(validated),
        )
    return bool(frame.trained)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared options live on a parent so they are accepted *after* the subcommand,
    # which is where anyone would naturally type them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        default=".",
        help="root that modality data paths resolve against (default: cwd)",
    )
    common.add_argument(
        "--source",
        action="append",
        metavar="MODALITY=PATH",
        help="override a modality's data source (repeatable)",
    )
    common.add_argument(
        "--option",
        action="append",
        metavar="MODALITY.KEY=VALUE",
        help="extra loader argument, JSON-parsed when possible (repeatable)",
    )
    common.add_argument("--out", type=Path, default=None, help="also write JSON here")

    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", parents=[common], help="what can run right now")
    status.set_defaults(func=command_status)

    fit = sub.add_parser("fit", parents=[common], help="train every modality that has data")
    fit.add_argument("--artifact-dir", default="runtime/cvd")
    fit.add_argument("--model", action="append", metavar="MODALITY=ESTIMATOR")
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--max-train", type=int, default=None)
    fit.add_argument("--max-test", type=int, default=None)
    fit.add_argument("--n-qubits", type=int, default=None)
    fit.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for CIs")
    fit.add_argument(
        "--strict",
        action="store_true",
        help="fail the run when any modality errors instead of skipping it",
    )
    fit.set_defaults(func=command_fit)

    pool = sub.add_parser(
        "pool", parents=[common], help="combine per-modality scores for one patient"
    )
    pool.add_argument("--score", action="append", metavar="MODALITY=VALUE|none")
    pool.add_argument("--threshold", action="append", metavar="MODALITY=VALUE")
    pool.add_argument("--decision-threshold", type=float, default=0.5)
    pool.add_argument(
        "--weights",
        default=None,
        help="a previous fit report, so pooled scores are weighted by demonstrated skill",
    )
    pool.set_defaults(func=command_pool)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
