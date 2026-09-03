"""Raw PTB-XL ECG ingestion and deterministic feature extraction.

The benchmark consumes WFDB records, not beat-level CSV rows.  Feature
extraction is deliberately small and deterministic so the existing classical
and Qiskit experiment runner can evaluate the same representation.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .experiment import LoadedDataset


LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PER_LEAD_FEATURES = (
    "raw_mean",
    "raw_std",
    "shape_min",
    "shape_max",
    "shape_ptp",
    "shape_rms",
    "shape_skew",
    "shape_kurtosis",
    "derivative_std",
    "second_derivative_std",
    "zero_crossings",
    "dominant_frequency",
    "spectral_entropy",
    "band_0p5_4",
    "band_4_8",
    "band_8_15",
    "band_15_40",
    "band_40_100",
)


def _safe_float(value: str | None) -> float:
    try:
        parsed = float(value or "")
    except ValueError:
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _diagnostic_classes(
    raw_codes: str,
    statement_classes: dict[str, str],
) -> set[str]:
    try:
        codes = ast.literal_eval(raw_codes)
    except (SyntaxError, ValueError):
        return set()
    if not isinstance(codes, dict):
        return set()
    return {
        statement_classes[code]
        for code in codes
        if code in statement_classes and statement_classes[code]
    }


def _read_statement_classes(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row.get("", "")).strip(): str(row.get("diagnostic_class", "")).strip()
            for row in rows
            if row.get("") and row.get("diagnostic_class")
        }


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


ACUTE_MI_STAGES = ("i", "i-ii")

# PTB-XL de-identifies every age above 89 by storing it as the literal 300.
# Left raw, those 293 records stretch the MinMax angle scaler so far that the
# real 2-89 range collapses into a sliver of the quantum encoding range.
PTBXL_AGE_CEILING = 89.0

# Bump whenever the extracted feature matrix changes meaning, so previously cached
# matrices are rejected instead of silently reused. v2 clamps the age sentinel.
FEATURE_SCHEMA_VERSION = 2


def ptbxl_age(raw_age: str | None) -> float:
    age = _safe_float(raw_age)
    if math.isfinite(age) and age > PTBXL_AGE_CEILING:
        return PTBXL_AGE_CEILING
    return age


def _infarction_stage(raw_stage: str | None) -> str:
    """Normalise PTB-XL's `infarction_stadium1` to a bare roman-numeral stage.

    Values arrive as free text such as "Stadium I", "Stadium II-III" or empty.
    """

    text = str(raw_stage or "").strip().lower()
    if not text:
        return ""
    return text.replace("stadium", "").replace("stage", "").strip()


def select_ptbxl_rows(
    metadata_csv: str | Path,
    statements_csv: str | Path,
    target: str = "abnormal",
    max_records: int = 0,
    human_validated_only: bool = True,
    seed: int = 7,
    mi_stages: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic, fold-balanced rows for a binary PTB-XL task.

    `mi_stages` restricts which MI infarction stages count as positive for the
    `mi` target, e.g. `ACUTE_MI_STAGES` to detect acute infarction only. MI
    records outside the requested stages are dropped rather than relabelled
    negative: a healed infarct is not a normal ECG, so moving it into the
    negative class would poison it. `None` keeps every MI stage (and records
    with no recorded stage), which is the coarse original behaviour.
    """

    if target not in {"abnormal", "mi"}:
        raise ValueError("target must be 'abnormal' or 'mi'")
    if mi_stages is not None and target != "mi":
        raise ValueError("mi_stages only applies to the 'mi' target")
    wanted_stages = (
        {_infarction_stage(stage) for stage in mi_stages} if mi_stages is not None else None
    )
    statement_classes = _read_statement_classes(Path(statements_csv))
    selected: list[dict[str, Any]] = []
    for row in _read_metadata(Path(metadata_csv)):
        if human_validated_only and row.get("validated_by_human", "").lower() != "true":
            continue
        classes = _diagnostic_classes(row.get("scp_codes", ""), statement_classes)
        if target == "abnormal":
            if classes == {"NORM"}:
                label = 0
            elif classes and "NORM" not in classes:
                label = 1
            else:
                continue
        else:
            if classes == {"NORM"}:
                label = 0
            elif "MI" in classes and "NORM" not in classes:
                if wanted_stages is not None and (
                    _infarction_stage(row.get("infarction_stadium1")) not in wanted_stages
                ):
                    continue
                label = 1
            else:
                continue
        item = dict(row)
        item["label"] = label
        item["diagnostic_classes"] = sorted(classes)
        selected.append(item)

    if max_records <= 0 or len(selected) <= max_records:
        return selected

    rng = np.random.default_rng(seed)
    picked: list[int] = []
    by_label_fold: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(selected):
        key = (int(row["label"]), int(float(row.get("strat_fold", 0))))
        by_label_fold.setdefault(key, []).append(index)
    quota = max_records // max(1, len(by_label_fold))
    for key in sorted(by_label_fold):
        candidates = by_label_fold[key]
        take = min(len(candidates), quota)
        if take:
            picked.extend(rng.choice(candidates, take, replace=False).tolist())
    remaining = max_records - len(picked)
    if remaining:
        pool = np.asarray(sorted(set(range(len(selected))) - set(picked)), dtype=int)
        picked.extend(rng.choice(pool, min(remaining, len(pool)), replace=False).tolist())
    return [selected[index] for index in sorted(picked)]


def _band_power(power: np.ndarray, freqs: np.ndarray, lower: float, upper: float) -> float:
    members = (freqs >= lower) & (freqs < upper)
    total = float(power.sum())
    return float(power[members].sum() / total) if total > 0 else 0.0


def _lead_features(lead: np.ndarray, sampling_rate: float) -> list[float]:
    lead = np.asarray(lead, dtype=float)
    lead = np.nan_to_num(lead, nan=float(np.nanmedian(lead)) if np.isfinite(lead).any() else 0.0)
    raw_mean = float(np.mean(lead))
    raw_std = float(np.std(lead))
    centered = lead - np.median(lead)
    scale = float(np.std(centered))
    shape = centered / (scale + 1e-8)
    derivative = np.diff(shape)
    second = np.diff(shape, n=2)
    signs = np.signbit(shape - np.mean(shape))
    zero_crossings = float(np.count_nonzero(signs[1:] != signs[:-1]))
    spectrum = np.abs(np.fft.rfft(shape)) ** 2
    freqs = np.fft.rfftfreq(len(shape), d=1.0 / sampling_rate)
    spectral = spectrum[1:] if len(spectrum) > 1 else spectrum
    spectral_freqs = freqs[1:] if len(freqs) > 1 else freqs
    spectral_total = float(spectral.sum())
    probabilities = spectral / (spectral_total + 1e-12)
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    dominant = float(spectral_freqs[int(np.argmax(spectral))]) if len(spectral) else 0.0
    return [
        raw_mean,
        raw_std,
        float(np.min(shape)),
        float(np.max(shape)),
        float(np.ptp(shape)),
        float(np.sqrt(np.mean(shape**2))),
        float(np.mean(shape**3)),
        float(np.mean(shape**4) - 3.0),
        float(np.std(derivative)),
        float(np.std(second)),
        zero_crossings,
        dominant,
        entropy,
        _band_power(spectrum, freqs, 0.5, 4.0),
        _band_power(spectrum, freqs, 4.0, 8.0),
        _band_power(spectrum, freqs, 8.0, 15.0),
        _band_power(spectrum, freqs, 15.0, 40.0),
        _band_power(spectrum, freqs, 40.0, min(100.0, sampling_rate / 2.0 + 1e-6)),
    ]


def _rhythm_features(signal: np.ndarray, sampling_rate: float) -> list[float]:
    lead = np.asarray(signal[:, 1], dtype=float)
    try:
        high = min(18.0, sampling_rate / 2.0 - 1.0)
        low = min(5.0, high - 1.0)
        b, a = butter(2, [low, high], btype="bandpass", fs=sampling_rate)
        filtered = filtfilt(b, a, lead)
        peaks, _ = find_peaks(
            filtered,
            distance=max(1, int(sampling_rate * 0.25)),
            prominence=max(float(np.std(filtered)) * 0.35, 1e-8),
        )
        rr = np.diff(peaks) / sampling_rate
        rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    except (ValueError, FloatingPointError):
        rr = np.asarray([], dtype=float)
    if len(rr) == 0:
        return [float("nan")] * 4
    return [
        float(np.mean(rr)),
        float(np.std(rr)),
        float(60.0 / np.mean(rr)),
        float(len(rr)),
    ]


def extract_ecg_features(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, list[str]]:
    """Extract device-agnostic morphology, spectrum, lead-correlation and rhythm features."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 2:
        raise ValueError("ECG signal must be a 2-D array")
    if values.shape[1] == len(LEADS):
        values = values.T
    if values.shape[0] != len(LEADS):
        raise ValueError(f"expected {len(LEADS)} ECG leads, got {values.shape}")

    features: list[float] = []
    names: list[str] = []
    for lead_name, lead in zip(LEADS, values, strict=True):
        features.extend(_lead_features(lead, sampling_rate))
        names.extend(f"{lead_name}_{name}" for name in PER_LEAD_FEATURES)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(values)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    for left in range(len(LEADS)):
        for right in range(left + 1, len(LEADS)):
            features.append(float(correlation[left, right]))
            names.append(f"corr_{LEADS[left]}_{LEADS[right]}")

    lead_std = np.std(values, axis=1)
    features.extend(
        [
            float(np.mean(lead_std)),
            float(np.std(lead_std)),
            float(np.min(lead_std)),
            float(np.max(lead_std)),
            *_rhythm_features(values.T, sampling_rate),
        ]
    )
    names.extend(
        [
            "lead_std_mean",
            "lead_std_std",
            "lead_std_min",
            "lead_std_max",
            "rr_mean_s",
            "rr_std_s",
            "heart_rate_bpm",
            "detected_rr_intervals",
        ]
    )
    return np.asarray(features, dtype=float), names


def align_raw_ecg_features(
    features: np.ndarray,
    feature_names: Iterable[str],
    expected_feature_names: Iterable[str],
    metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Align waveform features to an artifact, filling optional metadata.

    Training may include optional patient/device covariates such as age or
    sex, while a raw ECG export may contain only the waveform.  Missing
    optional values are represented as NaN so the artifact's training-fitted
    imputer supplies the documented median without changing the feature order.
    """

    values = np.asarray(features, dtype=float).reshape(-1)
    names = [str(name) for name in feature_names]
    expected = [str(name) for name in expected_feature_names]
    if len(values) != len(names):
        raise ValueError("raw ECG features and feature names must have equal length")
    provided = dict(zip(names, values, strict=True))
    optional = metadata or {}
    aligned: list[float] = []
    missing: list[str] = []
    for name in expected:
        if name in provided:
            aligned.append(float(provided[name]))
            continue
        if name in optional:
            aligned.append(_safe_float(str(optional[name])))
            continue
        aligned.append(float("nan"))
        missing.append(name)
    return np.asarray(aligned, dtype=float), missing


def _cache_paths(cache_path: Path) -> tuple[Path, Path]:
    return cache_path, cache_path.with_suffix(".json")


def load_ptbxl_ecg_dataset(
    metadata_csv: str | Path,
    statements_csv: str | Path,
    records_root: str | Path,
    target: str = "abnormal",
    max_records: int = 0,
    human_validated_only: bool = True,
    seed: int = 7,
    cache_path: str | Path | None = None,
    mi_stages: Sequence[str] | None = None,
) -> LoadedDataset:
    """Load local raw PTB-XL WFDB records into the platform's dataset contract."""

    metadata_path = Path(metadata_csv)
    statements_path = Path(statements_csv)
    records_base = Path(records_root)
    cache = Path(cache_path) if cache_path else None
    requested_stages = (
        sorted(_infarction_stage(stage) for stage in mi_stages) if mi_stages is not None else None
    )
    if cache and cache.exists() and cache.with_suffix(".json").exists():
        arrays = np.load(cache, allow_pickle=False)
        info = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
        cached_provenance = dict(info["provenance"])
        cached_stages = cached_provenance.get("mi_stages")
        if cached_stages != requested_stages:
            raise ValueError(
                f"cache {cache} was built with mi_stages={cached_stages!r}, "
                f"but mi_stages={requested_stages!r} was requested; "
                "use a different --cache path for each stage filter"
            )
        cached_schema = cached_provenance.get("feature_schema_version")
        if cached_schema != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"cache {cache} holds feature schema v{cached_schema}, but this build "
                f"extracts v{FEATURE_SCHEMA_VERSION}; delete the cache and re-extract "
                "so the change takes effect (v2 clamps PTB-XL's age=300 sentinel)"
            )
        return LoadedDataset(
            name=str(info["name"]),
            X=np.asarray(arrays["X"], dtype=float),
            y=np.asarray(arrays["y"], dtype=int),
            feature_names=list(info["feature_names"]),
            positive_label=str(info["positive_label"]),
            negative_label=str(info["negative_label"]),
            provenance=dict(info["provenance"]),
            groups=np.asarray(arrays["groups"], dtype=str),
            times=np.asarray(arrays["times"], dtype=str),
            row_ids=np.asarray(arrays["row_ids"], dtype=str),
            sites=np.asarray(arrays["sites"], dtype=str),
            subgroups={name: np.asarray(arrays[name], dtype=str) for name in info["subgroups"]},
            task_profile=dict(info["task_profile"]),
        )

    rows = select_ptbxl_rows(
        metadata_path,
        statements_path,
        target=target,
        max_records=max_records,
        human_validated_only=human_validated_only,
        mi_stages=mi_stages,
        seed=seed,
    )
    if not rows:
        raise ValueError("PTB-XL selection produced no records")
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("raw PTB-XL loading needs wfdb: pip install wfdb") from exc

    matrix: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    for index, row in enumerate(rows, start=1):
        record_path = records_base / str(row["filename_hr"])
        try:
            signal, fields = wfdb.rdsamp(str(record_path))
            extracted, names = extract_ecg_features(signal, float(fields["fs"]))
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise FileNotFoundError(
                f"missing or invalid raw ECG record {record_path}; download the selected PTB-XL records first"
            ) from exc
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("raw ECG records produced inconsistent feature names")
        matrix.append(extracted)
        kept.append(row)
        if index % 250 == 0 or index == len(rows):
            print(f"loaded ECG features {index}/{len(rows)}", flush=True)

    assert feature_names is not None
    X = np.asarray(matrix, dtype=float)
    y = np.asarray([int(row["label"]) for row in kept], dtype=int)
    groups = np.asarray([str(row["patient_id"]) for row in kept], dtype=str)
    folds = np.asarray([f"fold-{int(float(row['strat_fold'])):02d}" for row in kept], dtype=str)
    sites = np.asarray([str(row.get("site") or "unknown") for row in kept], dtype=str)
    row_ids = np.asarray([str(row["ecg_id"]) for row in kept], dtype=str)
    ages = np.asarray([ptbxl_age(row.get("age")) for row in kept])
    sex = np.asarray([str(row.get("sex") or "unknown") for row in kept], dtype=str)
    age_bucket = np.asarray(
        ["unknown" if not math.isfinite(age) else "<45" if age < 45 else "45-64" if age < 65 else "65+" for age in ages],
        dtype=str,
    )
    feature_matrix = np.column_stack(
        [
            X,
            np.nan_to_num(ages, nan=np.nanmedian(ages) if np.isfinite(ages).any() else 0.0),
            np.asarray([_safe_float(row.get("height")) for row in kept]),
            np.asarray([_safe_float(row.get("weight")) for row in kept]),
            (sex == "1").astype(float),
        ]
    )
    feature_matrix_names = feature_names + ["age", "height", "weight", "sex_male"]
    label_name = "clinically_abnormal_ecg" if target == "abnormal" else "mi_pattern_ecg"
    dataset = LoadedDataset(
        name=f"ptb-xl-{target}-raw-wfdb",
        X=feature_matrix,
        y=y,
        feature_names=feature_matrix_names,
        positive_label=label_name,
        negative_label="normal_ecg",
        provenance={
            "source": "PTB-XL 1.0.3 raw WFDB records",
            "metadata": str(metadata_path),
            "statements": str(statements_path),
            "waveform_root": str(records_base),
            "raw_leads": list(LEADS),
            "sampling_rate": "native PTB-XL 500 Hz records",
            "human_validated_only": human_validated_only,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "mi_stages": sorted(_infarction_stage(stage) for stage in mi_stages)
            if mi_stages is not None
            else None,
        },
        groups=groups,
        times=folds,
        row_ids=row_ids,
        sites=sites,
        subgroups={"age_group": age_bucket, "sex": sex},
        task_profile={
            "task_type": "binary_classification",
            "endpoint": label_name,
            "positive_definition": (
                "any PTB-XL diagnostic class other than NORM"
                if target == "abnormal"
                else "PTB-XL diagnostic class MI"
            ),
            "negative_definition": "PTB-XL diagnostic class NORM only",
            "unit_of_analysis": "one 10-second 12-lead ECG study",
            "raw_waveform": True,
        },
    )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            X=dataset.X,
            y=dataset.y,
            groups=dataset.groups,
            times=dataset.times,
            row_ids=dataset.row_ids,
            sites=dataset.sites,
            age_group=dataset.subgroups["age_group"],
            sex=dataset.subgroups["sex"],
        )
        cache.with_suffix(".json").write_text(
            json.dumps(
                {
                    "name": dataset.name,
                    "feature_names": dataset.feature_names,
                    "positive_label": dataset.positive_label,
                    "negative_label": dataset.negative_label,
                    "provenance": dataset.provenance,
                    "subgroups": sorted(dataset.subgroups),
                    "task_profile": dataset.task_profile,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return dataset


def download_ptbxl_records(
    rows: Iterable[dict[str, Any]],
    dataset_root: str | Path,
    base_url: str = "https://physionet.org/files/ptb-xl/1.0.3",
    workers: int = 8,
) -> dict[str, int]:
    """Download only selected raw HR WFDB records and preserve PTB-XL paths."""

    root = Path(dataset_root)
    jobs = []
    for row in rows:
        relative = str(row["filename_hr"])
        for extension in (".hea", ".dat"):
            destination = root / f"{relative}{extension}"
            if not destination.exists() or destination.stat().st_size == 0:
                jobs.append((relative + extension, destination))

    def fetch(job: tuple[str, Path]) -> tuple[bool, str]:
        relative, destination = job
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        url = f"{base_url.rstrip('/')}/{relative}"
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    temporary.write_bytes(response.read())
                temporary.replace(destination)
                return True, relative
            except Exception as exc:  # noqa: BLE001 - retry transient network errors
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    failed: list[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
            except Exception as exc:  # noqa: BLE001 - report every failed URL
                failed.append(str(exc))
            if (completed + len(failed)) % 100 == 0:
                print(f"downloaded {completed}/{len(jobs)} files", flush=True)
    if failed:
        raise RuntimeError(f"failed to download {len(failed)} PTB-XL files: {failed[:3]}")
    return {"requested_files": len(jobs), "downloaded_files": completed}
