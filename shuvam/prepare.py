#!/usr/bin/env python
"""Build a genuine early-detection cohort from the MUSIC heart-failure study.

Unlike every other cohort in this platform, MUSIC is *prospective*: 992 ambulatory
chronic-heart-failure patients had clinical, ECG, echocardiographic and laboratory
measurements taken at enrolment (2003-2004) and were then followed for a median of
44 months. The outcome therefore genuinely postdates the features, which is what
"early detection" requires and what the platform's existing cohorts cannot supply.

Endpoint: the study's own primary outcome, cardiac death — sudden cardiac death
(cause 3) or pump-failure death (cause 6) — occurring within a fixed horizon of
enrolment.

Labelling uses a landmark rule rather than treating everyone as a valid negative:

  positive  cardiac death with follow-up <= horizon
  negative  known to be alive at the horizon (follow-up > horizon)
  excluded  censored before the horizon without the event — lost to follow-up,
            transplanted, or the study simply ended for them. Their status at the
            horizon is unknown, and calling them negative would inject exactly the
            label noise that makes a prognostic model look better than it is.

Non-cardiac deaths (cause 1) and other deaths (cause 7) are competing risks. They
are excluded when they occur before the horizon: such a patient neither had the
endpoint nor was observed event-free through it.

Source: https://physionet.org/content/music-sudden-cardiac-death/1.0.1/
License: Open Data Commons Open Database License (ODbL) v1.0 — attribution and
share-alike apply to any redistributed derivative.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

SOURCE_URL = "https://physionet.org/files/music-sudden-cardiac-death/1.0.1/subject-info.csv"

CARDIAC_DEATH_CAUSES = {"3", "6"}  # sudden cardiac death, pump-failure death
OTHER_DEATH_CAUSES = {"1", "7"}  # non-cardiac, other — competing risks
EXIT_DEATH = "3"

# Enrolment dates are not published (only follow-up duration in days), so a nominal
# epoch stands in for the index time. This preserves what matters — the outcome is
# recorded a measured number of days *after* the features — while making clear that
# cross-patient chronological ordering is not recoverable from this release.
NOMINAL_ENROLMENT = date(2003, 4, 1)

# Columns that leak the outcome and must never become features.
LEAKAGE_COLUMNS = (
    "Follow-up period from enrollment (days)",
    "days_4years",
    "Exit of the study",
    "Cause of death",
)

# The exporter wrote this column without a decimal point, using commas as thousands
# separators, so its magnitude is not recoverable from the string: "4,970,833,333"
# is 49.708 (the patient's 24h count is 1193) while "7,946,666,667" is 794.67
# (24h count 19072). 236 rows are affected. Nothing is lost by dropping it — VPC per
# hour is just the clean "…in 24h" column divided by 24 — and guessing the decimal
# position would inject noise into exactly the ventricular-ectopy signal that
# predicts the sudden-cardiac-death endpoint.
UNRECOVERABLE_COLUMNS = ("Number of ventricular premature contractions per hour",)

# A clock time of day. As a string it one-hot expands into ~590 dummy columns on an
# 888-row cohort; as hour-of-day it stays one numeric feature and keeps whatever
# circadian information it carries.
CLOCK_TIME_COLUMNS = ("Holter onset (hh:mm:ss)",)

# MUSIC de-identifies the oldest patient as ">89", exactly the pattern PTB-XL uses
# with age=300. Left as a string it turns the whole Age column categorical, and age
# is among the strongest prognostic variables in heart failure.
AGE_CEILING = 89.0


def _clean(value: str) -> str:
    """Normalise one MUSIC cell: comma decimals, NA markers, censored ages."""

    text = (value or "").strip().strip("'")
    if not text or text.upper() == "NA":
        return ""
    if text.startswith(">"):
        ceiling = text[1:].strip()
        return ceiling if ceiling else ""
    # A single comma is the decimal separator ("112,5" -> 112.5).
    return text.replace(",", ".", 1) if text.count(",") == 1 else text.replace(",", "")


def _hour_of_day(value: str) -> str:
    """Turn "'11:38:43'" into 11.645 hours; blank when absent or malformed."""

    text = (value or "").strip().strip("'")
    if not text or text.upper() == "NA":
        return ""
    parts = text.split(":")
    if len(parts) != 3 or not all(part.strip().isdigit() for part in parts):
        return ""
    hours, minutes, seconds = (int(part) for part in parts)
    return f"{hours + minutes / 60.0 + seconds / 3600.0:.4f}"


def build(source: Path, destination: Path, horizon_days: int) -> dict[str, int]:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    if not rows:
        raise ValueError(f"no rows read from {source}")

    feature_columns = [
        name
        for name in rows[0]
        if name
        and name != "Patient ID"
        and name not in LEAKAGE_COLUMNS
        and name not in UNRECOVERABLE_COLUMNS
    ]

    counts = {"positive": 0, "negative": 0, "excluded_censored": 0, "excluded_competing": 0}
    written: list[dict[str, str]] = []

    for row in rows:
        follow_up_text = _clean(row.get("Follow-up period from enrollment (days)", ""))
        if not follow_up_text:
            counts["excluded_censored"] += 1
            continue
        follow_up = float(follow_up_text)
        exit_code = (row.get("Exit of the study") or "").strip()
        cause = (row.get("Cause of death") or "").strip()

        died_cardiac = exit_code == EXIT_DEATH and cause in CARDIAC_DEATH_CAUSES
        died_other = exit_code == EXIT_DEATH and cause in OTHER_DEATH_CAUSES

        if died_cardiac and follow_up <= horizon_days:
            label = 1
            counts["positive"] += 1
        elif follow_up > horizon_days:
            # Observed event-free through the whole horizon, whatever happened later.
            label = 0
            counts["negative"] += 1
        elif died_other:
            counts["excluded_competing"] += 1
            continue
        else:
            # Transplanted, lost, or follow-up ended early: status at horizon unknown.
            counts["excluded_censored"] += 1
            continue

        record = {
            "patient_id": (row.get("Patient ID") or "").strip(),
            "enrolment_date": NOMINAL_ENROLMENT.isoformat(),
            "outcome_date": (NOMINAL_ENROLMENT + timedelta(days=int(follow_up))).isoformat(),
            "cardiac_death_within_horizon": str(label),
        }
        for name in feature_columns:
            raw = row.get(name, "")
            if name == "Age":
                cleaned = _clean(raw)
                record[name] = (
                    f"{min(float(cleaned), AGE_CEILING):g}" if cleaned else ""
                )
            elif name in CLOCK_TIME_COLUMNS:
                record[name] = _hour_of_day(raw)
            else:
                record[name] = _clean(raw)
        written.append(record)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "enrolment_date",
        "outcome_date",
        "cardiac_death_within_horizon",
        *feature_columns,
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written)

    counts["rows_written"] = len(written)
    counts["feature_columns"] = len(feature_columns)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="MUSIC subject-info.csv")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent / "music_cardiac_death.csv",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=1461,
        help="prediction horizon; 1461 = 4 years, matching the study's own landmark",
    )
    args = parser.parse_args()

    counts = build(args.source, args.destination, args.horizon_days)
    print(f"horizon: {args.horizon_days} days ({args.horizon_days / 365.25:.1f} years)")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    print(f"wrote {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
