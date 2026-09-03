"""Polygenic risk scores from PGS Catalog scoring files.

This is the genomics modality's entry point, and it is deliberately not a trained
model. A published polygenic score *is* the model: the PGS Catalog distributes per
variant effect weights that were fitted on cohorts far larger than anything this
platform will ever hold, and the correct thing to do with them is apply them, not
re-derive them. `PGS000018` (metaGRS_CAD, Inouye et al., J Am Coll Cardiol 2018,
1,745,180 variants) is the canonical cardiovascular example.

The scoring arithmetic is trivial -- a dot product of effect weights against effect
allele dosages. Everything difficult about a PRS is in the parts that decide *which*
weights are allowed to enter that dot product, and every guard below exists because
the alternative silently produces a confident, meaningless number:

* **Coverage is reported and enforced.** A score computed over 3% of its variants is
  not a weak score, it is a different score, and it is not comparable to the
  published distribution the risk strata were defined on. `apply_scoring_file`
  refuses below `min_coverage` rather than returning something plottable.

* **Allele orientation is resolved, not assumed.** A genotype source may report
  dosage with respect to either allele. When the reported allele pair matches the
  scoring file reversed, the dosage is flipped; when it matches neither, the variant
  is dropped. Adding an unflipped variant contributes the weight with the wrong sign,
  which is worse than omitting it.

* **Strand-ambiguous variants are flagged.** A/T and C/G SNPs have complementary
  allele pairs, so allele matching alone cannot distinguish "same strand" from
  "opposite strand". They are counted separately so a caller can see how much of the
  score rests on variants whose orientation was assumed rather than established.

* **Missing genotypes are not imputed to zero.** Absent dosage means the variant does
  not contribute and is not counted as covered; treating it as 0 would silently mean
  "homozygous reference", which is a claim about the sample that was never measured.

The raw sum is not interpretable on its own -- PRS units are arbitrary and depend on
the weight set. `standardise` converts to a z-score against a reference cohort, which
is the form the published risk strata actually use.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REQUIRED_COLUMNS = ("effect_allele", "effect_weight")

# A/T and C/G variants are their own complements, so an allele-pair match cannot
# establish strand. Tracked, not silently trusted.
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
_AMBIGUOUS_PAIRS = frozenset({frozenset({"A", "T"}), frozenset({"C", "G"})})

DEFAULT_MIN_COVERAGE = 0.80


@dataclass(frozen=True)
class ScoringFile:
    """A parsed PGS Catalog scoring file: metadata plus per-variant weights."""

    pgs_id: str
    name: str
    trait: str
    genome_build: str
    metadata: dict[str, str]
    variant_ids: list[str]
    effect_alleles: list[str]
    other_alleles: list[str]
    weights: np.ndarray

    def __len__(self) -> int:
        return len(self.variant_ids)

    @property
    def declared_variants(self) -> int | None:
        raw = self.metadata.get("variants_number")
        try:
            return int(raw) if raw is not None else None
        except ValueError:
            return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pgs_id": self.pgs_id,
            "name": self.name,
            "trait": self.trait,
            "genome_build": self.genome_build,
            "variants_parsed": len(self),
            "variants_declared": self.declared_variants,
            "citation": self.metadata.get("citation"),
            "weight_type": self.metadata.get("weight_type"),
        }


@dataclass(frozen=True)
class PRSResult:
    """Per-sample polygenic scores plus the coverage that justifies trusting them."""

    pgs_id: str
    sample_ids: list[str]
    raw_score: np.ndarray
    matched_variants: int
    scored_variants: int
    flipped_variants: int
    ambiguous_variants: int
    dropped_allele_mismatch: int
    per_sample_missing: np.ndarray
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of the scoring file's variants that entered the dot product."""

        return float(self.matched_variants / self.scored_variants) if self.scored_variants else 0.0

    def standardise(
        self,
        reference_mean: float | None = None,
        reference_std: float | None = None,
    ) -> np.ndarray:
        """Z-score the raw sum, which is the form published risk strata use.

        With no reference supplied the cohort's own mean and standard deviation are
        used. That is only meaningful for a cohort large and unselected enough to
        stand in for a population; on a small or case-enriched sample it re-centres
        the score on that sample's own risk, which is not what a published stratum
        boundary means. The caller is told via `warnings` when this happens.
        """

        values = np.asarray(self.raw_score, dtype=float)
        mean = float(np.mean(values)) if reference_mean is None else float(reference_mean)
        std = float(np.std(values)) if reference_std is None else float(reference_std)
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(
                "cannot standardise: reference standard deviation is zero or undefined "
                "(a single sample, or every sample scoring identically)"
            )
        return (values - mean) / std

    def as_dict(self) -> dict[str, Any]:
        return {
            "pgs_id": self.pgs_id,
            "samples": len(self.sample_ids),
            "raw_score_mean": float(np.mean(self.raw_score)) if len(self.raw_score) else None,
            "coverage": self.coverage,
            "matched_variants": self.matched_variants,
            "scoring_file_variants": self.scored_variants,
            "flipped_variants": self.flipped_variants,
            "strand_ambiguous_variants": self.ambiguous_variants,
            "dropped_allele_mismatch": self.dropped_allele_mismatch,
            "warnings": list(self.warnings),
        }


def _open_text(path: str | Path) -> io.TextIOBase:
    """Open a scoring file, transparently handling gzip (the Catalog's default)."""

    file_path = Path(path)
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8")  # type: ignore[return-value]
    return file_path.open("r", encoding="utf-8")


def _variant_key(rsid: str, chrom: str, position: str) -> str:
    """Prefer the rsID; fall back to chr:pos so build-matched files still align."""

    if rsid and rsid.lower() not in {"na", ".", ""}:
        return rsid
    if chrom and position:
        return f"{chrom}:{position}"
    return ""


def read_scoring_file(path: str | Path) -> ScoringFile:
    """Parse a PGS Catalog scoring file (format_version 2.0), gzipped or plain."""

    metadata: dict[str, str] = {}
    variant_ids: list[str] = []
    effect_alleles: list[str] = []
    other_alleles: list[str] = []
    weights: list[float] = []

    with _open_text(path) as handle:
        header: list[str] | None = None
        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("#"):
                # '#key=value' carries the metadata; '##...' and '###...' are prose.
                stripped = line.lstrip("#")
                if "=" in stripped and not line.startswith("##"):
                    key, _, value = stripped.partition("=")
                    metadata[key.strip()] = value.strip()
                continue
            fields = line.split("\t")
            if header is None:
                header = [name.strip() for name in fields]
                missing = [name for name in REQUIRED_COLUMNS if name not in header]
                if missing:
                    raise ValueError(
                        f"scoring file is missing required columns: {', '.join(missing)}"
                    )
                continue
            if len(fields) != len(header):
                # A ragged line means the file is truncated or not actually a
                # scoring file; scoring a partial variant set is exactly the silent
                # failure this module exists to prevent.
                raise ValueError(
                    f"scoring file row has {len(fields)} fields, header declares {len(header)}"
                )
            row = dict(zip(header, fields, strict=True))
            try:
                weight = float(row["effect_weight"])
            except (KeyError, ValueError):
                continue
            if not np.isfinite(weight):
                continue
            key = _variant_key(
                row.get("rsID", "").strip(),
                row.get("chr_name", "").strip(),
                row.get("chr_position", "").strip(),
            )
            if not key:
                continue
            variant_ids.append(key)
            effect_alleles.append(row.get("effect_allele", "").strip().upper())
            other_alleles.append(row.get("other_allele", "").strip().upper())
            weights.append(weight)

    if header is None:
        raise ValueError("scoring file contained no header row")
    if not variant_ids:
        raise ValueError("scoring file contained no usable variants")

    return ScoringFile(
        pgs_id=metadata.get("pgs_id", "unknown"),
        name=metadata.get("pgs_name", metadata.get("pgs_id", "unknown")),
        trait=metadata.get("trait_reported", "unknown"),
        genome_build=metadata.get("genome_build", "unknown"),
        metadata=metadata,
        variant_ids=variant_ids,
        effect_alleles=effect_alleles,
        other_alleles=other_alleles,
        weights=np.asarray(weights, dtype=float),
    )


def _orientation(
    effect_allele: str,
    other_allele: str,
    reported_effect: str,
    reported_other: str,
) -> tuple[int, bool]:
    """Resolve a genotype's allele pair against the scoring file's.

    Returns `(sign, ambiguous)` where sign is +1 (dosage used as given), -1 (dosage
    must be flipped to 2 - dosage), or 0 (alleles do not correspond; drop). A
    variant whose allele pair is its own complement is reported as ambiguous: the
    match succeeded, but strand was assumed rather than established.
    """

    effect_allele = effect_allele.upper()
    other_allele = other_allele.upper()
    reported_effect = reported_effect.upper()
    reported_other = reported_other.upper()

    ambiguous = (
        frozenset({effect_allele, other_allele}) in _AMBIGUOUS_PAIRS
        if effect_allele and other_allele
        else False
    )

    # Match on the allele PAIR, not on the effect allele alone. Matching one allele is
    # not enough: for a C/T variant every letter in {A,C,G,T} corresponds to something
    # on one strand or the other, so a single-allele test accepts a genuinely different
    # variant sitting at the same key and contributes its weight with an arbitrary sign.
    expected = {effect_allele, other_allele} - {""}
    reported = {reported_effect, reported_other} - {""}

    if expected and reported == expected:
        return (1, ambiguous) if reported_effect == effect_allele else (-1, ambiguous)

    complemented_effect = _COMPLEMENT.get(reported_effect, "")
    complemented_other = _COMPLEMENT.get(reported_other, "")
    complemented = {complemented_effect, complemented_other} - {""}
    if expected and complemented == expected:
        return (1, ambiguous) if complemented_effect == effect_allele else (-1, ambiguous)

    # Some scoring files omit `other_allele`. With only one allele to go on, orientation
    # can be established against the effect allele alone -- on either strand -- but the
    # result is necessarily assumed rather than verified, so it is reported as ambiguous.
    if not other_allele or not reported_other:
        if reported_effect == effect_allele or complemented_effect == effect_allele:
            return 1, True
        if effect_allele and (reported_effect or complemented_effect):
            return -1, True

    return 0, ambiguous


def apply_scoring_file(
    scoring: ScoringFile,
    dosages: Mapping[str, Sequence[float | None]],
    sample_ids: Sequence[str],
    reported_alleles: Mapping[str, tuple[str, str]] | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> PRSResult:
    """Score samples against a PGS scoring file.

    `dosages` maps a variant key (rsID, or 'chr:pos') to that variant's effect-allele
    dosage per sample, in `sample_ids` order. `None` means not measured for that
    sample and contributes nothing. `reported_alleles` optionally gives the
    (effect, other) allele pair the dosages are expressed against, enabling
    orientation resolution; without it the dosages are taken at face value and every
    matched variant is counted as assumed-orientation.
    """

    if not sample_ids:
        raise ValueError("no samples to score")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")

    n_samples = len(sample_ids)
    totals = np.zeros(n_samples, dtype=float)
    missing_counts = np.zeros(n_samples, dtype=int)
    matched = 0
    flipped = 0
    ambiguous_total = 0
    dropped = 0
    warnings: list[str] = []

    for index, key in enumerate(scoring.variant_ids):
        sample_dosages = dosages.get(key)
        if sample_dosages is None:
            continue
        if len(sample_dosages) != n_samples:
            raise ValueError(
                f"variant {key} has {len(sample_dosages)} dosages for {n_samples} samples"
            )

        sign = 1
        ambiguous = False
        if reported_alleles is not None:
            pair = reported_alleles.get(key)
            if pair is None:
                dropped += 1
                continue
            sign, ambiguous = _orientation(
                scoring.effect_alleles[index],
                scoring.other_alleles[index],
                pair[0],
                pair[1],
            )
            if sign == 0:
                dropped += 1
                continue
        else:
            ambiguous = True

        matched += 1
        if sign < 0:
            flipped += 1
        if ambiguous:
            ambiguous_total += 1

        weight = float(scoring.weights[index])
        for sample_index, dosage in enumerate(sample_dosages):
            if dosage is None or not np.isfinite(float(dosage)):
                missing_counts[sample_index] += 1
                continue
            value = float(dosage)
            if sign < 0:
                value = 2.0 - value
            totals[sample_index] += weight * value

    scored = len(scoring)
    coverage = matched / scored if scored else 0.0
    if coverage < min_coverage:
        raise ValueError(
            f"polygenic score covers {coverage:.1%} of {scoring.pgs_id}'s "
            f"{scored} variants ({matched} matched), below the {min_coverage:.0%} floor. "
            "A score over a small fraction of its variants is not a weaker version of "
            "the published score, it is a different one, and published risk strata do "
            "not apply to it. Supply more genotypes or lower min_coverage deliberately."
        )

    if reported_alleles is None:
        warnings.append(
            "no reported alleles supplied: dosages were taken at face value, so "
            "effect-allele orientation is assumed rather than verified"
        )
    if ambiguous_total:
        warnings.append(
            f"{ambiguous_total} of {matched} scored variants are strand-ambiguous "
            "(A/T or C/G); their orientation could not be established by allele matching"
        )
    if int(missing_counts.max(initial=0)) > 0:
        warnings.append(
            f"up to {int(missing_counts.max())} variants were unmeasured in some samples "
            "and contributed nothing; per-sample totals are not equally covered"
        )

    return PRSResult(
        pgs_id=scoring.pgs_id,
        sample_ids=[str(item) for item in sample_ids],
        raw_score=totals,
        matched_variants=matched,
        scored_variants=scored,
        flipped_variants=flipped,
        ambiguous_variants=ambiguous_total,
        dropped_allele_mismatch=dropped,
        per_sample_missing=missing_counts,
        warnings=warnings,
    )


def prs_feature_matrix(result: PRSResult, standardised: bool = True) -> tuple[np.ndarray, list[str]]:
    """Shape a PRS as a one-column feature matrix for the modality frame.

    Genomic risk enters the platform as exactly one number per patient. That is not a
    limitation of this implementation -- it is what a polygenic score is, and packing
    it beside clinical features lets the combiner weigh it against them rather than
    letting a million variants overwhelm a handful of measurements.
    """

    if standardised:
        values = result.standardise()
        name = f"{result.pgs_id}_z"
    else:
        values = np.asarray(result.raw_score, dtype=float)
        name = f"{result.pgs_id}_raw"
    return values.reshape(-1, 1), [name]


def read_dosage_table(
    path: str | Path,
    sample_id_column: str = "sample_id",
) -> tuple[list[str], dict[str, list[float | None]], dict[str, tuple[str, str]]]:
    """Read a simple wide dosage CSV: one row per sample, one column per variant.

    Columns may be named `rsID` or `rsID:EFFECT_OTHER` -- the latter declares the
    allele pair the dosage is expressed against, which is what makes orientation
    resolution possible. Blank cells are unmeasured, not zero.
    """

    import csv

    file_path = Path(path)
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("dosage CSV must have a header row")
        if sample_id_column not in reader.fieldnames:
            raise ValueError(f"dosage CSV has no '{sample_id_column}' column")
        variant_columns = [name for name in reader.fieldnames if name != sample_id_column]
        rows = list(reader)

    if not rows:
        raise ValueError("dosage CSV contained no samples")

    sample_ids = [str(row[sample_id_column]).strip() for row in rows]
    dosages: dict[str, list[float | None]] = {}
    alleles: dict[str, tuple[str, str]] = {}

    for column in variant_columns:
        key, _, allele_spec = column.partition(":")
        key = key.strip()
        if allele_spec and "_" in allele_spec:
            effect, _, other = allele_spec.partition("_")
            alleles[key] = (effect.strip().upper(), other.strip().upper())
        values: list[float | None] = []
        for row in rows:
            raw = str(row.get(column, "")).strip()
            if not raw or raw.upper() in {"NA", "NAN", "."}:
                values.append(None)
                continue
            try:
                values.append(float(raw))
            except ValueError:
                values.append(None)
        dosages[key] = values

    return sample_ids, dosages, alleles


def describe_available_scores(trait_keyword: str = "coronary") -> list[dict[str, str]]:
    """Well-known PGS Catalog scores for cardiovascular risk, for discoverability.

    A static list rather than a live query: this is documentation of what is known to
    be openly downloadable, and it must not make the frame's readiness report depend
    on network availability.
    """

    catalog = [
        {
            "pgs_id": "PGS000018",
            "name": "metaGRS_CAD",
            "trait": "Coronary artery disease",
            "variants": "1745180",
            "citation": "Inouye M et al. J Am Coll Cardiol (2018)",
        },
        {
            "pgs_id": "PGS000011",
            "name": "GRS46K",
            "trait": "Coronary artery disease",
            "variants": "46000",
            "citation": "Khera AV et al. Nat Genet (2018)",
        },
        {
            "pgs_id": "PGS000337",
            "name": "GPS_CAD",
            "trait": "Coronary artery disease",
            "variants": "6630150",
            "citation": "Khera AV et al. Nat Genet (2018)",
        },
        {
            "pgs_id": "PGS003725",
            "name": "GPS_Mult_CAD",
            "trait": "Coronary artery disease",
            "variants": "1296396",
            "citation": "Patel AP et al. Nat Med (2023)",
        },
    ]
    keyword = trait_keyword.lower()
    return [entry for entry in catalog if keyword in entry["trait"].lower()]


def scoring_file_url(pgs_id: str) -> str:
    """Canonical open download URL for a PGS Catalog scoring file (no auth required)."""

    identifier = str(pgs_id).strip().upper()
    if not identifier.startswith("PGS"):
        raise ValueError("pgs_id must look like 'PGS000018'")
    return (
        "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/"
        f"{identifier}/ScoringFiles/{identifier}.txt.gz"
    )
