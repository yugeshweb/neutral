"""Tests for the polygenic score engine.

The arithmetic is a dot product and is not what these tests are about. Every test
below targets a guard whose absence would yield a confident, wrong number: allele
orientation, coverage enforcement, unmeasured genotypes, and strand ambiguity.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from qhealth_qml.prs import (
    apply_scoring_file,
    prs_feature_matrix,
    read_dosage_table,
    read_scoring_file,
    scoring_file_url,
)

SCORING_TEXT = """###PGS CATALOG SCORING FILE
#format_version=2.0
##POLYGENIC SCORE (PGS) INFORMATION
#pgs_id=PGS999999
#pgs_name=test_cad
#trait_reported=Coronary artery disease
#genome_build=hg19
#variants_number=4
#citation=Test et al. (2026)
rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight
rs001\t1\t1000\tA\tG\t0.5
rs002\t1\t2000\tC\tT\t-0.25
rs003\t2\t3000\tG\tC\t1.0
rs004\t2\t4000\tT\tA\t2.0
"""


@pytest.fixture()
def scoring_path(tmp_path: Path) -> Path:
    path = tmp_path / "PGS999999.txt"
    path.write_text(SCORING_TEXT, encoding="utf-8")
    return path


def test_reads_metadata_and_weights(scoring_path: Path) -> None:
    scoring = read_scoring_file(scoring_path)
    assert scoring.pgs_id == "PGS999999"
    assert scoring.trait == "Coronary artery disease"
    assert scoring.genome_build == "hg19"
    assert scoring.declared_variants == 4
    assert len(scoring) == 4
    assert scoring.weights.tolist() == [0.5, -0.25, 1.0, 2.0]


def test_reads_gzipped_scoring_file(tmp_path: Path) -> None:
    path = tmp_path / "PGS999999.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(SCORING_TEXT)
    assert len(read_scoring_file(path)) == 4


def test_ragged_row_is_rejected_rather_than_partially_scored(tmp_path: Path) -> None:
    path = tmp_path / "broken.txt"
    path.write_text(SCORING_TEXT + "rs005\t3\t5000\tA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        read_scoring_file(path)


def test_dosages_scored_against_declared_alleles(scoring_path: Path) -> None:
    scoring = read_scoring_file(scoring_path)
    dosages = {"rs001": [2.0], "rs002": [1.0], "rs003": [0.0], "rs004": [1.0]}
    alleles = {
        "rs001": ("A", "G"),
        "rs002": ("C", "T"),
        "rs003": ("G", "C"),
        "rs004": ("T", "A"),
    }
    result = apply_scoring_file(scoring, dosages, ["s1"], reported_alleles=alleles)
    # 0.5*2 + -0.25*1 + 1.0*0 + 2.0*1
    assert result.raw_score[0] == pytest.approx(1.0 - 0.25 + 0.0 + 2.0)
    assert result.coverage == 1.0


def test_reversed_alleles_flip_the_dosage_rather_than_adding_the_wrong_sign(
    scoring_path: Path,
) -> None:
    """A genotype reported against the other allele must be flipped, not taken as-is."""

    scoring = read_scoring_file(scoring_path)
    # rs001 reported as G/A: dosage 2 of G means 0 of the effect allele A.
    result = apply_scoring_file(
        scoring,
        {"rs001": [2.0]},
        ["s1"],
        reported_alleles={"rs001": ("G", "A")},
        min_coverage=0.0,
    )
    assert result.flipped_variants == 1
    assert result.raw_score[0] == pytest.approx(0.0)

    unflipped = apply_scoring_file(
        scoring,
        {"rs001": [2.0]},
        ["s1"],
        reported_alleles={"rs001": ("A", "G")},
        min_coverage=0.0,
    )
    assert unflipped.raw_score[0] == pytest.approx(1.0)


def test_allele_mismatch_is_dropped_not_scored(scoring_path: Path) -> None:
    scoring = read_scoring_file(scoring_path)
    result = apply_scoring_file(
        scoring,
        {"rs001": [2.0], "rs002": [2.0]},
        ["s1"],
        # rs002 is C/T in the scoring file. A/C is neither that pair nor its
        # complement (T/G), so it is a different variant at the same key.
        reported_alleles={"rs001": ("A", "G"), "rs002": ("A", "C")},
        min_coverage=0.0,
    )
    assert result.dropped_allele_mismatch == 1
    assert result.matched_variants == 1
    assert result.raw_score[0] == pytest.approx(1.0)


def test_strand_ambiguous_variants_are_flagged(scoring_path: Path) -> None:
    """rs003 (G/C) and rs004 (T/A) are their own complements and must be reported."""

    scoring = read_scoring_file(scoring_path)
    result = apply_scoring_file(
        scoring,
        {"rs003": [1.0], "rs004": [1.0]},
        ["s1"],
        reported_alleles={"rs003": ("G", "C"), "rs004": ("T", "A")},
        min_coverage=0.0,
    )
    assert result.ambiguous_variants == 2
    assert any("strand-ambiguous" in warning for warning in result.warnings)


def test_low_coverage_refuses_rather_than_returning_a_number(scoring_path: Path) -> None:
    """A score over a fraction of its variants is a different score, not a weak one."""

    scoring = read_scoring_file(scoring_path)
    with pytest.raises(ValueError, match="below the"):
        apply_scoring_file(
            scoring,
            {"rs001": [2.0]},
            ["s1"],
            reported_alleles={"rs001": ("A", "G")},
            min_coverage=0.8,
        )


def test_missing_genotype_contributes_nothing_and_is_counted(scoring_path: Path) -> None:
    """An unmeasured variant must not be silently read as homozygous reference."""

    scoring = read_scoring_file(scoring_path)
    result = apply_scoring_file(
        scoring,
        {"rs001": [None], "rs004": [1.0]},
        ["s1"],
        reported_alleles={"rs001": ("A", "G"), "rs004": ("T", "A")},
        min_coverage=0.0,
    )
    assert result.per_sample_missing[0] == 1
    assert result.raw_score[0] == pytest.approx(2.0)
    assert any("unmeasured" in warning for warning in result.warnings)


def test_absent_reported_alleles_are_flagged_as_assumed(scoring_path: Path) -> None:
    scoring = read_scoring_file(scoring_path)
    result = apply_scoring_file(
        scoring,
        {"rs001": [2.0], "rs002": [0.0], "rs003": [0.0], "rs004": [0.0]},
        ["s1"],
    )
    assert any("orientation is assumed" in warning for warning in result.warnings)


def test_dosage_count_must_match_sample_count(scoring_path: Path) -> None:
    scoring = read_scoring_file(scoring_path)
    with pytest.raises(ValueError, match="dosages for"):
        apply_scoring_file(scoring, {"rs001": [1.0]}, ["s1", "s2"], min_coverage=0.0)


def test_standardise_uses_reference_when_supplied(scoring_path: Path) -> None:
    scoring = read_scoring_file(scoring_path)
    result = apply_scoring_file(
        scoring,
        {"rs001": [0.0, 2.0, 1.0, 1.0]},
        ["a", "b", "c", "d"],
        min_coverage=0.0,
    )
    z = result.standardise(reference_mean=0.5, reference_std=0.5)
    assert z.tolist() == pytest.approx([-1.0, 1.0, 0.0, 0.0])


def test_standardise_refuses_zero_variance(scoring_path: Path) -> None:
    scoring = read_scoring_file(scoring_path)
    result = apply_scoring_file(scoring, {"rs001": [1.0, 1.0]}, ["a", "b"], min_coverage=0.0)
    with pytest.raises(ValueError, match="standard deviation"):
        result.standardise()


def test_feature_matrix_is_one_column_per_patient(scoring_path: Path) -> None:
    scoring = read_scoring_file(scoring_path)
    result = apply_scoring_file(
        scoring,
        {"rs001": [0.0, 2.0, 1.0]},
        ["a", "b", "c"],
        min_coverage=0.0,
    )
    X, names = prs_feature_matrix(result)
    assert X.shape == (3, 1)
    assert names == ["PGS999999_z"]


def test_dosage_table_parses_alleles_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "dosages.csv"
    path.write_text(
        "sample_id,rs001:A_G,rs002:C_T\n" "s1,2,1\n" "s2,,0\n",
        encoding="utf-8",
    )
    sample_ids, dosages, alleles = read_dosage_table(path)
    assert sample_ids == ["s1", "s2"]
    assert alleles["rs001"] == ("A", "G")
    assert dosages["rs001"] == [2.0, None]
    assert dosages["rs002"] == [1.0, 0.0]


def test_scoring_file_url_is_the_open_ftp_path() -> None:
    assert scoring_file_url("pgs000018") == (
        "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/"
        "PGS000018/ScoringFiles/PGS000018.txt.gz"
    )
    with pytest.raises(ValueError):
        scoring_file_url("18")
