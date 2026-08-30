import json

import pytest

from src.report_sosa_ssn_feedback_framing import (
    FRAMINGS,
    build_manifest,
    pairwise_rows,
    primary_rows,
    secondary_rows,
    validate_analysis,
    write_tables,
)


def interval(count, n=30):
    return {
        "count": count,
        "n": n,
        "rate": count / n,
        "lower_95": 0.0,
        "upper_95": 1.0,
    }


def analysis_payload():
    secondary = {}
    for index, name in enumerate(FRAMINGS):
        usable = 30 - index
        secondary[name] = {
            "n": 30,
            "usable_outputs": interval(usable),
            "controlled_target_removed": interval(index * 5),
            "owl_consistent": interval(index * 5),
            "exact_reference_recovery": interval(index),
            "output_failure": interval(index),
            "among_usable_outputs": {
                "collateral_edit": interval(10, usable),
                "new_raw_shacl_findings": interval(5, usable),
                "new_grounding_findings": interval(8, usable),
                "owl_inconsistent_after_target_removal": interval(0, usable),
            },
            "edit_distance_from_injected": {"mean": 1.0, "median": 1.0},
            "edit_distance_from_clean_reference": {"mean": 2.0, "median": 2.0},
        }
    pairwise = []
    for left, right in (
        ("explanation", "verdict"),
        ("explanation", "location"),
        ("location", "verdict"),
    ):
        pairwise.append({
            "left": left,
            "right": right,
            "n": 30,
            "both": 0,
            "left_only": 5,
            "right_only": 0,
            "neither": 25,
            "risk_difference_left_minus_right": 1 / 6,
            "risk_difference_lower_95": 0.0,
            "risk_difference_upper_95": 0.3,
            "p_value_raw": 0.0625,
            "p_value_holm": 0.1875,
            "reject_at_alpha_0_05": False,
        })
    return {
        "version": 1,
        "integrity": {
            "observations": 90,
            "paired_cases": 30,
            "observations_per_framing": {name: 30 for name in FRAMINGS},
            "complete_paired_matrix": True,
        },
        "primary_outcome": {
            "name": "controlled_target_removed",
            "by_framing": {name: secondary[name]["controlled_target_removed"] for name in FRAMINGS},
            "omnibus": {
                "test": "Cochran Q",
                "statistic": 10.0,
                "degrees_of_freedom": 2,
                "p_value": 0.0067,
            },
            "pairwise": pairwise,
            "multiplicity_correction": "Holm correction",
        },
        "secondary_outcomes": secondary,
        "cost": {
            "repair_generation": {
                "calls": 90,
                "prompt_tokens": 1000,
                "generated_tokens": 500,
                "recorded_duration_seconds": 20.0,
            },
            "live_grounding": {
                "unique_calls_within_case": 10,
                "prompt_tokens": 100,
                "generated_tokens": 50,
                "recorded_duration_seconds": 5.0,
            },
            "wall_seconds": 30.0,
        },
        "execution": {"models_or_validators_run": False},
    }


def test_validate_analysis_accepts_the_complete_paired_design():
    validate_analysis(analysis_payload())


def test_validate_analysis_rejects_a_missing_observation():
    payload = analysis_payload()
    payload["integrity"]["observations"] = 89
    with pytest.raises(RuntimeError, match="30 cases and 90 observations"):
        validate_analysis(payload)


def test_primary_and_secondary_rows_preserve_their_denominators():
    payload = analysis_payload()
    primary = primary_rows(payload)
    secondary = secondary_rows(payload)
    assert [row["n"] for row in primary] == [30, 30, 30]
    assert [row["usable_n"] for row in secondary] == [30, 29, 28]
    assert secondary[-1]["collateral_count"] == 10


def test_pairwise_rows_keep_discordant_counts_and_adjusted_p_values():
    rows = pairwise_rows(analysis_payload())
    assert len(rows) == 3
    assert rows[0]["comparison"] == "explanation_vs_verdict"
    assert rows[0]["left_only"] == 5
    assert rows[0]["right_only"] == 0
    assert rows[0]["p_value_holm"] == 0.1875


def test_tables_and_manifest_are_hash_bound(tmp_path):
    payload = analysis_payload()
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "report"
    tables = write_tables(payload, output_dir)
    assert len(tables) == 5
    assert {path.name for path in output_dir.iterdir()} == {
        "primary_outcomes_by_framing.csv",
        "secondary_outcomes_by_framing.csv",
        "paired_primary_comparisons.csv",
        "primary_omnibus_test.csv",
        "cost_summary.csv",
    }
    manifest = build_manifest(analysis_path, output_dir)
    assert manifest["models_or_validators_run"] is False
    assert set(manifest["outputs"]) == set(tables)
