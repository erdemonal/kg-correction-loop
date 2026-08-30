import json

import pytest

from src.report_sosa_ssn_confirmatory import (
    CONDITIONS,
    build_manifest,
    cost_rows,
    overall_rows,
    repair_rows,
    validate_analysis,
    validator_rows,
    write_tables,
)


def interval(count, n=30):
    rate = count / n
    return {"count": count, "n": n, "rate": rate, "lower_95": rate, "upper_95": rate}


def analysis_payload():
    coverage = {}
    grounding = {}
    repair = {}
    for index, condition in enumerate(CONDITIONS):
        coverage[condition] = {
            "n": 30,
            "raw_shacl": interval(30),
            "owl_consistency": interval(10),
            "grounding_v3": interval(index),
        }
        grounding[condition] = {"n": 30}
        repair[condition] = {
            "n": 30,
            "ever_target_resolution": interval(29),
            "end_to_end_target_resolution": interval(20),
            "validated_state": interval(18),
            "end_to_end_reference_recovery": interval(17),
            "output_failure": interval(6),
            "any_collateral_edit": interval(12),
            "any_new_violation": interval(14),
            "paired_f1_changes": {"improved": 15, "unchanged": 5, "worsened": 10},
            "mean_initial_f1": 0.9,
            "mean_last_validated_f1": 0.92,
            "mean_paired_f1_change": 0.02,
            "mean_repair_rounds": 2.0,
        }
    return {
        "version": 1,
        "integrity": {
            "cases": 180,
            "unique_case_ids": 180,
            "cases_per_condition": {name: 30 for name in CONDITIONS},
        },
        "grounding": {
            "overall": {"target_matches_expected": interval(150, 180)},
            "by_condition": grounding,
            "initial_cost": {
                "calls": 100,
                "prompt_tokens": 1000,
                "generated_tokens": 200,
                "duration_seconds": 10.0,
                "wall_seconds": 12.0,
            },
        },
        "validator_coverage_at_round_zero": {
            "overall": {
                "raw_shacl": interval(150, 180),
                "owl_consistency": interval(60, 180),
                "grounding_v3": interval(120, 180),
            },
            "by_condition": coverage,
        },
        "repair": {
            "overall": {
                "ever_target_resolution": interval(166, 180),
                "end_to_end_target_resolution": interval(117, 180),
                "validated_state": interval(87, 180),
                "end_to_end_reference_recovery": interval(81, 180),
                "output_failure": interval(63, 180),
                "any_collateral_edit": interval(99, 180),
                "any_new_violation": interval(111, 180),
            },
            "by_condition": repair,
            "cost": {
                "repair_calls": 429,
                "repair_prompt_tokens": 1000,
                "repair_generated_tokens": 500,
                "repair_duration_seconds": 50.0,
                "wall_seconds": 60.0,
                "live_grounding_calls": 252,
                "live_grounding_prompt_tokens": 800,
                "live_grounding_generated_tokens": 200,
                "live_grounding_duration_seconds": 20.0,
            },
        },
    }


def test_validate_analysis_accepts_locked_sample():
    validate_analysis(analysis_payload())


def test_validate_analysis_rejects_sample_drift():
    payload = analysis_payload()
    payload["integrity"]["unique_case_ids"] = 179
    with pytest.raises(RuntimeError, match="180 unique"):
        validate_analysis(payload)


def test_validator_rows_keep_validator_counts_separate():
    rows = validator_rows(analysis_payload())
    assert len(rows) == 6
    assert rows[0]["shacl_count"] == 30
    assert rows[0]["owl_count"] == 10
    assert rows[-1]["grounding_count"] == 5


def test_repair_rows_keep_ever_final_validated_and_exact_separate():
    row = repair_rows(analysis_payload())[0]
    outcomes = (
        row["ever_count"],
        row["final_count"],
        row["validated_count"],
        row["exact_count"],
    )
    assert outcomes == (29, 20, 18, 17)
    assert row["f1_improved"] + row["f1_unchanged"] + row["f1_worsened"] == 30


def test_overall_and_cost_rows_preserve_denominators():
    payload = analysis_payload()
    overall = {row["metric"]: row for row in overall_rows(payload)}
    assert overall["ever_target_resolution"]["n"] == 180
    assert overall["ever_target_resolution"]["count"] == 166
    costs = cost_rows(payload)
    assert [row["model_calls"] for row in costs] == [100, 429, 252]


def test_tables_and_manifest_are_hash_bound(tmp_path):
    payload = analysis_payload()
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "report"
    validators, repairs = write_tables(payload, output_dir)
    assert len(validators) == len(repairs) == 6
    assert {path.name for path in output_dir.iterdir()} == {
        "overall_summary.csv",
        "validator_coverage_by_condition.csv",
        "repair_outcomes_by_condition.csv",
        "cost_summary.csv",
    }
    manifest = build_manifest(analysis_path, output_dir)
    assert manifest["models_or_validators_run"] is False
    assert set(manifest["outputs"]) == {path.name for path in output_dir.iterdir()}
