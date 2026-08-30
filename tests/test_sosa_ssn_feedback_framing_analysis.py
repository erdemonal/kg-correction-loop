import pytest

from src.analyze_sosa_ssn_feedback_framing import (
    cochran_q,
    cost_summary,
    exact_mcnemar,
    framing_summary,
    holm_adjust,
    validate_successful_outcome,
    wilson_interval,
)


def outcome(**overrides):
    value = {
        "controlled_target_removed": False,
        "owl_consistent": False,
        "exact_reference_recovery": False,
        "collateral_edit": True,
        "new_raw_shacl_findings": False,
        "new_grounding_findings": True,
        "owl_inconsistent_after_target_removal": False,
        "output_failure": None,
        "edit_distance_from_injected": 2,
        "edit_distance_from_clean_reference": 3,
    }
    value.update(overrides)
    return value


def test_wilson_interval_keeps_count_denominator_and_rate():
    value = wilson_interval(19, 30)
    assert value["count"] == 19
    assert value["n"] == 30
    assert value["rate"] == 19 / 30
    assert value["lower_95"] < value["rate"] < value["upper_95"]


def test_cochran_q_detects_the_locked_pattern():
    matrix = [[False, False, index < 19] for index in range(30)]
    value = cochran_q(matrix)
    assert value["statistic"] == 38.0
    assert value["degrees_of_freedom"] == 2
    assert value["p_value"] == pytest.approx(5.602796437537268e-9)


def test_cochran_q_handles_no_discordance():
    value = cochran_q([[False, False, False] for _ in range(30)])
    assert value["statistic"] == 0.0
    assert value["p_value"] == 1.0


def test_exact_mcnemar_keeps_paired_transitions():
    left = [True] * 19 + [False] * 11
    right = [False] * 30
    value = exact_mcnemar(left, right)
    assert value["left_only"] == 19
    assert value["right_only"] == 0
    assert value["neither"] == 11
    assert value["p_value_raw"] == pytest.approx(3.814697265625e-6)
    assert value["risk_difference_left_minus_right"] == pytest.approx(19 / 30)


def test_holm_adjustment_is_monotone_for_tied_small_p_values():
    adjusted = holm_adjust([3.814697265625e-6, 3.814697265625e-6, 1.0])
    assert adjusted[0] == adjusted[1] == pytest.approx(1.1444091796875e-5)
    assert adjusted[2] == 1.0


def test_framing_summary_uses_all_cases_for_primary_and_usable_for_side_effects():
    rows = [
        {"outcome": outcome(controlled_target_removed=True, collateral_edit=False)},
        {"outcome": outcome()},
        {
            "outcome": outcome(
                output_failure="unparseable_output",
                owl_consistent=None,
                collateral_edit=None,
                new_raw_shacl_findings=None,
                new_grounding_findings=None,
                edit_distance_from_injected=None,
                edit_distance_from_clean_reference=None,
            )
        },
    ]
    value = framing_summary(rows)
    assert value["controlled_target_removed"]["count"] == 1
    assert value["controlled_target_removed"]["n"] == 3
    assert value["output_failure"]["count"] == 1
    assert value["among_usable_outputs"]["collateral_edit"]["n"] == 2
    assert value["among_usable_outputs"]["collateral_edit"]["count"] == 1


def test_successful_outcome_is_recomputed_from_post_repair_measurement():
    row = {
        "case_id": "case",
        "framing": "explanation",
        "outcome": outcome(
            controlled_target_removed=True,
            owl_consistent=True,
            exact_reference_recovery=True,
            collateral_edit=False,
            new_raw_shacl_findings=True,
            new_grounding_findings=False,
            edit_distance_from_injected=1,
            edit_distance_from_clean_reference=0,
        ),
        "post_repair_measurement": {
            "controlled_target_removed": True,
            "owl_consistent": True,
            "new_raw_shacl_violation_ids": ["new"],
            "grounding": {"new_actionable_violation_ids": []},
            "reference": {
                "reference_recovery": True,
                "collateral_symmetric_difference": 0,
                "reference_symmetric_difference": 0,
            },
            "edit": {"symmetric_difference_from_injected": 1},
        },
    }
    validate_successful_outcome(row)


def test_successful_outcome_rejects_a_recorded_metric_mismatch():
    row = {
        "case_id": "case",
        "framing": "verdict",
        "outcome": outcome(collateral_edit=False),
        "post_repair_measurement": {
            "controlled_target_removed": False,
            "owl_consistent": False,
            "new_raw_shacl_violation_ids": [],
            "grounding": {"new_actionable_violation_ids": ["new"]},
            "reference": {
                "reference_recovery": False,
                "collateral_symmetric_difference": 1,
                "reference_symmetric_difference": 3,
            },
            "edit": {"symmetric_difference_from_injected": 2},
        },
    }
    with pytest.raises(RuntimeError, match="recorded outcome"):
        validate_successful_outcome(row)


def test_cost_summary_counts_a_shared_grounding_judgment_once_per_case():
    judgment = {
        "source": "repair_round",
        "triple": ["s", "p", "o"],
        "verdict": "SUPPORTED",
        "prompt_eval_count": 10,
        "eval_count": 2,
        "total_duration_ns": 100,
    }
    repair = {"prompt_eval_count": 20, "eval_count": 3, "total_duration_ns": 200}
    rows = [
        {
            "case_id": "case",
            "repair": repair,
            "post_repair_measurement": {"grounding": {"judgments": [judgment]}},
        },
        {
            "case_id": "case",
            "repair": repair,
            "post_repair_measurement": {"grounding": {"judgments": [judgment]}},
        },
    ]
    metadata = {
        "created_at_utc": "2026-08-30T14:00:00+00:00",
        "completed_at_utc": "2026-08-30T14:01:00+00:00",
    }
    value = cost_summary(rows, metadata)
    assert value["repair_generation"]["calls"] == 2
    assert value["live_grounding"]["unique_calls_within_case"] == 1
    assert value["live_grounding"]["prompt_tokens"] == 10
    assert value["wall_seconds"] == 60.0
