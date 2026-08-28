from src.analyze_sosa_ssn_confirmatory import (
    confusion,
    grounding_group_summary,
    prf,
    repair_group_summary,
    round_cost,
    scores_from_round,
    validator_coverage_summary,
    wilson_interval,
)


def test_wilson_interval_records_count_denominator_and_rate():
    value = wilson_interval(24, 30)
    assert value["count"] == 24
    assert value["n"] == 30
    assert value["rate"] == 0.8
    assert 0.62 < value["lower_95"] < 0.64
    assert 0.90 < value["upper_95"] < 0.91


def test_reference_scores_use_recorded_missing_and_extra_triples():
    round_row = {
        "triples": [["s", "p", "kept"], ["s", "p", "extra"]],
        "validation": {
            "reference": {
                "new_not_in_clean_reference": [["s", "p", "extra"]],
                "clean_reference_removed": [["s", "p", "missing"]],
                "reference_recovery": False,
                "reference_symmetric_difference": 2,
                "collateral_symmetric_difference": 1,
                "collateral_added": [["s", "p", "extra"]],
                "collateral_removed": [],
            }
        },
    }
    scores = scores_from_round(round_row)
    assert scores["true_positive"] == 1
    assert scores["false_positive"] == 1
    assert scores["false_negative"] == 1
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5
    assert scores["reference_difference"] == 2


def test_live_grounding_cost_counts_a_cached_triple_once_per_case():
    judgment = {
        "source": "repair_round",
        "triple": ["s", "p", "o"],
        "prompt_eval_count": 10,
        "eval_count": 2,
        "total_duration_ns": 50,
    }
    round_row = {
        "validation": {"grounding": {"judgments": [judgment]}},
        "repair": {
            "prompt_eval_count": 20,
            "eval_count": 3,
            "total_duration_ns": 70,
            "parse": {"ok": True, "failure": None},
        },
    }
    seen = set()
    first = round_cost(round_row, seen)
    second = round_cost(round_row, seen)
    assert first["repair_calls"] == second["repair_calls"] == 1
    assert first["live_grounding_calls"] == 1
    assert first["live_grounding_prompt_tokens"] == 10
    assert second["live_grounding_calls"] == 0
    assert second["live_grounding_prompt_tokens"] == 0


def target_row(expected, observed):
    return {
        "grounding_expected_target_error": expected,
        "grounding_observed_target_error": observed,
        "grounding_target_matches_expected": expected == observed,
        "clean_grounding_error": True,
        "injected_grounding_error": True,
        "clean_grounding_unsupported_count": 1,
        "injected_grounding_unsupported_count": 2,
    }


def test_grounding_confusion_keeps_positive_and_negative_targets_separate():
    rows = [
        target_row(True, True),
        target_row(True, False),
        target_row(False, False),
        target_row(False, True),
    ]
    value = confusion(rows)
    assert value == {
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 1,
        "true_negative": 1,
        "sensitivity": 0.5,
        "specificity": 0.5,
        "precision": 0.5,
        "accuracy": 0.5,
    }
    summary = grounding_group_summary(rows)
    assert summary["target_matches_expected"]["count"] == 2
    assert summary["clean_graph_flagged"]["count"] == 4


def test_validator_coverage_keeps_overlap_patterns():
    rows = [
        {"initial_feedback_sources": "raw_shacl"},
        {"initial_feedback_sources": "grounding_v3+raw_shacl"},
        {"initial_feedback_sources": "grounding_v3+owl_consistency+raw_shacl"},
    ]
    summary = validator_coverage_summary(rows)
    assert summary["raw_shacl"]["count"] == 3
    assert summary["owl_consistency"]["count"] == 1
    assert summary["grounding_v3"]["count"] == 2
    assert summary["overlap_patterns"] == {
        "grounding_v3+owl_consistency+raw_shacl": 1,
        "grounding_v3+raw_shacl": 1,
        "raw_shacl": 1,
    }


def repair_row(**overrides):
    row = {
        "received_initial_feedback": True,
        "end_to_end_target_resolved": True,
        "ever_target_resolved": True,
        "last_validated_target_resolved": True,
        "validated_state": True,
        "end_to_end_reference_recovery": True,
        "last_validated_reference_recovery": True,
        "output_failure": "",
        "target_regressed_after_resolution": False,
        "output_failure_after_resolution": False,
        "any_collateral_edit": False,
        "last_validated_collateral_difference": 0,
        "any_new_violation": False,
        "repair_rounds": 1,
        "stop_reason": "validated",
        "first_resolution_round": 1,
        "initial_f1": 0.9,
        "last_validated_f1": 1.0,
        "f1_delta": 0.1,
        "f1_change": "improved",
        "last_validated_reference_difference": 0,
        "distinct_new_violation_count": 0,
    }
    row.update(overrides)
    return row


def test_repair_summary_does_not_merge_ever_final_and_last_validated_outcomes():
    rows = [
        repair_row(),
        repair_row(
            end_to_end_target_resolved=False,
            validated_state=False,
            end_to_end_reference_recovery=False,
            output_failure="unparseable_output",
            output_failure_after_resolution=True,
            stop_reason="output_failure",
            last_validated_reference_recovery=False,
            f1_delta=-0.1,
            f1_change="worsened",
        ),
        repair_row(
            end_to_end_target_resolved=False,
            ever_target_resolved=False,
            last_validated_target_resolved=False,
            validated_state=False,
            end_to_end_reference_recovery=False,
            last_validated_reference_recovery=False,
            first_resolution_round="",
            stop_reason="max_rounds",
            f1_delta=0.0,
            f1_change="unchanged",
        ),
    ]
    summary = repair_group_summary(rows)
    assert summary["end_to_end_target_resolution"]["count"] == 1
    assert summary["ever_target_resolution"]["count"] == 2
    assert summary["last_validated_target_resolution"]["count"] == 2
    assert summary["validated_state"]["count"] == 1
    assert summary["output_failure"]["count"] == 1
    assert summary["output_failure_given_ever_resolved"]["count"] == 1
    assert summary["paired_f1_changes"] == {
        "improved": 1,
        "unchanged": 1,
        "worsened": 1,
    }


def test_prf_handles_exact_empty_graph_as_perfect():
    assert prf(0, 0, 0) == {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
