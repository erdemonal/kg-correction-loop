
from src.report_repair_dynamics import (
    condition_rows,
    draft_results_notes,
    first_resolution_rows,
    latex_escape,
    stop_rows,
    validate_analysis,
)


def make_payload():
    condition_counts = {
        "disjointness": {
            "n": 10,
            "final_target_resolved": 7,
            "ever_target_resolved": 10,
            "last_validated_target_resolved": 9,
            "reference_recovery": 2,
            "validated_stop": 5,
            "validated_state": 5,
            "any_collateral_edit": 10,
            "any_new_violation": 8,
        },
        "domain_range": {
            "n": 10,
            "final_target_resolved": 3,
            "ever_target_resolved": 5,
            "last_validated_target_resolved": 5,
            "reference_recovery": 0,
            "validated_stop": 0,
            "validated_state": 0,
            "any_collateral_edit": 8,
            "any_new_violation": 8,
        },
        "cardinality": {
            "n": 10,
            "final_target_resolved": 9,
            "ever_target_resolved": 10,
            "last_validated_target_resolved": 9,
            "reference_recovery": 7,
            "validated_stop": 8,
            "validated_state": 8,
            "any_collateral_edit": 3,
            "any_new_violation": 2,
        },
        "temporal": {
            "n": 10,
            "final_target_resolved": 9,
            "ever_target_resolved": 9,
            "last_validated_target_resolved": 9,
            "reference_recovery": 8,
            "validated_stop": 9,
            "validated_state": 9,
            "any_collateral_edit": 1,
            "any_new_violation": 1,
        },
        "grounding": {
            "n": 10,
            "final_target_resolved": 9,
            "ever_target_resolved": 9,
            "last_validated_target_resolved": 10,
            "reference_recovery": 5,
            "validated_stop": 8,
            "validated_state": 9,
            "any_collateral_edit": 4,
            "any_new_violation": 1,
        },
    }

    overall = {
        "n": 50,
        "received_initial_feedback": 49,
        "final_target_resolved": 37,
        "final_target_resolved_given_feedback": 37,
        "ever_target_resolved": 43,
        "last_validated_target_resolved": 42,
        "graph_regressed_after_resolution": 1,
        "output_failure_after_resolution": 5,
        "reference_recovery": 22,
        "validated_stop": 30,
        "validated_state": 31,
        "any_collateral_edit": 26,
        "any_new_violation": 20,
        "stop_reasons": {
            "validated": 30,
            "output_failure": 9,
            "max_rounds": 4,
            "oscillation": 3,
            "stalled": 3,
            "no_feedback": 1,
        },
        "first_resolution_round": {
            "1": 41,
            "2": 1,
            "3": 1,
            "never": 7,
        },
    }

    return {
        "analysis_unit": "controlled case",
        "overall": {"counts": overall, "intervals": {}},
        "by_condition": {
            condition: {"counts": counts, "intervals": {}}
            for condition, counts in condition_counts.items()
        },
        "cases": [{"id": f"case-{i}"} for i in range(50)],
        "analysis_provenance": {
            "analysis_git_head": "abc",
            "analysis_script_sha256": "def",
        },
        "input": {
            "trajectory_sha256": "ghi",
            "run_git_head": "jkl",
        },
    }


def test_validate_analysis_accepts_complete_payload():
    validate_analysis(make_payload())


def test_condition_rows_preserve_locked_counts():
    rows = condition_rows(make_payload())
    selected = {row["condition"]: row for row in rows}

    assert selected["domain_range"]["final_target_resolved"] == 3
    assert selected["domain_range"]["reference_recovery"] == 0
    assert selected["disjointness"]["any_collateral_edit"] == 10


def test_first_resolution_rows_keep_never_separate():
    rows = first_resolution_rows(make_payload())

    assert [(row["round"], row["count"]) for row in rows] == [
        ("1", 41),
        ("2", 1),
        ("3", 1),
        ("never", 7),
    ]


def test_stop_rows_use_scientific_stop_outcomes():
    rows = stop_rows(make_payload())
    values = {
        row["stop_reason"]: row["count"]
        for row in rows
    }

    assert values == {
        "validated": 30,
        "output_failure": 9,
        "max_rounds": 4,
        "oscillation": 3,
        "stalled": 3,
        "no_feedback": 1,
    }


def test_results_notes_keep_reference_graph_wording():
    text = draft_results_notes(make_payload())

    assert "clean reference graph" in text
    assert "source graph" not in text
    assert "37/50" in text
    assert "22/50" in text
    assert "41 cases" in text


def test_latex_escape_handles_underscores():
    assert latex_escape("domain_range") == r"domain\_range"
