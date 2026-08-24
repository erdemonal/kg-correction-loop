import json
from pathlib import Path

from src.analyze_repair_dynamics import (
    bootstrap_intervals,
    case_row,
    summary_counts,
    validate_full_run_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def validation(
    *,
    target=False,
    actionable=None,
    reference_recovery=False,
    reference_difference=1,
    collateral=0,
):
    actionable = actionable or []

    return {
        "symbolic": {
            "shacl": {
                "conforms": not actionable,
                "violations": [],
            },
            "owl_consistent": True,
        },
        "grounding": {
            "judgments": [],
            "clean_baseline_unsupported_excluded": [],
        },
        "actionable_feedback": actionable,
        "target_resolved": target,
        "reference": {
            "reference_recovery": reference_recovery,
            "clean_reference_removed": [],
            "new_not_in_clean_reference": [],
            "reference_symmetric_difference": reference_difference,
            "collateral_removed": [],
            "collateral_added": [],
            "collateral_symmetric_difference": collateral,
        },
    }


def make_trajectory(
    *,
    case_id="case-1",
    condition="cardinality",
    received=True,
    round1_target=True,
    final_target=True,
    stop_reason="validated",
    validated_state=True,
    reference_recovery=True,
    output_failure=None,
    extra_rounds=None,
):
    feedback = [
        {
            "validator": "raw_shacl",
            "violation_id": "initial",
            "error_type": "cardinality_breach",
            "focus": "A",
            "path": "p",
            "message": "Missing value",
        }
    ] if received else []

    rounds = [
        {
            "round": 0,
            "triples": [],
            "validation": validation(
                target=False,
                actionable=feedback,
            ),
            "new_violation_ids": [],
        },
        {
            "round": 1,
            "repair": {
                "parse": {
                    "ok": True,
                    "failure": None,
                }
            },
            "triples": [["A", "p", "B"]],
            "validation": validation(
                target=round1_target,
                actionable=[],
                reference_recovery=reference_recovery,
                reference_difference=(
                    0 if reference_recovery else 1
                ),
            ),
            "new_violation_ids": [],
        },
    ]

    if extra_rounds:
        rounds.extend(extra_rounds)

    first = None

    for row in rounds[1:]:
        current = row.get("validation")

        if (
            isinstance(current, dict)
            and current["target_resolved"]
        ):
            first = row["round"]
            break

    return {
        "id": case_id,
        "domain": "movie",
        "condition": condition,
        "received_initial_feedback": received,
        "initial_feedback_sources": (
            ["raw_shacl"] if received else []
        ),
        "rounds": rounds,
        "final": {
            "stop_reason": stop_reason,
            "repair_rounds": len(rounds) - 1,
            "target_resolved": final_target,
            "validated_state": validated_state,
            "reference_recovery": reference_recovery,
            "rounds_to_resolution": first,
            "output_failure": output_failure,
        },
    }


def test_case_row_distinguishes_first_resolution_from_regression():
    round2 = {
        "round": 2,
        "repair": {
            "parse": {
                "ok": True,
                "failure": None,
            }
        },
        "triples": [],
        "validation": validation(
            target=False,
            actionable=[
                {
                    "validator": "grounding_v3",
                    "violation_id": "new-grounding",
                    "error_type": "grounding_error",
                    "focus": "p(A, B)",
                    "message": "Unsupported",
                }
            ],
            collateral=2,
        ),
        "new_violation_ids": ["new-grounding"],
    }
    trajectory = make_trajectory(
        round1_target=True,
        final_target=False,
        stop_reason="max_rounds",
        validated_state=False,
        reference_recovery=False,
        extra_rounds=[round2],
    )

    row = case_row(trajectory)

    assert row["ever_target_resolved"] is True
    assert row["first_resolution_round"] == 1
    assert row["final_target_resolved"] is False
    assert row["last_validated_target_resolved"] is False
    assert row["graph_regressed_after_resolution"] is True
    assert row["output_failure_after_resolution"] is False
    assert row["any_collateral_edit"] is True
    assert row["peak_collateral_difference"] == 2
    assert row["distinct_new_violation_count"] == 1
    assert row["new_grounding_violation_count"] == 1


def test_output_failure_does_not_erase_prior_resolution_timing():
    failure_round = {
        "round": 2,
        "repair": {
            "parse": {
                "ok": False,
                "failure": "unparseable_output",
            }
        },
        "triples": None,
        "validation": None,
        "new_violation_ids": [],
    }
    trajectory = make_trajectory(
        round1_target=True,
        final_target=False,
        stop_reason="output_failure",
        validated_state=False,
        reference_recovery=False,
        output_failure="unparseable_output",
        extra_rounds=[failure_round],
    )

    row = case_row(trajectory)

    assert row["ever_target_resolved"] is True
    assert row["first_resolution_round"] == 1
    assert row["final_target_resolved"] is False
    assert row["last_validated_target_resolved"] is True
    assert row["graph_regressed_after_resolution"] is False
    assert row["output_failure_after_resolution"] is True
    assert row["last_validated_round"] == 1
    assert row["output_failure"] == "unparseable_output"


def test_summary_keeps_validated_state_separate_from_validated_stop():
    repaired = case_row(make_trajectory())
    no_feedback = {
        "id": "case-2",
        "domain": "movie",
        "condition": "grounding",
        "received_initial_feedback": False,
        "initial_feedback_sources": [],
        "rounds": [
            {
                "round": 0,
                "triples": [["A", "p", "B"]],
                "validation": validation(
                    target=False,
                    actionable=[],
                    reference_recovery=False,
                ),
                "new_violation_ids": [],
            }
        ],
        "final": {
            "stop_reason": "no_feedback",
            "repair_rounds": 0,
            "target_resolved": False,
            "validated_state": True,
            "reference_recovery": False,
            "rounds_to_resolution": None,
            "output_failure": None,
        },
    }

    counts = summary_counts(
        [repaired, case_row(no_feedback)]
    )

    assert counts["validated_state"] == 2
    assert counts["validated_stop"] == 1
    assert counts["final_target_resolved"] == 1
    assert counts["received_initial_feedback"] == 1


def test_bootstrap_is_deterministic():
    rows = []

    for index in range(10):
        row = case_row(
            make_trajectory(
                case_id=f"case-{index}",
                final_target=index < 7,
                round1_target=index < 7,
                validated_state=index < 7,
                reference_recovery=index < 4,
                stop_reason=(
                    "validated"
                    if index < 7
                    else "max_rounds"
                ),
            )
        )
        rows.append(row)

    first = bootstrap_intervals(
        rows,
        samples=200,
        seed=42,
        stratify_by_condition=False,
    )
    second = bootstrap_intervals(
        rows,
        samples=200,
        seed=42,
        stratify_by_condition=False,
    )

    assert first == second
    assert first["final_target_resolution"]["estimate"] == 0.7
    assert first["reference_recovery"]["estimate"] == 0.4


def test_main_run_metadata_requirements():
    metadata = {
        "cases": 50,
        "start": 1,
        "limit": None,
        "case_id": None,
        "max_repair_rounds": 5,
        "invalid_model_output_retry": False,
    }

    validate_full_run_metadata(metadata)

    invalid = dict(metadata)
    invalid["cases"] = 5

    try:
        validate_full_run_metadata(invalid)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "partial run metadata must be rejected"
        )


def test_analysis_spec_records_post_run_summary_rules():
    payload = json.loads(
        (
            ROOT
            / "experiments"
            / "repair_analysis_spec.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["version"] == 2
    assert payload["analysis_unit"] == "controlled case"
    assert payload["bootstrap"]["samples"] == 10000
    assert payload["bootstrap"]["seed"] == 42
    assert payload["bootstrap"]["confidence_level"] == 0.95
    assert (
        payload["validated_state_note"]
        == (
            "validated_state means that no actionable feedback "
            "remains. no_feedback is reported separately and is "
            "not counted as a validated stop."
        )
    )
