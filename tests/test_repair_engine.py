from src.repair_engine import (
    primary_delta,
    reference_metrics,
    stop_reason_after_repair,
    trajectory_state,
)


def feedback(name):
    return [
        {
            "violation_id": name,
        }
    ]


def test_primary_delta():
    clean = [
        ["A", "p", "B"],
        ["A", "q", "C"],
    ]
    injected = [
        ["A", "p", "B"],
        ["A", "q", "D"],
    ]

    result = primary_delta(clean, injected)

    assert result["removed"] == {
        ("A", "q", "C"),
    }
    assert result["added"] == {
        ("A", "q", "D"),
    }


def test_reference_metrics_distinguish_primary_from_collateral():
    clean = [
        ["A", "p", "B"],
        ["A", "q", "C"],
    ]
    injected = [
        ["A", "p", "B"],
        ["A", "q", "D"],
    ]
    current = [
        ["A", "q", "D"],
        ["X", "r", "Y"],
    ]

    result = reference_metrics(
        clean,
        injected,
        current,
    )

    assert result["reference_recovery"] is False
    assert result["reference_symmetric_difference"] == 4
    assert result["collateral_removed"] == [
        ["A", "p", "B"]
    ]
    assert result["collateral_added"] == [
        ["X", "r", "Y"]
    ]
    assert result["collateral_symmetric_difference"] == 2


def test_round_one_same_state_does_not_stall():
    state = trajectory_state(
        [["A", "p", "B"]],
        feedback("v1"),
    )

    reason = stop_reason_after_repair(
        1,
        state,
        [],
        feedback("v1"),
        5,
    )

    assert reason is None


def test_round_two_same_state_stalls():
    state = trajectory_state(
        [["A", "p", "B"]],
        feedback("v1"),
    )

    reason = stop_reason_after_repair(
        2,
        state,
        [state],
        feedback("v1"),
        5,
    )

    assert reason == "stalled"


def test_nonadjacent_repair_state_is_oscillation():
    first = trajectory_state(
        [["A", "p", "B"]],
        feedback("v1"),
    )
    second = trajectory_state(
        [["A", "p", "C"]],
        feedback("v2"),
    )

    reason = stop_reason_after_repair(
        3,
        first,
        [first, second],
        feedback("v1"),
        5,
    )

    assert reason == "oscillation"


def test_no_actionable_feedback_is_validated():
    state = trajectory_state(
        [["A", "p", "B"]],
        [],
    )

    reason = stop_reason_after_repair(
        1,
        state,
        [],
        [],
        5,
    )

    assert reason == "validated"


def test_round_five_stops_at_cap():
    current = trajectory_state(
        [["A", "p", "E"]],
        feedback("v5"),
    )
    history = [
        trajectory_state(
            [["A", "p", str(index)]],
            feedback(f"v{index}"),
        )
        for index in range(1, 5)
    ]

    reason = stop_reason_after_repair(
        5,
        current,
        history,
        feedback("v5"),
        5,
    )

    assert reason == "max_rounds"
