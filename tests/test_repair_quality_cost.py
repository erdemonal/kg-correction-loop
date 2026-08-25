import json
from pathlib import Path

from src.analyze_repair_dynamics import TRAJECTORIES, read_json, read_jsonl
from src.analyze_repair_quality_cost import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    SURVIVOR_ROUND_LABEL,
    SURVIVOR_ROUND_WARNING,
    CONVENTION_SUMMARY_NOTE,
    scores_from_reference,
    analyze_records,
    analyze_trajectory,
    bootstrap_mean,
    load_spec,
    new_grounding_cost,
    prf,
    write_outputs,
)


def validation(
    *,
    extra=None,
    missing=None,
    target=False,
    recovered=False,
    collateral_removed=None,
    judgments=None,
    feedback=None,
):
    extra = extra or []
    missing = missing or []
    return {
        "actionable_feedback": feedback or [],
        "target_resolved": target,
        "grounding": {
            "judgments": judgments or [],
            "clean_baseline_unsupported_excluded": [],
        },
        "reference": {
            "reference_recovery": recovered,
            "clean_reference_removed": missing,
            "new_not_in_clean_reference": extra,
            "reference_symmetric_difference": len(missing) + len(extra),
            "collateral_removed": collateral_removed or [],
            "collateral_added": extra,
            "collateral_symmetric_difference": len(collateral_removed or []) + len(extra),
        },
    }


def repair(prompt=10, generated=4, duration=1000, ok=True, failure=None):
    return {
        "prompt_eval_count": prompt,
        "eval_count": generated,
        "total_duration_ns": duration,
        "parse": {"ok": ok, "failure": failure, "triples": []},
    }


def test_spec_does_not_claim_wall_clock_or_human_faithfulness():
    spec = load_spec()
    assert spec["models_or_validators_run_by_analysis"] is False
    assert spec["analysis_unit"] == "controlled case"
    unavailable = " ".join(spec["unavailable"])
    assert "wall clock" in unavailable
    assert "reasoner" in unavailable
    assert "monetary" in unavailable
    assert "human source faithfulness" in unavailable
    assert "Text2KGBench benchmark F1" in spec["unavailable"]
    assert spec["survivor_round_summaries"] == SURVIVOR_ROUND_LABEL
    assert spec["survivor_round_warning"] == SURVIVOR_ROUND_WARNING
    assert spec["all_case_convention_based_summary"] == CONVENTION_SUMMARY_NOTE
    assert "not a population confidence interval" in spec["bootstrap_interpretation"]


def test_prf_matches_the_clean_reference():
    current = [["A", "p", "1"], ["A", "p", "2"]]
    extra = [["A", "p", "2"]]
    missing = [["A", "p", "3"]]
    scores = scores_from_reference(
        current,
        {
            "new_not_in_clean_reference": extra,
            "clean_reference_removed": missing,
            "reference_recovery": False,
            "collateral_removed": [["A", "p", "3"]],
            "collateral_added": extra,
        },
    )
    assert scores["true_positive"] == 1
    assert scores["false_positive"] == 1
    assert scores["false_negative"] == 1
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5
    assert scores["collateral_removed"] == 1


def test_empty_graphs_have_perfect_prf():
    scores = prf(0, 0, 0)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_cached_grounding_listings_are_not_extra_calls():
    seen = set()
    judgment = {
        "triple": ["A", "p", "1"],
        "source": "repair_round",
        "prompt_eval_count": 50,
        "eval_count": 20,
        "total_duration_ns": 9,
    }
    first = new_grounding_cost({"validation": {"grounding": {"judgments": [judgment]}}}, seen)
    second = new_grounding_cost({"validation": {"grounding": {"judgments": [judgment]}}}, seen)
    frozen = new_grounding_cost(
        {
            "validation": {
                "grounding": {
                    "judgments": [{"triple": ["A", "p", "1"], "source": "frozen_injected"}]
                }
            }
        },
        seen,
    )
    assert first["grounding_assessor_calls"] == 1
    assert second["grounding_assessor_calls"] == 0
    assert frozen["grounding_assessor_calls"] == 0
    assert first["grounding_prompt_eval_count"] == 50


def sample_case(stop="validated", output_failure=None, extra_round=None):
    feedback = [
        {
            "validator": "raw_shacl",
            "violation_id": "v1",
            "error_type": "cardinality_breach",
            "focus": "A",
            "path": "p",
            "message": "missing",
        }
    ]
    rounds = [
        {
            "round": 0,
            "triples": [["A", "p", "1"]],
            "validation": validation(
                missing=[["A", "p", "2"]],
                feedback=feedback,
            ),
            "new_violation_ids": [],
        },
        {
            "round": 1,
            "repair": repair(),
            "triples": [["A", "p", "1"], ["A", "p", "2"]],
            "validation": validation(target=True, recovered=True),
            "new_violation_ids": [],
        },
    ]
    if extra_round:
        rounds.append(extra_round)
    return {
        "id": "case-1",
        "domain": "movie",
        "condition": "cardinality",
        "received_initial_feedback": True,
        "initial_feedback_sources": ["raw_shacl"],
        "rounds": rounds,
        "final": {
            "stop_reason": stop,
            "repair_rounds": len(rounds) - 1,
            "target_resolved": stop != "output_failure",
            "validated_state": stop == "validated",
            "reference_recovery": stop != "output_failure",
            "rounds_to_resolution": 1,
            "output_failure": output_failure,
        },
    }


def empty_reference_case(*, last_triples=None, case_id="empty-1"):
    last_triples = [["B", "q", "1"]] if last_triples is None else last_triples
    feedback = [
        {
            "validator": "raw_shacl",
            "violation_id": "v1",
            "error_type": "domain_range_violation",
            "focus": "B",
            "path": "q",
            "message": "range",
        }
    ]
    return {
        "id": case_id,
        "domain": "movie",
        "condition": "domain_range",
        "received_initial_feedback": True,
        "initial_feedback_sources": ["raw_shacl"],
        "rounds": [
            {
                "round": 0,
                "triples": [["B", "q", "1"]],
                "validation": validation(extra=[["B", "q", "1"]], feedback=feedback),
                "new_violation_ids": [],
            },
            {
                "round": 1,
                "repair": repair(),
                "triples": last_triples,
                "validation": validation(extra=last_triples, target=True),
                "new_violation_ids": [],
            },
        ],
        "final": {
            "stop_reason": "validated",
            "repair_rounds": 1,
            "target_resolved": True,
            "validated_state": True,
            "reference_recovery": last_triples == [],
            "rounds_to_resolution": 1,
            "output_failure": None,
        },
    }


def test_output_failure_keeps_repair_cost_and_omits_graph_scores():
    failure_round = {
        "round": 2,
        "repair": repair(prompt=7, generated=3, duration=5, ok=False, failure="unparseable_output"),
        "triples": None,
        "validation": None,
        "new_violation_ids": [],
    }
    row = analyze_trajectory(
        sample_case(
            stop="output_failure",
            output_failure="unparseable_output",
            extra_round=failure_round,
        )
    )
    last = row["rounds"][-1]
    assert last["has_validated_graph"] is False
    assert last["precision"] is None
    assert last["f1"] is None
    assert last["repair_calls"] == 1
    assert last["repair_prompt_eval_count"] == 7
    assert last["parse_failure"] == "unparseable_output"
    assert row["end_to_end_target_resolved"] is False
    assert row["last_validated_f1"] == 1.0
    assert row["last_validated_f1"] != 0
    assert row["repair_calls"] == 2
    assert row["f1_delta"] == 1.0 - row["initial_f1"]
    assert last["f1"] is None
    assert len(row["transitions"]) == 1
    assert row["transitions"][0]["to_round"] == 1


def test_case_level_summary_does_not_treat_rounds_as_independent_cases():
    spec = load_spec()
    payload = analyze_records([sample_case()], {}, spec, verify_inputs=False)
    assert payload["overall"]["n"] == 1
    assert payload["overall"]["aggregation"] == "one last validated value per controlled case"
    assert payload["primary_f1"]["n"] == 1
    assert payload["primary_f1"]["primary"] is True
    assert payload["primary_f1"]["mean_last_validated_f1"] == 1.0
    assert payload["all_case_convention_based_summary"]["primary"] is False
    assert payload["by_condition"]["cardinality"]["n"] == 1


def test_writes_case_and_round_csv_without_claiming_unavailable_costs(tmp_path):
    spec = load_spec()
    payload = analyze_records([sample_case()], {}, spec, verify_inputs=False)
    json_path = tmp_path / "summary.json"
    cases_path = tmp_path / "cases.csv"
    rounds_path = tmp_path / "rounds.csv"
    write_outputs(payload, json_path, cases_path, rounds_path)
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert "end to end wall clock runtime" in written["unavailable"]
    assert "human source faithfulness scores" in written["unavailable"]
    assert written["survivor_round_warning"] == SURVIVOR_ROUND_WARNING
    assert cases_path.exists() and rounds_path.exists()
    assert "recorded_model_duration_ns" in cases_path.read_text(encoding="utf-8")


def test_paired_delta_is_last_validated_minus_initial():
    row = analyze_trajectory(sample_case())
    assert row["initial_f1"] == 2 / 3
    assert row["last_validated_f1"] == 1.0
    assert row["f1_delta"] == 1.0 - (2 / 3)
    payload = analyze_records([sample_case()], {}, load_spec(), verify_inputs=False)
    paired = payload["primary_f1"]
    assert paired["n"] == 1
    assert paired["mean_delta"] == row["f1_delta"]
    assert paired["median_delta"] == row["f1_delta"]
    assert paired["improved"] == 1
    assert paired["unchanged"] == 0
    assert paired["worsened"] == 0
    assert paired["bootstrap_mean_delta"]["estimate"] == row["f1_delta"]
    assert paired["bootstrap_mean_delta"]["samples"] == BOOTSTRAP_SAMPLES
    assert paired["bootstrap_mean_delta"]["seed"] == BOOTSTRAP_SEED
    assert "not a population confidence interval" in paired["bootstrap_mean_delta"]["interpretation"]


def test_bootstrap_mean_is_deterministic_with_seed_42():
    values = [0.1, -0.05, 0.2, 0.0, 0.4]
    first = bootstrap_mean(values)
    second = bootstrap_mean(values)
    assert first == second
    assert first["samples"] == 10000
    assert first["seed"] == 42
    other = bootstrap_mean(values, seed=43)
    assert other != first


def test_within_case_transitions_skip_failed_rounds_and_count_consecutive_graphs():
    drop_round = {
        "round": 2,
        "repair": repair(),
        "triples": [["A", "p", "1"]],
        "validation": validation(missing=[["A", "p", "2"]], target=True),
        "new_violation_ids": [],
    }
    failure_round = {
        "round": 3,
        "repair": repair(ok=False, failure="unparseable_output"),
        "triples": None,
        "validation": None,
        "new_violation_ids": [],
    }
    first = sample_case(extra_round=drop_round)
    first["rounds"].append(failure_round)
    first["final"]["stop_reason"] = "output_failure"
    first["final"]["output_failure"] = "unparseable_output"
    first["final"]["target_resolved"] = False
    first["final"]["validated_state"] = False
    first["final"]["reference_recovery"] = False
    first["final"]["repair_rounds"] = 3
    row = analyze_trajectory(first)
    assert [round_row["f1"] for round_row in row["rounds"]][-1] is None
    assert len(row["transitions"]) == 2
    assert row["transitions"][0]["repair_transition"] == 1
    assert row["transitions"][0]["change"] == "improved"
    assert row["transitions"][1]["repair_transition"] == 2
    assert row["transitions"][1]["change"] == "worsened"
    payload = analyze_records([first], {}, load_spec(), verify_inputs=False)
    transitions = payload["primary_f1"]["transitions"]
    assert transitions["n"] == 2
    assert transitions["improved"] == 1
    assert transitions["worsened"] == 1
    assert transitions["unchanged"] == 0
    assert transitions["by_repair_transition"][1]["n"] == 1
    assert transitions["by_repair_transition"][2]["n"] == 1
    assert payload["primary_f1"]["mean_delta"] == row["f1_delta"]


def test_survivor_round_summaries_do_not_track_the_same_cases_at_every_round():
    payload = analyze_records([sample_case()], {}, load_spec(), verify_inputs=False)
    summaries = payload["primary_f1"]["survivor_round_summaries"]
    assert summaries["label"] == SURVIVOR_ROUND_LABEL
    assert summaries["warning"] == SURVIVOR_ROUND_WARNING
    assert payload["survivor_round_warning"] == SURVIVOR_ROUND_WARNING
    assert "still running" in summaries["warning"]
    assert "fixed set of cases" in summaries["label"]
    assert summaries["by_round"][0]["n"] == 1
    assert summaries["by_round"][1]["n"] == 1


def test_empty_reference_cases_are_excluded_from_primary_f1_and_use_extra_counts():
    recovered = empty_reference_case(last_triples=[], case_id="empty-recover")
    grown = empty_reference_case(
        last_triples=[["B", "q", "1"], ["C", "q", "2"]],
        case_id="empty-grown",
    )
    payload = analyze_records(
        [sample_case(), recovered, grown],
        {},
        load_spec(),
        verify_inputs=False,
    )
    primary = payload["primary_f1"]
    empty = payload["empty_reference"]
    convention = payload["all_case_convention_based_summary"]
    assert primary["n"] == 1
    assert primary["primary"] is True
    assert all(row["initial_reference_size"] > 0 for row in payload["cases"] if not row["empty_reference"])
    assert {row["id"] for row in empty["cases"]} == {"empty-recover", "empty-grown"}
    assert empty["n"] == 2
    assert "mean_last_validated_f1" not in empty
    assert empty["primary_metric"] == "extra triples, not F1"
    assert empty["exact_empty_graph_recovery"] == 1
    assert empty["improved"] == 1
    assert empty["worsened"] == 1
    assert empty["unchanged"] == 0
    assert convention["n"] == 3
    assert convention["primary"] is False
    assert "not the primary graph quality estimate" in convention["note"]
    assert "computational convention" in convention["note"]


def test_frozen_trajectories_support_reconstructed_clean_reference_f1():
    rows = read_jsonl(TRAJECTORIES)
    spec = load_spec()
    payload = analyze_records(
        rows,
        read_json(Path("results/controlled_repair_trajectories.jsonl.meta.json")),
        spec,
    )
    assert payload["overall"]["n"] == 50
    assert payload["by_domain"]["movie"]["n"] == 25
    assert payload["by_domain"]["music"]["n"] == 25
    for condition in ("disjointness", "domain_range", "cardinality", "temporal", "grounding"):
        assert payload["by_condition"][condition]["n"] == 10
    validated_rounds = [
        round_row
        for case in payload["cases"]
        for round_row in case["rounds"]
        if round_row["has_validated_graph"]
    ]
    assert len(validated_rounds) == 138
    assert all(round_row["f1"] is not None for round_row in validated_rounds)
    failed = [case for case in payload["cases"] if case["output_failure"]]
    assert len(failed) == 9
    assert all(case["rounds"][-1]["f1"] is None for case in failed)
    assert payload["overall"]["sum_repair_calls"] == 97
    assert payload["overall"]["sum_grounding_assessor_calls"] == 103
    primary = payload["primary_f1"]
    assert primary["n"] == 40
    assert primary["primary"] is True
    assert all(case["initial_reference_size"] > 0 for case in payload["cases"] if not case["empty_reference"])
    assert sum(case["empty_reference"] for case in payload["cases"]) == 10
    primary_ids = {case["id"] for case in payload["cases"] if not case["empty_reference"]}
    empty_ids = {row["id"] for row in payload["empty_reference"]["cases"]}
    assert len(primary_ids) == 40
    assert len(empty_ids) == 10
    assert primary_ids.isdisjoint(empty_ids)
    assert "domain_range" not in primary["by_condition"]
    assert primary["improved"] + primary["unchanged"] + primary["worsened"] == 40
    nonempty = [case for case in payload["cases"] if not case["empty_reference"]]
    assert primary["mean_delta"] == sum(case["f1_delta"] for case in nonempty) / 40
    assert primary["bootstrap_mean_delta"]["samples"] == 10000
    assert primary["bootstrap_mean_delta"]["seed"] == 42
    assert "not a population confidence interval" in primary["bootstrap_mean_delta"]["interpretation"]
    convention = payload["all_case_convention_based_summary"]
    assert convention["n"] == 50
    assert convention["primary"] is False
    assert convention["includes_empty_reference_domain_range_cases"] == 10
    assert convention["note"] == CONVENTION_SUMMARY_NOTE
    empty = payload["empty_reference"]
    assert empty["n"] == 10
    assert "mean_last_validated_f1" not in empty
    assert empty["primary_metric"] == "extra triples, not F1"
    again = analyze_records(
        rows,
        read_json(Path("results/controlled_repair_trajectories.jsonl.meta.json")),
        spec,
    )
    assert again["primary_f1"]["bootstrap_mean_delta"] == primary["bootstrap_mean_delta"]
    expected_transitions = sum(
        max(0, sum(1 for round_row in case["rounds"] if round_row["has_validated_graph"]) - 1)
        for case in nonempty
    )
    assert primary["transitions"]["n"] == expected_transitions
    assert (
        primary["transitions"]["improved"]
        + primary["transitions"]["unchanged"]
        + primary["transitions"]["worsened"]
        == expected_transitions
    )
    assert primary["survivor_round_summaries"]["label"] == SURVIVOR_ROUND_LABEL
    assert "still running" in payload["survivor_round_warning"]
    diagnostic = payload["domain_range_diagnostic"]
    assert diagnostic["n"] == 10
    assert len(diagnostic["cases"]) == 10
    assert diagnostic["mean_last_validated_f1"] == 0.0
    for row in diagnostic["cases"]:
        reconstructed = prf(
            row["last_validated_true_positive"],
            row["last_validated_false_positive"],
            row["last_validated_false_negative"],
        )
        assert reconstructed["precision"] == row["last_validated_precision"]
        assert reconstructed["recall"] == row["last_validated_recall"]
        assert reconstructed["f1"] == row["last_validated_f1"]
        assert row["last_validated_f1"] == 0.0
        assert row["last_validated_true_positive"] == 0
        assert row["initial_reference_size"] == 0
