import json

import pytest

import src.run_sosa_ssn_feedback_framing as runner
from src.run_sosa_ssn_feedback_framing import (
    CONDITIONS,
    controlled_target_removed,
    edit_metrics,
    execution_schedule,
    failed_outcome,
    feedback_item,
    load_framing_spec,
    load_rq3_cases,
    prompt_outside_feedback,
    render_condition_prompt,
    run_task,
    validate_resume_prefix,
)


def inputs():
    spec, base = load_framing_spec()
    cases = load_rq3_cases(base)
    return spec, base, cases


def test_design_is_thirty_paired_cases_and_ninety_single_step_generations():
    spec, _base, cases = inputs()
    assert len(cases) == 30
    assert spec["sample"]["repair_generations"] == 90
    assert spec["design"]["repair_steps_per_condition"] == 1
    assert spec["design"]["paired_by_case"] is True
    assert spec["sample"]["outcome_based_selection"] is False


def test_all_cases_are_unique_controlled_disjointness_cases():
    _spec, _base, cases = inputs()
    assert len({case["case_id"] for case in cases}) == 30
    assert len({case["source_unit_id"] for case in cases}) == 30
    for case in cases:
        assert case["condition"] == "disjointness"
        added = case["primary_modification"]["added"]
        assert len(added) == 1
        assert added[0]["predicate"] == "type"
        assert added[0]["object"] == "SampleCollection"


def test_execution_schedule_is_complete_unique_and_rotated():
    _spec, _base, cases = inputs()
    rows = execution_schedule(cases)
    assert len(rows) == 90
    assert len({(row["case_id"], row["framing"]) for row in rows}) == 90
    assert [row["framing"] for row in rows[:3]] == list(CONDITIONS)
    assert [row["framing"] for row in rows[3:6]] == [
        "location",
        "explanation",
        "verdict",
    ]
    assert [row["framing"] for row in rows[6:9]] == [
        "explanation",
        "verdict",
        "location",
    ]


def test_feedback_arms_add_only_location_then_explanation():
    spec, _base, cases = inputs()
    case = cases[0]
    verdict = feedback_item(case, "verdict", spec)
    location = feedback_item(case, "location", spec)
    explanation = feedback_item(case, "explanation", spec)
    assert tuple(verdict) == tuple(spec["feedback"]["fixed_fields"])
    assert verdict["focus"] is None
    assert location["focus"] == explanation["focus"]
    assert location["message"] == verdict["message"]
    assert explanation["message"] != verdict["message"]
    for row in (verdict, location, explanation):
        assert row["validator"] == "owl_consistency"
        assert case["case_id"] not in row["violation_id"]
        assert "disjointness" not in row["violation_id"]
        assert row["error_type"] is None
        assert row["path"] is None


def test_prompt_content_outside_feedback_is_identical_across_arms():
    spec, base, cases = inputs()
    template = open(
        base["inputs"]["repair_prompt"]["path"], encoding="utf-8"
    ).read()
    prompts = {
        framing: render_condition_prompt(cases[0], framing, spec, template)[0]
        for framing in CONDITIONS
    }
    outside = {framing: prompt_outside_feedback(prompt) for framing, prompt in prompts.items()}
    assert len(set(outside.values())) == 1
    for framing, prompt in prompts.items():
        assert f'"framing": "{framing}"' not in prompt
        assert "clean reference" not in prompt.lower()
        assert "scaffold" not in prompt.lower()


def test_explanation_names_conflict_without_stating_a_repair_action():
    spec, _base, cases = inputs()
    message = feedback_item(cases[0], "explanation", spec)["message"]
    assert "ObservationCollection" in message
    assert "SampleCollection" in message
    assert "disjoint" in message
    for forbidden in ("remove", "delete", "replace", "add the", "correct triple"):
        assert forbidden not in message.lower()


def test_controlled_target_removal_is_not_arbitrary_graph_change():
    _spec, _base, cases = inputs()
    case = cases[0]
    injected = [
        [row["subject"], row["predicate"], row["object"]]
        for row in case["injected_content_triples"]
    ]
    assert controlled_target_removed(case, injected) is False
    target = tuple(
        case["primary_modification"]["added"][0][key]
        for key in ("subject", "predicate", "object")
    )
    changed_other = list(injected)
    changed_other.pop(next(index for index, row in enumerate(injected) if tuple(row) != target))
    assert controlled_target_removed(case, changed_other) is False
    repaired = [row for row in injected if tuple(row) != target]
    assert controlled_target_removed(case, repaired) is True


def test_edit_metrics_use_the_same_injected_start_for_every_arm():
    value = edit_metrics(
        [("s", "p", "old"), ("s", "q", "kept")],
        [("s", "q", "kept"), ("s", "p", "new")],
    )
    assert value["symmetric_difference_from_injected"] == 2
    assert value["graph_size_delta"] == 0
    assert value["removed_from_injected"] == [["s", "p", "old"]]
    assert value["added_to_injected"] == [["s", "p", "new"]]


def test_output_failure_has_no_fabricated_graph_measurements():
    outcome = failed_outcome("unparseable_output")
    assert outcome["controlled_target_removed"] is False
    assert outcome["exact_reference_recovery"] is False
    assert outcome["output_failure"] == "unparseable_output"
    for field in (
        "owl_consistent",
        "collateral_edit",
        "new_raw_shacl_findings",
        "new_grounding_findings",
        "edit_distance_from_injected",
        "edit_distance_from_clean_reference",
    ):
        assert outcome[field] is None


def test_resume_accepts_only_the_fixed_schedule_prefix():
    _spec, _base, cases = inputs()
    schedule = execution_schedule(cases)
    validate_resume_prefix(schedule[:7], schedule)
    changed = json.loads(json.dumps(schedule[:7]))
    changed[-1]["framing"] = "location"
    with pytest.raises(RuntimeError, match="fixed schedule prefix"):
        validate_resume_prefix(changed, schedule)


def test_each_arm_makes_one_generation_and_starts_from_injected_graph(monkeypatch):
    spec, base, cases = inputs()
    case = cases[0]
    clean = [
        (row["subject"], row["predicate"], row["object"])
        for row in case["clean_content_triples"]
    ]
    response = "\n".join(
        f"{predicate}({subject}, {obj})"
        for subject, predicate, obj in clean
    )
    calls = []

    def fake_generate(prompt, _base):
        calls.append(prompt)
        return {
            "raw_response": response,
            "model": spec["generation"]["model"],
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
            "total_duration_ns": 1,
        }

    monkeypatch.setattr(runner, "generate_repair", fake_generate)
    monkeypatch.setattr(
        runner,
        "validate_symbolic_state",
        lambda *_args: {"shacl": {"violations": []}, "owl_consistent": True},
    )
    monkeypatch.setattr(runner, "clean_unsupported", lambda *_args: set())
    monkeypatch.setattr(
        runner,
        "later_grounding_feedback",
        lambda *_args: ([], []),
    )
    initial = {
        "triples": [
            [row["subject"], row["predicate"], row["object"]]
            for row in case["injected_content_triples"]
        ],
        "shacl": {"violations": []},
        "shacl_violation_ids": [],
        "owl_consistent": False,
        "grounding_actionable_violation_ids": [],
    }
    task = execution_schedule([case])[0]
    template = open(
        base["inputs"]["repair_prompt"]["path"], encoding="utf-8"
    ).read()
    row = run_task(
        task,
        case,
        spec,
        base,
        template,
        "unused grounding template",
        {},
        initial,
        None,
        None,
    )
    assert len(calls) == 1
    assert row["framing"] == "verdict"
    assert row["outcome"]["controlled_target_removed"] is True
    assert row["outcome"]["owl_consistent"] is True
    assert row["outcome"]["exact_reference_recovery"] is True
    assert row["outcome"]["output_failure"] is None
