import copy
import hashlib
import json

import pytest

from src.analyze_owl_feedback_framing import (
    CONDITIONS,
    GRAPH_FIELDS,
    paired_target_comparison,
    rate,
    summarize_rows,
    validate_feedback_and_prompt,
    validate_outcome,
    validate_run,
)


def feedback_spec():
    return {
        "feedback": {
            "fixed_fields": ["validator", "violation_id", "error_type", "focus", "path", "message"],
            "verdict": {"message": "The graph is logically inconsistent."},
            "location": {"message": "The graph is logically inconsistent."},
            "explanation": {
                "message_by_domain": {
                    "movie": "The focus entity belongs to disjoint Movie classes.",
                    "music": "The focus entity belongs to disjoint Music classes.",
                }
            },
        }
    }


def make_row(case_id="case-1", condition="verdict", domain="movie", status="consistent", index=1):
    settings = feedback_spec()["feedback"]
    feedback = {
        "validator": "owl_consistency",
        "violation_id": f"owl:inconsistent:{case_id}",
        "error_type": None,
        "focus": None if condition == "verdict" else "Focus Entity",
        "path": None,
        "message": (
            settings["explanation"]["message_by_domain"][domain]
            if condition == "explanation"
            else settings[condition]["message"]
        ),
    }
    prompt = "Source and injected graph\n\nValidation feedback:\n" + json.dumps(
        [feedback], indent=2, sort_keys=True
    ) + "\n\nRepaired graph:\n"
    failure = status == "failure"
    removed = status in ("consistent", "residual")
    owl_consistent = status == "consistent"
    reference = status == "consistent" and case_id.endswith("reference")
    post = None
    if not failure:
        post = {
            "controlled_target_removed": removed,
            "owl_consistent": owl_consistent,
            "owl_inconsistent_after_target_removal": removed and not owl_consistent,
            "edit": {"symmetric_difference_from_injected": 1 if removed else 0},
            "reference": {
                "reference_recovery": reference,
                "collateral_symmetric_difference": 0 if reference else 1,
                "reference_symmetric_difference": 0 if reference else 2,
            },
            "new_raw_shacl_violation_ids": ["shacl:new"] if status == "residual" else [],
            "new_grounding_violation_ids": [] if reference else ["grounding:new"],
        }
    outcome = {
        "controlled_target_removed": removed,
        "owl_consistent": None if failure else owl_consistent,
        "reference_recovery": reference,
        "collateral_edit": None if failure else not reference,
        "new_raw_shacl_findings": None if failure else status == "residual",
        "new_grounding_findings": None if failure else not reference,
        "owl_inconsistent_after_target_removal": None if failure else removed and not owl_consistent,
        "output_failure": "unparseable_output" if failure else None,
        "edit_distance_from_injected": None if failure else 1 if removed else 0,
        "edit_distance_from_clean_reference": None if failure else 0 if reference else 2,
    }
    return {
        "id": case_id,
        "case_index": index,
        "execution_index": index,
        "domain": domain,
        "condition": condition,
        "initial_measurement": {"owl_consistent": False, "triples": [["source", "relation", "target"]]},
        "post_repair_measurement": post,
        "outcome": outcome,
        "repair": {
            "feedback": feedback,
            "rendered_prompt": prompt,
            "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "parse": {"ok": not failure, "failure": "unparseable_output" if failure else None},
        },
    }


def complete_run():
    rows, schedule, case_ids = [], [], []
    for case_index in range(10):
        domain = "movie" if case_index < 5 else "music"
        case_id = f"{domain}-{case_index}"
        case_ids.append(case_id)
        for condition in CONDITIONS:
            row = make_row(case_id, condition, domain, index=case_index + 1)
            row["execution_index"] = len(rows) + 1
            rows.append(row)
            schedule.append(
                {key: row[key] for key in ("execution_index", "case_index", "id", "domain", "condition")}
            )
    return rows, {"case_ids": case_ids, "execution_schedule": schedule}


def test_rate_keeps_explicit_denominator():
    assert rate(6, 8) == {"numerator": 6, "denominator": 8, "estimate": 0.75}
    assert rate(0, 0)["estimate"] is None


def test_usable_outputs_define_graph_dependent_denominators():
    rows = [make_row(status="consistent"), make_row("case-2", status="failure")]
    result = summarize_rows(rows)
    assert result["rates"]["controlled_target_removed"]["denominator"] == 2
    assert result["rates"]["owl_consistent"]["denominator"] == 1
    assert result["rates"]["collateral_edit"]["denominator"] == 1
    assert result["rates"]["output_failures"] == rate(1, 2)


def test_output_failure_never_becomes_an_inconsistent_graph():
    row = make_row(status="failure")
    validate_outcome(row)
    assert all(row["outcome"][key] is None for key in GRAPH_FIELDS)
    row["outcome"]["owl_consistent"] = False
    with pytest.raises(RuntimeError, match="Outcomes that depend on a parsed graph"):
        validate_outcome(row)


def test_target_removal_and_remaining_owl_inconsistency_stay_separate():
    row = make_row(status="residual")
    validate_outcome(row)
    result = summarize_rows([row])
    assert result["counts"]["controlled_target_removed"] == 1
    assert result["counts"]["owl_consistent"] == 0
    assert result["counts"]["owl_inconsistent_after_target_removal"] == 1


def test_validator_measurements_are_checked_without_rerunning_validators():
    row = make_row(status="consistent")
    row["outcome"]["new_grounding_findings"] = False
    with pytest.raises(RuntimeError, match="New grounding findings"):
        validate_outcome(row)


def test_feedback_prompt_schema_and_hash_are_verified():
    row = make_row(condition="explanation")
    prefix = validate_feedback_and_prompt(row, feedback_spec())
    assert prefix == "Source and injected graph\n\n"
    row["repair"]["feedback"]["path"] = "leaked-path"
    with pytest.raises(RuntimeError, match="fixed feedback field"):
        validate_feedback_and_prompt(row, feedback_spec())


def test_verdict_condition_must_not_reveal_focus_entity():
    row = make_row(condition="verdict")
    row["repair"]["feedback"]["focus"] = "Focus Entity"
    with pytest.raises(RuntimeError, match="Verdict only"):
        validate_feedback_and_prompt(row, feedback_spec())


def test_paired_comparison_counts_discordance_by_case():
    grouped = {
        "case-1": {
            "explanation": make_row("case-1", "explanation", status="consistent"),
            "verdict": make_row("case-1", "verdict", status="failure"),
        },
        "case-2": {
            "explanation": make_row("case-2", "explanation", status="failure"),
            "verdict": make_row("case-2", "verdict", status="consistent"),
        },
        "case-3": {
            "explanation": make_row("case-3", "explanation", status="consistent"),
            "verdict": make_row("case-3", "verdict", status="consistent"),
        },
    }
    result = paired_target_comparison(grouped, "explanation", "verdict")
    assert result["n_paired_cases"] == 3
    assert (result["left_only"], result["right_only"], result["same"]) == (1, 1, 1)


def test_validate_run_accepts_ten_complete_paired_cases(monkeypatch):
    rows, metadata = complete_run()
    monkeypatch.setattr("src.analyze_owl_feedback_framing.validate_metadata", lambda *args: None)
    grouped = validate_run(rows, metadata, {}, feedback_spec())
    assert len(grouped) == 10
    assert set(grouped["movie-0"]) == set(CONDITIONS)


def test_validate_run_rejects_prompt_changes_outside_feedback(monkeypatch):
    rows, metadata = complete_run()
    monkeypatch.setattr("src.analyze_owl_feedback_framing.validate_metadata", lambda *args: None)
    row = rows[1]
    row["repair"]["rendered_prompt"] = "Leaked scaffold\n" + row["repair"]["rendered_prompt"]
    row["repair"]["rendered_prompt_sha256"] = hashlib.sha256(
        row["repair"]["rendered_prompt"].encode()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="outside the OWL feedback block"):
        validate_run(rows, metadata, {}, feedback_spec())


def test_validate_run_rejects_changes_to_paired_injected_state(monkeypatch):
    rows, metadata = complete_run()
    monkeypatch.setattr("src.analyze_owl_feedback_framing.validate_metadata", lambda *args: None)
    rows[1]["initial_measurement"] = copy.deepcopy(rows[1]["initial_measurement"])
    rows[1]["initial_measurement"]["triples"].append(["leaked", "relation", "value"])
    with pytest.raises(RuntimeError, match="same injected state"):
        validate_run(rows, metadata, {}, feedback_spec())
