import json

import pytest

import src.run_owl_feedback_framing as runner
from src.controlled_cases import (
    ControlledCase,
    PrimaryModification,
    Statement,
)
from src.run_owl_feedback_framing import (
    CONDITIONS,
    controlled_target_removed,
    edit_metrics,
    execution_schedule,
    feedback_item,
    finalize_row,
    render_condition_prompt,
    run_experiment,
    selected_disjointness_cases,
    write_results,
)


def spec():
    from src.run_owl_feedback_framing import SPEC_PATH

    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def fake_context(domain="movie", case_id=None):
    if case_id is None:
        case_id = f"fake_{domain}"

    if domain == "movie":
        clean_content = (
            Statement("Film", "director", "Person"),
        )
        injected_statement = Statement(
            "Film", "production_company", "Person"
        )
    else:
        clean_content = (
            Statement("Person A", "composer", "Song"),
            Statement("Person B", "composer", "Song"),
        )
        injected_statement = Statement(
            "Person A", "performer", "Person B"
        )

    clean = ControlledCase(
        case_id=case_id,
        domain=domain,
        source_text="A source sentence.",
        content=clean_content,
    )
    injected = ControlledCase(
        case_id=clean.case_id,
        domain=domain,
        source_text=clean.source_text,
        content=clean_content + (injected_statement,),
        primary_modification=PrimaryModification(
            error_type="disjointness_violation",
            operation="add",
            details={
                "subject": injected_statement.subject,
                "predicate": injected_statement.predicate,
                "object": injected_statement.object,
            },
        ),
    )
    return {
        "selected": {
            "id": clean.case_id,
            "domain": domain,
            "error_type": "disjointness",
        },
        "clean": clean,
        "injected": injected,
        "case_shapes": None,
        "owl_context": None,
    }


def test_selection_contains_exactly_ten_paired_disjointness_cases():
    rows = selected_disjointness_cases(spec())

    assert len(rows) == 10
    assert len({row["id"] for row in rows}) == 10
    assert sum(row["domain"] == "movie" for row in rows) == 5
    assert sum(row["domain"] == "music" for row in rows) == 5


def test_execution_schedule_is_complete_unique_and_rotated():
    case_rows = [
        {"id": f"c{i}", "domain": "movie"}
        for i in range(10)
    ]
    rows = execution_schedule(case_rows)

    assert len(rows) == 30
    assert len({(row["id"], row["condition"]) for row in rows}) == 30
    assert [row["condition"] for row in rows[:3]] == [
        "verdict",
        "location",
        "explanation",
    ]
    assert [row["condition"] for row in rows[3:6]] == [
        "location",
        "explanation",
        "verdict",
    ]
    assert [row["condition"] for row in rows[6:9]] == [
        "explanation",
        "verdict",
        "location",
    ]
    assert {condition: sum(
        row["condition"] == condition for row in rows
    ) for condition in CONDITIONS} == {
        "verdict": 10,
        "location": 10,
        "explanation": 10,
    }


@pytest.mark.parametrize("domain", ["movie", "music"])
def test_feedback_schema_is_fixed_and_information_is_nested(domain):
    context = fake_context(domain)
    rows = {
        condition: feedback_item(context, condition, spec())
        for condition in CONDITIONS
    }

    assert list(rows["verdict"]) == list(rows["location"])
    assert list(rows["location"]) == list(rows["explanation"])
    assert rows["verdict"]["focus"] is None
    assert rows["location"]["focus"] is not None
    assert rows["explanation"]["focus"] == rows["location"]["focus"]
    assert rows["location"]["message"] == rows["verdict"]["message"]
    assert rows["explanation"]["message"] != rows["location"]["message"]
    assert rows["verdict"]["error_type"] is None
    assert rows["verdict"]["path"] is None


def test_prompts_share_inputs_and_never_show_condition_label():
    context = fake_context("movie")
    template = (
        "S={source_text}\nR={allowed_relations}\n"
        "G={current_graph}\nF={feedback}"
    )
    prompts = {}

    for condition in CONDITIONS:
        prompt, feedback = render_condition_prompt(
            context,
            condition,
            spec(),
            template,
        )
        prompts[condition] = prompt
        assert f'"violation_id": "{feedback["violation_id"]}"' in prompt
        assert f"condition: {condition}" not in prompt.lower()
        assert "raw_shacl" not in prompt
        assert "grounding_v3" not in prompt

    for prompt in prompts.values():
        assert "A source sentence." in prompt
        assert "director(Film, Person)" in prompt
        assert "production_company(Film, Person)" in prompt


def test_target_removal_is_separate_from_arbitrary_graph_change():
    context = fake_context("movie")
    clean = [
        ["Film", "director", "Person"],
    ]
    keeps_target_but_deletes_clean = [
        ["Film", "production_company", "Person"],
    ]

    assert controlled_target_removed(context, clean) is True
    assert controlled_target_removed(
        context,
        keeps_target_but_deletes_clean,
    ) is False


def test_edit_metrics_count_changes_from_the_same_injected_graph():
    metrics = edit_metrics(
        (("A", "p", "B"), ("A", "q", "C")),
        (("A", "p", "B"), ("A", "r", "D")),
    )

    assert metrics["symmetric_difference_from_injected"] == 2
    assert metrics["removed_from_injected"] == [["A", "q", "C"]]
    assert metrics["added_to_injected"] == [["A", "r", "D"]]
    assert metrics["graph_size_delta"] == 0


def test_output_failure_has_no_fabricated_graph_measurements():
    row = {
        "repair": {
            "parse": {
                "ok": False,
                "failure": "unparseable_output",
                "triples": [],
                "details": None,
            }
        }
    }
    finalized = finalize_row(
        row,
        fake_context("movie"),
        {
            "grounding_unsupported_ids": [],
            "raw_shacl_violation_ids": [],
        },
        {},
    )

    assert finalized["post_repair_measurement"] is None
    assert finalized["outcome"]["controlled_target_removed"] is False
    assert finalized["outcome"]["reference_recovery"] is False
    assert finalized["outcome"]["owl_consistent"] is None
    assert finalized["outcome"]["collateral_edit"] is None
    assert finalized["outcome"]["output_failure"] == "unparseable_output"


def test_experiment_makes_exactly_one_independent_generation_per_arm(
    monkeypatch,
):
    case_rows = [
        {
            "id": f"case_{index}",
            "domain": "movie" if index < 5 else "music",
        }
        for index in range(10)
    ]
    contexts = {
        row["id"]: fake_context(row["domain"], row["id"])
        for row in case_rows
    }
    calls = []

    monkeypatch.setattr(
        runner,
        "controlled_context",
        lambda case_id: contexts[case_id],
    )

    def cache_for_case(_):
        context = contexts[_["id"]]
        cache = {}

        for case in (context["clean"], context["injected"]):
            for statement in case.content:
                triple = (
                    statement.subject,
                    statement.predicate,
                    statement.object,
                )
                cache[triple] = {
                    "triple": list(triple),
                    "verdict": "SUPPORTED",
                    "reason": "fixture",
                    "source": "fixture",
                }

        return cache, set()

    monkeypatch.setattr(runner, "grounding_cache_for_case", cache_for_case)
    monkeypatch.setattr(
        runner,
        "initial_measurement",
        lambda context, cache: {
            "triples": [
                [row.subject, row.predicate, row.object]
                for row in context["injected"].content
            ],
            "raw_shacl_violation_ids": ["initial:shacl"],
            "grounding_unsupported_ids": [],
        },
    )

    def generate_fixture(context, condition, _spec, _template):
        calls.append((context["selected"]["id"], condition))
        triples = [
            [row.subject, row.predicate, row.object]
            for row in context["clean"].content
        ]
        return {
            "feedback": feedback_item(context, condition, spec()),
            "rendered_prompt_sha256": "fixture",
            "rendered_prompt": "fixture",
            "raw_response": "fixture",
            "model": "fixture",
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
            "total_duration_ns": 1,
            "parse": {
                "ok": True,
                "failure": None,
                "triples": triples,
                "details": None,
            },
        }

    monkeypatch.setattr(runner, "generate_one", generate_fixture)

    def symbolic_fixture(context, triples):
        return {
            "triples": triples,
            "raw_shacl": {"conforms": True, "violations": []},
            "raw_shacl_violation_ids": [],
            "owl_consistent": True,
            "reference": {
                "reference_recovery": True,
                "reference_symmetric_difference": 0,
                "collateral_symmetric_difference": 0,
            },
            "edit": edit_metrics(
                [
                    [row.subject, row.predicate, row.object]
                    for row in context["injected"].content
                ],
                triples,
            ),
            "controlled_target_removed": True,
            "owl_inconsistent_after_target_removal": False,
        }

    monkeypatch.setattr(
        runner,
        "symbolic_post_measurement",
        symbolic_fixture,
    )

    def judge_fixture(_source, triples, cache):
        for triple in triples:
            if triple not in cache:
                cache[triple] = {
                    "triple": list(triple),
                    "verdict": "SUPPORTED",
                    "reason": "fixture",
                    "source": "repair_round",
                }
        return [cache[triple] for triple in triples]

    monkeypatch.setattr(runner, "judge_current_grounding", judge_fixture)
    frozen = {row["id"]: {"id": row["id"]} for row in case_rows}
    rows, novel_calls = run_experiment(spec(), case_rows, frozen)

    assert len(rows) == 30
    assert len(calls) == 30
    assert calls == [
        (row["id"], row["condition"])
        for row in execution_schedule(case_rows)
    ]
    assert len({(row["id"], row["condition"]) for row in rows}) == 30
    assert all(row["outcome"]["controlled_target_removed"] for row in rows)
    assert novel_calls == 0


def test_writer_refuses_to_overwrite_results(tmp_path):
    output = tmp_path / "rq3.jsonl"
    write_results(output, [{"id": "one"}], {"meta": True})

    with pytest.raises(RuntimeError, match="never overwritten"):
        write_results(output, [{"id": "two"}], {"meta": True})
