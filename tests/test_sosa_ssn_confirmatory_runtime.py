import json
from collections import Counter

import pytest

from src.build_sosa_ssn_confirmatory_cases import triple_key
from src.preflight_sosa_ssn_confirmatory import run_preflight
from src.sosa_ssn_confirmatory_runtime import (
    AUDIT_GATE_PATH,
    case_content,
    expected_target_grounding,
    infer_object_kind,
    load_cases,
    load_complete_jsonl,
    load_runner_spec,
    parse_repair_response,
    primary_added,
    render_repair_prompt,
    require_accepted_audit_gate,
    rows_for_repair,
    target_resolved,
    validate_resume_prefix,
)


def inputs():
    spec = load_runner_spec()
    cases = load_cases(spec)
    return spec, cases


def test_runner_inputs_are_hash_locked_and_loadable():
    spec, cases = inputs()
    assert spec["status"] == "prepared_for_pre_run_audit"
    assert len(cases) == 180


def test_runner_sample_is_180_unique_cases_with_thirty_per_condition():
    spec, cases = inputs()
    assert len({case["case_id"] for case in cases}) == 180
    assert len({case["source_unit_id"] for case in cases}) == 180
    assert Counter(case["condition"] for case in cases) == {
        condition: 30 for condition in spec["sample"]["conditions"]
    }


def test_frozen_models_and_options_match_preliminary_components():
    spec, _cases = inputs()
    assert spec["models"]["repair"]["digest"] == (
        "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"
    )
    assert spec["models"]["grounding"]["digest"] == (
        "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"
    )
    assert spec["models"]["grounding"]["judge_version"] == "v3"
    assert spec["models"]["grounding"]["treated_as_ground_truth"] is False


def test_grounding_target_expectations_follow_grouped_injection():
    spec, cases = inputs()
    by_condition = {
        condition: [case for case in cases if case["condition"] == condition]
        for condition in spec["sample"]["conditions"]
    }
    for case in by_condition["cardinality"]:
        assert primary_added(case) == ()
        assert expected_target_grounding(case, spec) is False
    for condition in set(by_condition) - {"cardinality"}:
        for case in by_condition[condition]:
            assert primary_added(case)
            assert expected_target_grounding(case, spec) is True


def test_all_fixed_graphs_roundtrip_through_object_kind_reconstruction():
    _spec, cases = inputs()
    for case in cases:
        for state, key in (
            ("clean", "clean_content_triples"),
            ("injected", "injected_content_triples"),
        ):
            rebuilt = rows_for_repair(case, case_content(case, state))
            assert sorted(rebuilt, key=triple_key) == sorted(case[key], key=triple_key)


def test_new_value_object_kind_rules_are_deterministic():
    _spec, cases = inputs()
    case = cases[0]
    assert infer_object_kind(case, ("x", "hasFeatureOfInterest", "y")) == "entity"
    assert infer_object_kind(case, ("x", "hasSimpleResult", "12.5")) == "decimal"
    assert infer_object_kind(case, ("x", "hasSimpleResult", "true")) == "boolean"
    assert infer_object_kind(case, ("x", "resultUnit", "litres")) == "string"
    assert infer_object_kind(case, ("x", "phenomenonTime", "2026-01-01")) == "date"
    assert infer_object_kind(
        case, ("x", "phenomenonTime", "2026-01-01T10:00:00+00:00")
    ) == "datetime"


def test_invalid_class_and_time_are_not_silently_corrected():
    _spec, cases = inputs()
    case = cases[0]
    with pytest.raises(ValueError, match="unknown SOSA/SSN profile class"):
        infer_object_kind(case, ("x", "type", "InventedClass"))
    with pytest.raises(ValueError, match="invalid ISO time"):
        infer_object_kind(case, ("x", "resultTime", "tomorrow"))


def test_strict_parser_accepts_complete_graph_and_removes_duplicates():
    parsed = parse_repair_response(
        "type(obs, Observation)\n"
        "observedProperty(obs, temperature)\n"
        "observedProperty(obs, temperature)",
        {"type", "observedProperty"},
    )
    assert parsed["ok"] is True
    assert parsed["triples"] == [
        ["obs", "type", "Observation"],
        ["obs", "observedProperty", "temperature"],
    ]


@pytest.mark.parametrize(
    "text,failure",
    [
        ("", "empty_output"),
        ("Here is the repaired graph:\ntype(obs, Observation)", "unparseable_output"),
        ("invented(obs, value)", "relation_outside_allowed_set"),
        ("type(obs, Observation) because it is correct", "unparseable_output"),
    ],
)
def test_strict_parser_records_output_failures(text, failure):
    parsed = parse_repair_response(text, {"type"})
    assert parsed["ok"] is False
    assert parsed["failure"] == failure


def test_rendered_prompt_excludes_clean_reference_and_scaffold():
    spec, cases = inputs()
    case = cases[0]
    template = open(spec["inputs"]["repair_prompt"]["path"], encoding="utf-8").read()
    rendered = render_repair_prompt(
        template,
        case,
        case_content(case, "injected"),
        [
            {
                "validator": "raw_shacl",
                "violation_id": "v1",
                "error_type": case["condition"],
                "focus": "obs",
                "path": "observedProperty",
                "message": "Missing value.",
            }
        ],
    )
    assert case["source_text"] in rendered
    assert "\"violation_id\": \"v1\"" in rendered
    assert "clean reference" not in rendered.lower()
    assert "scaffold" not in rendered.lower()


def test_audit_gate_has_a_valid_pending_or_accepted_state():
    gate = json.loads(AUDIT_GATE_PATH.read_text(encoding="utf-8"))
    assert set(gate["allowed_post_audit_changes"]) == {
        "experiments/sosa_ssn_confirmatory_audit_gate.json",
        "docs/audit_register.md",
    }
    if gate["status"] == "pending_pre_run_audit":
        assert gate["execution_allowed"] is False
        with pytest.raises(RuntimeError, match="pending accepted pre-run audit"):
            require_accepted_audit_gate()
    else:
        assert gate["status"] == "accepted"
        assert gate["execution_allowed"] is True
        assert gate["verdict"] == "A"
        assert len(gate["audited_commit"]) == 40


def test_target_resolution_tracks_original_symbolic_violation_identity():
    spec, cases = inputs()
    case = next(case for case in cases if case["condition"] == "temporal")
    symbolic_spec = json.loads(
        open(spec["inputs"]["symbolic_spec"]["path"], encoding="utf-8").read()
    )
    original = {
        "shacl": {
            "violations": [
                {
                    "violation_id": "original-target",
                    "constraint_component": "http://www.w3.org/ns/shacl#SPARQLConstraintComponent",
                }
            ]
        }
    }
    assert target_resolved(
        case, case_content(case, "injected"), original, symbolic_spec
    ) is False
    assert target_resolved(
        case,
        case_content(case, "injected"),
        original,
        symbolic_spec,
        target_violation_ids={"original-target"},
    ) is False
    replacement_violation = {
        "shacl": {
            "violations": [
                {
                    "violation_id": "new-violation",
                    "constraint_component": "http://www.w3.org/ns/shacl#SPARQLConstraintComponent",
                }
            ]
        }
    }
    assert target_resolved(
        case,
        case_content(case, "injected"),
        replacement_violation,
        symbolic_spec,
        target_violation_ids={"original-target"},
    ) is True


def test_grounding_target_resolution_requires_controlled_added_value_to_leave():
    spec, cases = inputs()
    case = next(case for case in cases if case["condition"] == "grounding")
    symbolic_spec = json.loads(
        open(spec["inputs"]["symbolic_spec"]["path"], encoding="utf-8").read()
    )
    empty_symbolic = {"shacl": {"violations": []}}
    assert target_resolved(
        case, case_content(case, "injected"), empty_symbolic, symbolic_spec
    ) is False
    assert target_resolved(
        case, case_content(case, "clean"), empty_symbolic, symbolic_spec
    ) is True


def test_resume_loader_rejects_duplicate_case_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"case_id": "a"}) + "\n" + json.dumps({"case_id": "a"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate completed case"):
        load_complete_jsonl(path, {"a"})


def test_resume_rows_must_be_the_fixed_case_order_prefix(tmp_path):
    source = tmp_path / "rows.jsonl"
    cases = [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
    validate_resume_prefix([{"case_id": "a"}, {"case_id": "b"}], cases, source)
    with pytest.raises(RuntimeError, match="fixed case-order prefix"):
        validate_resume_prefix([{"case_id": "a"}, {"case_id": "c"}], cases, source)


def test_offline_preflight_passes_without_opening_audit_gate():
    summary = run_preflight()
    assert summary["cases"] == 180
    assert summary["grounding_target_expected_positive"] == 150
    assert summary["grounding_target_expected_negative"] == 30
    assert summary["symbolic_mismatches"] == 0
    assert summary["audit_gate"] in {"pending_pre_run_audit", "accepted"}
