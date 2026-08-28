#!/usr/bin/env python3
"""Offline fail-fast preflight for the SOSA and SSN confirmatory runners."""

from __future__ import annotations

from collections import Counter

from src import extract_text2kg, grounding_judge
from src.build_sosa_ssn_confirmatory_cases import read_json, read_jsonl, triple_key
from src.sosa_ssn_confirmatory_runtime import (
    AUDIT_GATE_PATH,
    case_content,
    expected_target_grounding,
    load_cases,
    load_runner_spec,
    parse_repair_response,
    primary_added,
    render_repair_prompt,
    repository_path,
    rows_for_repair,
)


def run_preflight() -> dict:
    spec = load_runner_spec()
    cases = load_cases(spec)

    repair_model = spec["models"]["repair"]
    if repair_model["name"] != extract_text2kg.MODEL:
        raise RuntimeError("repair model differs from preliminary extraction model")
    if repair_model["options"] != extract_text2kg.OPTIONS:
        raise RuntimeError("repair options differ from preliminary extraction options")

    grounding_model = spec["models"]["grounding"]
    if grounding_model["name"] != grounding_judge.MODEL:
        raise RuntimeError("grounding model differs from frozen v3 assessor")
    if grounding_model["digest"] != grounding_judge.EXPECTED_DIGEST:
        raise RuntimeError("grounding digest differs from frozen v3 assessor")
    if grounding_model["options"] != grounding_judge.OPTIONS:
        raise RuntimeError("grounding options differ from frozen v3 assessor")
    if grounding_model["judge_version"] != grounding_judge.JUDGE_VERSION:
        raise RuntimeError("grounding judge version differs from frozen v3 assessor")

    for case in cases:
        for state, source_key in (
            ("clean", "clean_content_triples"),
            ("injected", "injected_content_triples"),
        ):
            rebuilt = rows_for_repair(case, case_content(case, state))
            expected = sorted(case[source_key], key=triple_key)
            if sorted(rebuilt, key=triple_key) != expected:
                raise RuntimeError(
                    f"object-kind reconstruction drift: {case['case_id']} {state}"
                )

    symbolic_results = read_jsonl(
        repository_path(spec["inputs"]["symbolic_results"]["path"])
    )
    if len(symbolic_results) != len(cases):
        raise RuntimeError("symbolic preflight denominator differs from runner cases")
    if any(row.get("mismatches") for row in symbolic_results):
        raise RuntimeError("symbolic preflight contains a mismatch")

    expected_target = Counter(
        expected_target_grounding(case, spec) for case in cases
    )
    target_sizes = Counter(bool(primary_added(case)) for case in cases)
    if expected_target != {False: 30, True: 150}:
        raise RuntimeError("grounding target expectation denominators changed")
    if target_sizes != {False: 30, True: 150}:
        raise RuntimeError("grounding target assertion denominators changed")

    prompt_template = repository_path(
        spec["inputs"]["repair_prompt"]["path"]
    ).read_text(encoding="utf-8")
    fixture = cases[0]
    fixture_triples = case_content(fixture, "injected")
    rendered = render_repair_prompt(
        prompt_template,
        fixture,
        fixture_triples,
        [
            {
                "validator": "raw_shacl",
                "violation_id": "fixture",
                "error_type": fixture["condition"],
                "focus": "fixture",
                "path": "observedProperty",
                "message": "Fixture only.",
            }
        ],
    )
    parsed = parse_repair_response(
        "\n".join(
            f"{predicate}({subject}, {obj})"
            for subject, predicate, obj in fixture_triples
        ),
        fixture["allowed_relations"],
    )
    if not parsed["ok"] or len(parsed["triples"]) != len(fixture_triples):
        raise RuntimeError("strict complete-graph parser fixture failed")
    if "clean reference" in rendered.lower() or "scaffold" in rendered.lower():
        raise RuntimeError("model-visible prompt leaked a forbidden field")

    gate = read_json(AUDIT_GATE_PATH)
    if gate.get("status") == "pending_pre_run_audit":
        if gate.get("execution_allowed") is not False:
            raise RuntimeError("pending pre-run audit gate is open")
    elif gate.get("status") == "accepted":
        if gate.get("execution_allowed") is not True or gate.get("verdict") != "A":
            raise RuntimeError("accepted pre-run audit gate is incomplete")
        if not isinstance(gate.get("audited_commit"), str) or len(gate["audited_commit"]) != 40:
            raise RuntimeError("accepted pre-run audit gate lacks a full commit SHA")
    else:
        raise RuntimeError("pre-run audit gate has an unsupported status")

    return {
        "cases": len(cases),
        "by_condition": dict(sorted(Counter(case["condition"] for case in cases).items())),
        "grounding_target_expected_positive": expected_target[True],
        "grounding_target_expected_negative": expected_target[False],
        "symbolic_mismatches": 0,
        "audit_gate": gate["status"],
    }


def main() -> None:
    summary = run_preflight()
    print(f"confirmatory cases: {summary['cases']}")
    for condition, count in summary["by_condition"].items():
        print(f"  {condition}: {count}")
    print(
        "grounding target expected positive/negative: "
        f"{summary['grounding_target_expected_positive']}/"
        f"{summary['grounding_target_expected_negative']}"
    )
    print(f"symbolic mismatches: {summary['symbolic_mismatches']}")
    print(f"audit gate: {summary['audit_gate']}")
    print("Confirmatory runner offline preflight passed.")
    print("No model, grounding assessor, validator, reasoner, or repair was executed.")


if __name__ == "__main__":
    main()
