from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src import extract_text2kg, grounding_judge
from src.build_sosa_ssn_confirmatory_cases import sha256_file
from src.repair_engine import reference_metrics, stop_reason_after_repair, trajectory_state
from src.sosa_ssn_confirmatory_runtime import (
    ROOT,
    RUNNER_SPEC_PATH,
    append_jsonl,
    atomic_write_json,
    case_content,
    clean_unsupported,
    git_head,
    initial_grounding_feedback,
    judgment_cache,
    later_grounding_feedback,
    load_cases,
    load_complete_jsonl,
    load_runner_spec,
    load_symbolic_context,
    normalized_triples,
    owl_feedback,
    parse_repair_response,
    render_repair_prompt,
    repository_path,
    require_accepted_audit_gate,
    sha256_bytes,
    shacl_feedback,
    target_resolved,
    validate_parsed_kinds,
    validate_symbolic_state,
    validate_resume_prefix,
    verify_existing_metadata,
    verify_model_metadata,
)


def load_grounding_results(cases: list[dict], spec: dict) -> tuple[dict, dict]:
    result_path = repository_path(spec["outputs"]["grounding_results"])
    metadata_path = repository_path(spec["outputs"]["grounding_metadata"])
    if not result_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("complete confirmatory grounding output is required before repair")
    metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError("confirmatory grounding metadata is not complete")
    if metadata.get("results_sha256") != sha256_file(result_path):
        raise RuntimeError("confirmatory grounding result hash mismatch")
    rows = load_complete_jsonl(result_path, {case["case_id"] for case in cases})
    if len(rows) != len(cases):
        raise RuntimeError(f"expected {len(cases)} grounding rows, found {len(rows)}")
    by_id = {row["case_id"]: row for row in rows}
    for case in cases:
        row = by_id[case["case_id"]]
        if row.get("condition") != case["condition"]:
            raise RuntimeError(f"grounding condition mismatch: {case['case_id']}")
        if row.get("source_text_sha256") != case["source_text_sha256"]:
            raise RuntimeError(f"grounding source hash mismatch: {case['case_id']}")
        if row.get("target", {}).get("expected_grounding_error") != spec["grounding"][
            "expected_target_error"
        ][case["condition"]]:
            raise RuntimeError(f"grounding target expectation drift: {case['case_id']}")
        cache = judgment_cache(row)
        required = set(case_content(case, "clean")) | set(case_content(case, "injected"))
        if set(cache) != required:
            raise RuntimeError(f"grounding cache does not cover the fixed case union: {case['case_id']}")
    return by_id, metadata


def generate_repair(prompt: str, spec: dict) -> dict:
    model = spec["models"]["repair"]
    response = extract_text2kg.api(
        "/api/generate",
        {
            "model": model["name"],
            "prompt": prompt,
            "stream": False,
            "options": model["options"],
        },
    )
    return {
        "raw_response": response.get("response", ""),
        "model": response.get("model"),
        "done_reason": response.get("done_reason"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "total_duration_ns": response.get("total_duration"),
    }


def evaluate_state(
    case: dict,
    triples,
    cache: dict,
    baseline_unsupported: set,
    symbolic_spec: dict,
    profile,
    ontology,
    grounding_template: str,
    target_violation_ids: set[str] | None = None,
    *,
    initial: bool,
) -> dict:
    current = normalized_triples(triples)
    symbolic = validate_symbolic_state(case, current, profile, ontology)
    feedback = shacl_feedback(case, current, symbolic["shacl"], initial=initial)
    owl = owl_feedback(case, symbolic["owl_consistent"], initial=initial)
    if owl is not None:
        feedback.append(owl)

    if initial:
        missing = [value for value in current if value not in cache]
        if missing:
            raise RuntimeError(f"round 0 grounding cache gap: {case['case_id']} {missing}")
        grounding_judgments = [cache[value] for value in current]
        feedback.extend(initial_grounding_feedback(case, cache))
    else:
        grounding_rows, grounding_judgments = later_grounding_feedback(
            case,
            current,
            cache,
            baseline_unsupported,
            grounding_template,
        )
        feedback.extend(grounding_rows)

    feedback.sort(key=lambda row: (row["validator"], row["violation_id"]))
    clean = case_content(case, "clean")
    injected = case_content(case, "injected")
    return {
        "symbolic": symbolic,
        "grounding": {
            "judgments": grounding_judgments,
            "clean_baseline_unsupported_excluded": [
                list(value) for value in sorted(baseline_unsupported)
            ],
        },
        "actionable_feedback": feedback,
        "target_resolved": target_resolved(
            case,
            current,
            symbolic,
            symbolic_spec,
            target_violation_ids=target_violation_ids,
        ),
        "reference": reference_metrics(clean, injected, current),
    }


def run_case(
    case: dict,
    grounding_row: dict,
    spec: dict,
    symbolic_spec: dict,
    profile,
    ontology,
    repair_template: str,
    grounding_template: str,
) -> dict:
    injected = case_content(case, "injected")
    cache = judgment_cache(grounding_row)
    baseline_unsupported = clean_unsupported(case, cache)
    initial = evaluate_state(
        case,
        injected,
        cache,
        baseline_unsupported,
        symbolic_spec,
        profile,
        ontology,
        grounding_template,
        target_violation_ids=None,
        initial=True,
    )
    initial_feedback = initial["actionable_feedback"]
    initial_ids = {row["violation_id"] for row in initial_feedback}
    allowed_target_components = set(
        symbolic_spec["allowed_injected_shacl_components"][case["condition"]]
    )
    target_violation_ids = {
        row["violation_id"]
        for row in initial["symbolic"]["shacl"]["violations"]
        if row["constraint_component"] in allowed_target_components
    }
    rounds = [
        {
            "round": 0,
            "triples": [list(value) for value in injected],
            "validation": initial,
            "new_violation_ids": [],
        }
    ]
    if not initial_feedback:
        return {
            "case_id": case["case_id"],
            "condition": case["condition"],
            "source_family": case["source_family"],
            "received_initial_feedback": False,
            "initial_feedback_sources": [],
            "rounds": rounds,
            "final": {
                "stop_reason": "no_feedback",
                "repair_rounds": 0,
                "target_resolved": initial["target_resolved"],
                "validated_state": True,
                "reference_recovery": initial["reference"]["reference_recovery"],
                "rounds_to_resolution": None,
                "output_failure": None,
            },
        }

    current = injected
    current_feedback = initial_feedback
    history = []
    first_resolution = None
    stop_reason = None
    output_failure = None
    max_rounds = spec["repair"]["max_rounds"]

    for round_number in range(1, max_rounds + 1):
        prompt = render_repair_prompt(
            repair_template, case, current, current_feedback
        )
        generation = generate_repair(prompt, spec)
        if generation["done_reason"] == "length":
            parsed = {
                "ok": False,
                "failure": "generation_truncated",
                "triples": [],
                "details": None,
            }
        else:
            parsed = parse_repair_response(
                generation["raw_response"], case["allowed_relations"]
            )
            parsed = validate_parsed_kinds(case, parsed)
        repair_record = {
            "rendered_prompt": prompt,
            "rendered_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            **generation,
            "parse": parsed,
        }
        if not parsed["ok"]:
            output_failure = parsed["failure"]
            rounds.append(
                {
                    "round": round_number,
                    "repair": repair_record,
                    "triples": None,
                    "validation": None,
                    "new_violation_ids": [],
                }
            )
            stop_reason = "output_failure"
            break

        current = normalized_triples(parsed["triples"])
        state = evaluate_state(
            case,
            current,
            cache,
            baseline_unsupported,
            symbolic_spec,
            profile,
            ontology,
            grounding_template,
            target_violation_ids=target_violation_ids,
            initial=False,
        )
        current_feedback = state["actionable_feedback"]
        current_ids = {row["violation_id"] for row in current_feedback}
        current_state = trajectory_state(current, current_feedback)
        rounds.append(
            {
                "round": round_number,
                "repair": repair_record,
                "triples": [list(value) for value in current],
                "validation": state,
                "new_violation_ids": sorted(current_ids - initial_ids),
            }
        )
        if state["target_resolved"] and first_resolution is None:
            first_resolution = round_number
        stop_reason = stop_reason_after_repair(
            round_number, current_state, history, current_feedback, max_rounds
        )
        if stop_reason is not None:
            break
        history.append(current_state)

    final_validation = rounds[-1].get("validation")
    if final_validation is None:
        final_target = False
        final_validated = False
        final_reference = False
    else:
        final_target = final_validation["target_resolved"]
        final_validated = not final_validation["actionable_feedback"]
        final_reference = final_validation["reference"]["reference_recovery"]
    return {
        "case_id": case["case_id"],
        "condition": case["condition"],
        "source_family": case["source_family"],
        "received_initial_feedback": True,
        "initial_feedback_sources": sorted(
            {row["validator"] for row in initial_feedback}
        ),
        "rounds": rounds,
        "final": {
            "stop_reason": stop_reason,
            "repair_rounds": len(rounds) - 1,
            "target_resolved": final_target,
            "validated_state": final_validated,
            "reference_recovery": final_reference,
            "rounds_to_resolution": first_resolution,
            "output_failure": output_failure,
        },
    }


def preflight_only() -> None:
    spec = load_runner_spec()
    cases = load_cases(spec)
    print(f"repair preflight cases: {len(cases)}")
    print("audit gate and model identity are checked only for experimental execution.")
    print("No grounding assessment or repair generation was executed.")


def run() -> None:
    spec = load_runner_spec()
    gate = require_accepted_audit_gate()
    cases = load_cases(spec)
    grounding_rows, grounding_metadata = load_grounding_results(cases, spec)
    repair_model, grounding_model = verify_model_metadata(spec)
    symbolic_spec, profile, ontology = load_symbolic_context(spec)

    output_path = repository_path(spec["outputs"]["repair_results"])
    metadata_path = repository_path(spec["outputs"]["repair_metadata"])
    head = git_head()
    spec_hash = sha256_file(RUNNER_SPEC_PATH)
    existing_metadata = verify_existing_metadata(metadata_path, spec_hash, head)
    existing = load_complete_jsonl(output_path, {case["case_id"] for case in cases})
    validate_resume_prefix(existing, cases, output_path)
    completed = {row["case_id"] for row in existing}

    if output_path.exists() and existing_metadata is None:
        raise RuntimeError("repair result exists without matching metadata")
    if existing_metadata is None:
        metadata = {
            "version": 1,
            "status": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": head,
            "audited_commit": gate["audited_commit"],
            "runner_spec": str(RUNNER_SPEC_PATH.relative_to(ROOT)),
            "runner_spec_sha256": spec_hash,
            "repair_model": repair_model,
            "grounding_model": grounding_model,
            "repair_prompt": spec["inputs"]["repair_prompt"],
            "grounding_prompt": spec["inputs"]["grounding_prompt"],
            "grounding_results_sha256": grounding_metadata["results_sha256"],
            "case_count": len(cases),
            "completed_case_count": len(completed),
            "resume_policy": "complete case rows only",
            "invalid_model_output_retry": False,
            "transport_retries": spec["repair"]["transport_retries"],
        }
    else:
        metadata = existing_metadata

    repair_template = repository_path(
        spec["inputs"]["repair_prompt"]["path"]
    ).read_text(encoding="utf-8")
    grounding_template = grounding_judge.load_prompt()
    remaining = [case for case in cases if case["case_id"] not in completed]
    for index, case in enumerate(remaining, start=1):
        print(
            f"[{index}/{len(remaining)} remaining] {case['case_id']} "
            f"({case['condition']})"
        )
        trajectory = run_case(
            case,
            grounding_rows[case["case_id"]],
            spec,
            symbolic_spec,
            profile,
            ontology,
            repair_template,
            grounding_template,
        )
        append_jsonl(output_path, trajectory)
        completed.add(case["case_id"])
        metadata["completed_case_count"] = len(completed)
        metadata["status"] = "complete" if len(completed) == len(cases) else "running"
        metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(metadata_path, metadata)
        final = trajectory["final"]
        print(
            f"  stop={final['stop_reason']} target={final['target_resolved']} "
            f"rounds={final['repair_rounds']}"
        )

    metadata["status"] = "complete"
    metadata["completed_case_count"] = len(completed)
    metadata["results_sha256"] = sha256_file(output_path)
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(metadata_path, metadata)
    print(f"repair cases complete: {len(completed)}/{len(cases)}")
    print(f"results: {output_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight_only()
        return
    run()


if __name__ == "__main__":
    main()
