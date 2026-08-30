from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src import grounding_judge
from src.build_sosa_ssn_confirmatory_cases import sha256_file
from src.sosa_ssn_confirmatory_runtime import (
    ROOT,
    RUNNER_SPEC_PATH,
    append_jsonl,
    atomic_write_json,
    case_content,
    expected_target_grounding,
    git_head,
    load_cases,
    load_complete_jsonl,
    load_runner_spec,
    normalized_triples,
    primary_added,
    repository_path,
    require_accepted_audit_gate,
    verify_existing_metadata,
    verify_model_metadata,
    validate_resume_prefix,
)


def state_summary(values, cache) -> dict:
    judgments = [cache[value] for value in values]
    unsupported = sum(row["verdict"] == "UNSUPPORTED" for row in judgments)
    return {
        "grounding_error": bool(unsupported),
        "unsupported_count": unsupported,
        "triple_count": len(judgments),
        "judgments": judgments,
    }


def judge_case(case: dict, spec: dict, template: str) -> dict:
    clean = case_content(case, "clean")
    injected = case_content(case, "injected")
    union = normalized_triples([*clean, *injected])
    cache = {}
    for index, value in enumerate(union, start=1):
        print(f"    assertion {index}/{len(union)}")
        judgment = grounding_judge.judge_triple(
            case["source_text"], list(value), template=template
        )
        cache[value] = {**judgment, "source": "confirmatory_frozen_union"}

    target = primary_added(case)
    target_judgments = [cache[value] for value in target]
    observed_target = any(
        row["verdict"] == "UNSUPPORTED" for row in target_judgments
    )
    expected_target = expected_target_grounding(case, spec)
    return {
        "case_id": case["case_id"],
        "condition": case["condition"],
        "source_family": case["source_family"],
        "source_text_sha256": case["source_text_sha256"],
        "judgments": [cache[value] for value in union],
        "clean": state_summary(clean, cache),
        "injected": state_summary(injected, cache),
        "target": {
            "triples": [list(value) for value in target],
            "triple_count": len(target),
            "expected_grounding_error": expected_target,
            "observed_grounding_error": observed_target,
            "matches_expected": observed_target == expected_target,
            "judgments": target_judgments,
        },
    }


def preflight_only() -> None:
    spec = load_runner_spec()
    cases = load_cases(spec)
    print(f"grounding preflight cases: {len(cases)}")
    print("audit gate and model identity are checked only for experimental execution.")
    print("No grounding assessment or repair generation was executed.")


def run() -> None:
    spec = load_runner_spec()
    gate = require_accepted_audit_gate()
    cases = load_cases(spec)
    repair_model, grounding_model = verify_model_metadata(spec)
    del repair_model

    output_path = repository_path(spec["outputs"]["grounding_results"])
    metadata_path = repository_path(spec["outputs"]["grounding_metadata"])
    head = git_head()
    spec_hash = sha256_file(RUNNER_SPEC_PATH)
    existing_metadata = verify_existing_metadata(metadata_path, spec_hash, head)
    existing = load_complete_jsonl(output_path, {case["case_id"] for case in cases})
    validate_resume_prefix(existing, cases, output_path)
    completed = {row["case_id"] for row in existing}

    if output_path.exists() and existing_metadata is None:
        raise RuntimeError("grounding result exists without matching metadata")
    if metadata_path.exists() and not output_path.exists():
        raise RuntimeError("grounding metadata exists without its result file")

    if existing_metadata is None:
        metadata = {
            "version": 1,
            "status": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": head,
            "audited_commit": gate["audited_commit"],
            "runner_spec": str(RUNNER_SPEC_PATH.relative_to(ROOT)),
            "runner_spec_sha256": spec_hash,
            "model": grounding_model,
            "prompt": spec["inputs"]["grounding_prompt"],
            "case_count": len(cases),
            "completed_case_count": len(completed),
            "resume_policy": "complete case rows only",
            "shared_assertion_policy": "judge once per case union",
        }
    else:
        metadata = existing_metadata

    template = grounding_judge.load_prompt()
    remaining = [case for case in cases if case["case_id"] not in completed]
    for index, case in enumerate(remaining, start=1):
        print(
            f"[{index}/{len(remaining)} remaining] {case['case_id']} "
            f"({case['condition']})"
        )
        row = judge_case(case, spec, template)
        append_jsonl(output_path, row)
        completed.add(case["case_id"])
        metadata["completed_case_count"] = len(completed)
        metadata["status"] = "complete" if len(completed) == len(cases) else "running"
        metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(metadata_path, metadata)

    metadata["status"] = "complete"
    metadata["completed_case_count"] = len(completed)
    metadata["results_sha256"] = sha256_file(output_path)
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(metadata_path, metadata)
    print(f"grounding cases complete: {len(completed)}/{len(cases)}")
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
