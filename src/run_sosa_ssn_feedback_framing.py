from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src import grounding_judge
from src.build_sosa_ssn_confirmatory_cases import read_json, read_jsonl, sha256_file
from src.repair_engine import reference_metrics
from src.run_sosa_ssn_confirmatory_repair import generate_repair, load_grounding_results
from src.sosa_ssn_confirmatory_runtime import (
    ROOT,
    append_jsonl,
    atomic_write_json,
    case_content,
    clean_unsupported,
    git_head,
    initial_grounding_feedback,
    judgment_cache,
    later_grounding_feedback,
    load_cases,
    load_runner_spec,
    load_symbolic_context,
    normalized_triples,
    parse_repair_response,
    primary_added,
    render_repair_prompt,
    repository_path,
    sha256_bytes,
    validate_parsed_kinds,
    validate_symbolic_state,
    verify_model_metadata,
)


SPEC_PATH = ROOT / "experiments" / "sosa_ssn_feedback_framing_spec.json"
CONDITIONS = ("verdict", "location", "explanation")
FEEDBACK_FIELDS = (
    "validator",
    "violation_id",
    "error_type",
    "focus",
    "path",
    "message",
)


def verify_hashes(mapping: dict[str, str]) -> None:
    for relative, expected in mapping.items():
        path = repository_path(relative)
        if not path.is_file():
            raise RuntimeError(f"missing locked RQ3 input: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"RQ3 input hash mismatch for {relative}: expected {expected}, got {actual}"
            )


def load_framing_spec() -> tuple[dict, dict]:
    spec = read_json(SPEC_PATH)
    if spec.get("version") != 1 or spec.get("status") != "ready_for_execution":
        raise RuntimeError("unexpected SOSA and SSN feedback framing specification")
    if tuple(spec["design"]["condition_order"]) != CONDITIONS:
        raise RuntimeError("feedback framing condition order changed")
    if spec["sample"]["paired_cases"] != 30:
        raise RuntimeError("RQ3 must use 30 paired cases")
    if spec["sample"]["repair_generations"] != 90:
        raise RuntimeError("RQ3 must contain 90 repair generations")
    if spec["design"]["repair_steps_per_condition"] != 1:
        raise RuntimeError("RQ3 must use one independent repair step per condition")
    if spec["generation"]["invalid_output_retry"] is not False:
        raise RuntimeError("invalid model outputs must not be retried")
    if spec["design"]["post_repair_measurement_visible_to_model"] is not False:
        raise RuntimeError("measurement after repair must remain hidden")
    if tuple(spec["feedback"]["fixed_fields"]) != FEEDBACK_FIELDS:
        raise RuntimeError("feedback schema changed")
    if spec["feedback"]["error_type"] is not None or spec["feedback"]["path"] is not None:
        raise RuntimeError("fixed feedback fields leak condition information")
    verify_hashes(spec["locked_inputs"])

    base = load_runner_spec()
    generation = spec["generation"]
    expected_model = {
        "name": generation["model"],
        "digest": generation["digest"],
        "options": generation["options"],
    }
    actual_model = {
        key: base["models"]["repair"][key]
        for key in ("name", "digest", "options")
    }
    if actual_model != expected_model:
        raise RuntimeError("RQ3 repair model differs from the confirmatory repair model")
    prompt = repository_path(generation["prompt_path"])
    if sha256_file(prompt) != generation["prompt_sha256"]:
        raise RuntimeError("RQ3 repair prompt hash mismatch")
    return spec, base


def load_rq3_cases(base_spec: dict) -> list[dict]:
    rows = [case for case in load_cases(base_spec) if case["condition"] == "disjointness"]
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 disjointness cases, found {len(rows)}")
    if len({case["case_id"] for case in rows}) != 30:
        raise RuntimeError("duplicate RQ3 case id")
    if len({case["source_unit_id"] for case in rows}) != 30:
        raise RuntimeError("RQ3 source unit reused")
    for case in rows:
        added = primary_added(case)
        if len(added) != 1 or added[0][1:] != ("type", "SampleCollection"):
            raise RuntimeError(f"unexpected disjointness target: {case['case_id']}")
        clean = set(case_content(case, "clean"))
        if (added[0][0], "type", "ObservationCollection") not in clean:
            raise RuntimeError(f"clean ObservationCollection type missing: {case['case_id']}")
    return rows


def execution_schedule(cases: list[dict]) -> list[dict]:
    schedule = []
    for case_index, case in enumerate(cases):
        offset = case_index % len(CONDITIONS)
        order = CONDITIONS[offset:] + CONDITIONS[:offset]
        for framing in order:
            schedule.append(
                {
                    "execution_index": len(schedule) + 1,
                    "case_index": case_index + 1,
                    "case_id": case["case_id"],
                    "source_family": case["source_family"],
                    "framing": framing,
                }
            )
    return schedule


def feedback_item(case: dict, framing: str, spec: dict) -> dict:
    if framing not in CONDITIONS:
        raise ValueError(f"unknown feedback framing: {framing}")
    focus = primary_added(case)[0][0]
    settings = spec["feedback"]
    opaque_id = sha256_bytes(case["case_id"].encode("utf-8"))[:20]
    row = {
        "validator": settings["validator"],
        "violation_id": f"owl:inconsistent:{opaque_id}",
        "error_type": settings["error_type"],
        "focus": None,
        "path": settings["path"],
        "message": settings["verdict"]["message"],
    }
    if framing in {"location", "explanation"}:
        row["focus"] = focus
    if framing == "explanation":
        row["message"] = settings["explanation"]["message"]
    if tuple(row) != FEEDBACK_FIELDS:
        raise RuntimeError("feedback fields changed across conditions")
    return row


def render_condition_prompt(
    case: dict, framing: str, spec: dict, template: str
) -> tuple[str, dict]:
    feedback = feedback_item(case, framing, spec)
    prompt = render_repair_prompt(
        template,
        case,
        case_content(case, "injected"),
        [feedback],
    )
    return prompt, feedback


def prompt_outside_feedback(prompt: str) -> tuple[str, str]:
    marker = "Validation feedback:\n"
    if prompt.count(marker) != 1:
        raise RuntimeError("repair prompt has an unexpected feedback block")
    prefix, remainder = prompt.split(marker, 1)
    _feedback, separator, suffix = remainder.partition("\n\nRepaired graph:")
    if not separator:
        raise RuntimeError("repair prompt feedback boundary is missing")
    return prefix, suffix


def shacl_ids(symbolic: dict) -> set[str]:
    return {
        row["violation_id"]
        for row in symbolic["shacl"]["violations"]
    }


def actionable_grounding_ids(rows: list[dict]) -> set[str]:
    return {row["violation_id"] for row in rows}


def initial_measurement(
    case: dict,
    cache: dict,
    profile,
    ontology,
) -> dict:
    injected = case_content(case, "injected")
    symbolic = validate_symbolic_state(case, injected, profile, ontology)
    if symbolic["owl_consistent"]:
        raise RuntimeError(f"RQ3 injected graph is OWL consistent: {case['case_id']}")
    if not set(primary_added(case)) <= set(injected):
        raise RuntimeError(f"RQ3 target absent from injected graph: {case['case_id']}")
    required = set(case_content(case, "clean")) | set(injected)
    if not required <= set(cache):
        raise RuntimeError(f"locked grounding cache gap: {case['case_id']}")
    grounding_rows = initial_grounding_feedback(case, cache)
    return {
        "triples": [list(value) for value in injected],
        "shacl": symbolic["shacl"],
        "shacl_violation_ids": sorted(shacl_ids(symbolic)),
        "owl_consistent": symbolic["owl_consistent"],
        "grounding_actionable_violation_ids": sorted(
            actionable_grounding_ids(grounding_rows)
        ),
    }


def controlled_target_removed(case: dict, triples) -> bool:
    current = set(normalized_triples(triples))
    return all(value not in current for value in primary_added(case))


def edit_metrics(before, after) -> dict:
    initial = set(normalized_triples(before))
    repaired = set(normalized_triples(after))
    removed = sorted(initial - repaired)
    added = sorted(repaired - initial)
    return {
        "removed_from_injected": [list(value) for value in removed],
        "added_to_injected": [list(value) for value in added],
        "symmetric_difference_from_injected": len(removed) + len(added),
        "graph_size_before": len(initial),
        "graph_size_after": len(repaired),
        "graph_size_delta": len(repaired) - len(initial),
    }


def failed_outcome(failure: str) -> dict:
    return {
        "controlled_target_removed": False,
        "owl_consistent": None,
        "exact_reference_recovery": False,
        "collateral_edit": None,
        "new_raw_shacl_findings": None,
        "new_grounding_findings": None,
        "owl_inconsistent_after_target_removal": None,
        "output_failure": failure,
        "edit_distance_from_injected": None,
        "edit_distance_from_clean_reference": None,
    }


def run_task(
    task: dict,
    case: dict,
    spec: dict,
    base_spec: dict,
    template: str,
    grounding_template: str,
    cache: dict,
    initial: dict,
    profile,
    ontology,
) -> dict:
    prompt, feedback = render_condition_prompt(case, task["framing"], spec, template)
    generation = generate_repair(prompt, base_spec)
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
    repair = {
        "feedback": feedback,
        "rendered_prompt": prompt,
        "rendered_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        **generation,
        "parse": parsed,
    }
    row = {
        **task,
        "initial_measurement": initial,
        "repair": repair,
        "post_repair_measurement": None,
        "outcome": failed_outcome(parsed["failure"]),
    }
    if not parsed["ok"]:
        return row

    repaired = normalized_triples(parsed["triples"])
    symbolic = validate_symbolic_state(case, repaired, profile, ontology)
    baseline = clean_unsupported(case, cache)
    grounding_rows, judgments = later_grounding_feedback(
        case,
        repaired,
        cache,
        baseline,
        grounding_template,
    )
    reference = reference_metrics(
        case_content(case, "clean"),
        case_content(case, "injected"),
        repaired,
    )
    target_removed = controlled_target_removed(case, repaired)
    current_shacl_ids = shacl_ids(symbolic)
    initial_shacl_ids = set(initial["shacl_violation_ids"])
    current_grounding_ids = actionable_grounding_ids(grounding_rows)
    initial_grounding_ids = set(initial["grounding_actionable_violation_ids"])
    edit = edit_metrics(case_content(case, "injected"), repaired)
    row["post_repair_measurement"] = {
        "triples": [list(value) for value in repaired],
        "shacl": symbolic["shacl"],
        "shacl_violation_ids": sorted(current_shacl_ids),
        "new_raw_shacl_violation_ids": sorted(
            current_shacl_ids - initial_shacl_ids
        ),
        "owl_consistent": symbolic["owl_consistent"],
        "grounding": {
            "judgments": judgments,
            "actionable_violation_ids": sorted(current_grounding_ids),
            "new_actionable_violation_ids": sorted(
                current_grounding_ids - initial_grounding_ids
            ),
            "clean_baseline_unsupported_excluded": [
                list(value) for value in sorted(baseline)
            ],
        },
        "reference": reference,
        "edit": edit,
        "controlled_target_removed": target_removed,
    }
    row["outcome"] = {
        "controlled_target_removed": target_removed,
        "owl_consistent": symbolic["owl_consistent"],
        "exact_reference_recovery": reference["reference_recovery"],
        "collateral_edit": reference["collateral_symmetric_difference"] > 0,
        "new_raw_shacl_findings": bool(current_shacl_ids - initial_shacl_ids),
        "new_grounding_findings": bool(
            current_grounding_ids - initial_grounding_ids
        ),
        "owl_inconsistent_after_target_removal": (
            target_removed and not symbolic["owl_consistent"]
        ),
        "output_failure": None,
        "edit_distance_from_injected": edit[
            "symmetric_difference_from_injected"
        ],
        "edit_distance_from_clean_reference": reference[
            "reference_symmetric_difference"
        ],
    }
    return row


def load_locked_grounding(
    cases: list[dict], base_spec: dict, spec: dict
) -> tuple[dict, dict]:
    rows, metadata = load_grounding_results(load_cases(base_spec), base_spec)
    locked = spec["locked_grounding"]
    if metadata.get("results_sha256") != locked["results_sha256"]:
        raise RuntimeError("locked confirmatory grounding hash changed")
    if metadata.get("git_head") != locked["run_git_head"]:
        raise RuntimeError("locked confirmatory grounding run commit changed")
    return {case["case_id"]: rows[case["case_id"]] for case in cases}, metadata


def read_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def validate_resume_prefix(rows: list[dict], schedule: list[dict]) -> None:
    if len(rows) > len(schedule):
        raise RuntimeError("RQ3 output contains too many rows")
    seen = set()
    for row, expected in zip(rows, schedule):
        observed = {key: row.get(key) for key in expected}
        if observed != expected:
            raise RuntimeError("RQ3 completed rows are not the fixed schedule prefix")
        key = (row["case_id"], row["framing"])
        if key in seen:
            raise RuntimeError("duplicate RQ3 case and framing row")
        seen.add(key)


def restore_cached_judgments(caches: dict[str, dict], rows: list[dict]) -> None:
    for row in rows:
        post = row.get("post_repair_measurement")
        if not isinstance(post, dict):
            continue
        cache = caches[row["case_id"]]
        for judgment in post.get("grounding", {}).get("judgments", []):
            key = tuple(judgment["triple"])
            previous = cache.get(key)
            if previous is not None and previous["verdict"] != judgment["verdict"]:
                raise RuntimeError("cached grounding verdict changed across RQ3 arms")
            cache[key] = judgment


def require_clean_committed_worktree() -> None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("commit the RQ3 runner and start from a clean working tree")


def run_preflight() -> tuple[dict, dict, list[dict], dict, dict]:
    spec, base = load_framing_spec()
    cases = load_rq3_cases(base)
    grounding, grounding_metadata = load_locked_grounding(cases, base, spec)
    schedule = execution_schedule(cases)
    if len(schedule) != 90 or len(
        {(row["case_id"], row["framing"]) for row in schedule}
    ) != 90:
        raise RuntimeError("RQ3 execution schedule is incomplete")
    return spec, base, cases, grounding, grounding_metadata


def preflight_only() -> None:
    _spec, _base, cases, _grounding, _metadata = run_preflight()
    print(f"paired disjointness cases: {len(cases)}")
    print(f"repair generations: {len(cases) * len(CONDITIONS)}")
    print("framings: verdict, location, explanation")
    print("RQ3 offline preflight passed.")
    print("No model, grounding assessor, validator, reasoner, or repair was executed.")


def run() -> None:
    require_clean_committed_worktree()
    spec, base, cases, grounding_rows, grounding_metadata = run_preflight()
    schedule = execution_schedule(cases)
    output = repository_path(spec["output"]["results_path"])
    metadata_path = repository_path(spec["output"]["metadata_path"])
    existing = read_existing_rows(output)
    validate_resume_prefix(existing, schedule)
    head = git_head()
    spec_hash = sha256_file(SPEC_PATH)
    if output.exists() and not metadata_path.exists():
        raise RuntimeError("RQ3 result exists without metadata")
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("spec_sha256") != spec_hash:
            raise RuntimeError("existing RQ3 output belongs to another specification")
        if metadata.get("git_head") != head:
            raise RuntimeError("existing RQ3 output belongs to another commit")
        if metadata.get("status") == "complete" and len(existing) != 90:
            raise RuntimeError("RQ3 metadata is complete but rows are missing")
        if metadata.get("status") == "complete":
            if metadata.get("results_sha256") != sha256_file(output):
                raise RuntimeError("completed RQ3 result hash mismatch")
            print("RQ3 is already complete. No model was executed.")
            print(f"results: {output.relative_to(ROOT)}")
            return
    else:
        metadata = None

    repair_model, grounding_model = verify_model_metadata(base)
    _symbolic_spec, profile, ontology = load_symbolic_context(base)
    template = repository_path(spec["generation"]["prompt_path"]).read_text(
        encoding="utf-8"
    )
    grounding_template = grounding_judge.load_prompt()
    case_by_id = {case["case_id"]: case for case in cases}
    caches = {
        case_id: judgment_cache(row)
        for case_id, row in grounding_rows.items()
    }
    restore_cached_judgments(caches, existing)
    initial = {
        case["case_id"]: initial_measurement(
            case,
            caches[case["case_id"]],
            profile,
            ontology,
        )
        for case in cases
    }

    if metadata is None:
        metadata = {
            "version": 1,
            "status": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": head,
            "spec_path": str(SPEC_PATH.relative_to(ROOT)),
            "spec_sha256": spec_hash,
            "base_runner_spec_sha256": sha256_file(
                ROOT / "experiments" / "sosa_ssn_confirmatory_runner_spec.json"
            ),
            "locked_grounding_results_sha256": grounding_metadata[
                "results_sha256"
            ],
            "repair_model": repair_model,
            "grounding_model": grounding_model,
            "case_ids": [case["case_id"] for case in cases],
            "execution_schedule": schedule,
            "repair_generations": 90,
            "completed_observations": len(existing),
            "invalid_output_retry": False,
            "post_repair_measurement_visible_to_model": False,
            "resume_policy": "complete rows for a case and framing only",
        }
    completed = len(existing)
    remaining = schedule[completed:]
    for index, task in enumerate(remaining, start=1):
        case = case_by_id[task["case_id"]]
        print(
            f"[{index}/{len(remaining)} remaining] "
            f"{task['case_id']} ({task['framing']})"
        )
        row = run_task(
            task,
            case,
            spec,
            base,
            template,
            grounding_template,
            caches[case["case_id"]],
            initial[case["case_id"]],
            profile,
            ontology,
        )
        append_jsonl(output, row)
        completed += 1
        metadata["completed_observations"] = completed
        metadata["status"] = "complete" if completed == 90 else "running"
        metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(metadata_path, metadata)
        outcome = row["outcome"]
        print(
            f"  target={outcome['controlled_target_removed']} "
            f"owl={outcome['owl_consistent']} "
            f"failure={outcome['output_failure']}"
        )
    metadata["status"] = "complete"
    metadata["completed_observations"] = completed
    metadata["results_sha256"] = sha256_file(output)
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(metadata_path, metadata)
    print(f"RQ3 observations complete: {completed}/90")
    print(f"results: {output.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight_only()
    else:
        run()


if __name__ == "__main__":
    main()
