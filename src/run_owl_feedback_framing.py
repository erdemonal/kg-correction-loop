import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src import grounding_judge
from src.repair_engine import (
    normalize_triples,
    primary_delta,
    reference_metrics,
    triple_set,
)
from src.repair_validation import (
    allowed_relations,
    controlled_context,
    controlled_owl_focus,
    revalidate_symbolic,
)
from src.run_controlled_repair import (
    generate_repair,
    grounding_cache_for_case,
    grounding_identity,
    index_unique,
    judge_current_grounding,
    load_frozen_grounding,
    local_repair_model_metadata,
    parse_repair_response,
    read_json,
    render_repair_prompt,
    verify_grounding_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    ROOT / "experiments" / "owl_feedback_framing_spec.json"
)
PROTOCOL_PATH = (
    ROOT / "experiments" / "owl_feedback_framing_protocol.md"
)
PROMPT_PATH = ROOT / "experiments" / "repair_prompt.txt"
SELECTION_PATH = (
    ROOT / "experiments" / "controlled_selection.json"
)
BASELINE_PATH = (
    ROOT / "experiments" / "text2kgbench_llama31_baseline.json"
)
FROZEN_GROUNDING_RESULTS = (
    ROOT / "results" / "controlled_grounding_validation.jsonl"
)
FROZEN_TARGET_ANALYSIS = (
    ROOT / "results" / "controlled_grounding_target_analysis.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "owl_feedback_framing.jsonl"

CONDITIONS = ("verdict", "location", "explanation")
FEEDBACK_FIELDS = (
    "validator",
    "violation_id",
    "error_type",
    "focus",
    "path",
    "message",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def git_head():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def verify_locked_rq2_ancestor(commit):
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Locked RQ2 commit is not an ancestor of HEAD: {commit}"
        )


def verify_file_hashes(mapping):
    for relative, expected in mapping.items():
        path = ROOT / relative

        if not path.exists():
            raise RuntimeError(f"Missing locked dependency: {path}")

        actual = sha256_file(path)

        if actual != expected:
            raise RuntimeError(
                f"Locked dependency hash mismatch for {relative}. "
                f"Expected {expected}, got {actual}."
            )


def load_spec():
    spec = read_json(SPEC_PATH)

    if spec.get("version") != 1:
        raise RuntimeError("Unexpected OWL feedback framing spec version")

    if spec.get("status") != "pre_run_candidate":
        raise RuntimeError(
            "RQ3 spec must be the audited candidate recorded before the run"
        )

    verify_locked_rq2_ancestor(spec["rq2_locked_commit"])
    verify_file_hashes(spec["locked_dependencies"])

    prompt_spec = spec["prompt"]

    if prompt_spec["path"] != str(PROMPT_PATH.relative_to(ROOT)):
        raise RuntimeError("RQ3 must reuse the locked RQ2 repair prompt")

    if sha256_file(PROMPT_PATH) != prompt_spec["sha256"]:
        raise RuntimeError("RQ3 repair prompt hash mismatch")

    case_spec = spec["cases"]

    if case_spec["selection_path"] != str(
        SELECTION_PATH.relative_to(ROOT)
    ):
        raise RuntimeError("Unexpected RQ3 controlled selection path")

    if sha256_file(SELECTION_PATH) != case_spec["selection_sha256"]:
        raise RuntimeError("RQ3 controlled selection hash mismatch")

    baseline = read_json(BASELINE_PATH)
    baseline_model = baseline["model"]
    model_spec = spec["repair_model"]

    if model_spec["name"] != baseline_model["name"]:
        raise RuntimeError(
            "RQ3 repair model differs from extraction baseline"
        )

    expected_options = {
        "temperature": baseline_model["temperature"],
        "seed": baseline_model["seed"],
        "num_ctx": baseline_model["num_ctx"],
        "num_predict": baseline_model["num_predict"],
    }

    if model_spec["options"] != expected_options:
        raise RuntimeError(
            "RQ3 generation settings differ from extraction baseline"
        )

    if tuple(spec["design"]["condition_order"]) != CONDITIONS:
        raise RuntimeError("Unexpected RQ3 condition order")

    if spec["design"]["repair_steps_per_condition"] != 1:
        raise RuntimeError("RQ3 must contain exactly one repair step")

    if spec["design"]["repair_generations"] != 30:
        raise RuntimeError("RQ3 must contain exactly 30 generations")

    if tuple(spec["feedback"]["fixed_fields"]) != FEEDBACK_FIELDS:
        raise RuntimeError("Unexpected RQ3 feedback schema")

    feedback = spec["feedback"]

    if not feedback["model_receives_only_owl_feedback"]:
        raise RuntimeError("RQ3 repair model must receive only OWL feedback")

    if (
        feedback["shacl_feedback_included"]
        or feedback["grounding_feedback_included"]
    ):
        raise RuntimeError("Hidden validators leaked into RQ3 feedback")

    if spec["post_repair_measurement"]["visible_to_repair_model"]:
        raise RuntimeError("Measurement after repair is visible to the model")

    if spec["output"]["invalid_output_retry"]:
        raise RuntimeError("RQ3 retry after invalid output must remain disabled")

    if spec["runner"]["custom_result_path_allowed"]:
        raise RuntimeError("RQ3 must use its single fixed result path")

    if not spec["pre_run_audit"]["required"]:
        raise RuntimeError("RQ3 pre-run leakage audit must be required")

    return spec, baseline


def selected_disjointness_cases(spec, selection=None):
    if selection is None:
        selection = read_json(SELECTION_PATH)["cases"]

    rows = [
        row for row in selection
        if row.get("error_type") == spec["cases"]["error_type"]
    ]

    if len(rows) != spec["cases"]["count"]:
        raise RuntimeError(
            "Controlled selection does not contain exactly 10 "
            "disjointness cases"
        )

    ids = [row.get("id") for row in rows]

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate RQ3 controlled case id")

    by_domain = {
        domain: sum(row.get("domain") == domain for row in rows)
        for domain in ("movie", "music")
    }

    if by_domain != {
        "movie": spec["cases"]["movie_count"],
        "music": spec["cases"]["music_count"],
    }:
        raise RuntimeError(f"Unexpected RQ3 domain balance: {by_domain}")

    return rows


def execution_schedule(case_rows, conditions=CONDITIONS):
    rows = []

    for case_index, case in enumerate(case_rows):
        offset = case_index % len(conditions)
        rotated = conditions[offset:] + conditions[:offset]

        for condition in rotated:
            rows.append(
                {
                    "execution_index": len(rows) + 1,
                    "case_index": case_index + 1,
                    "id": case["id"],
                    "domain": case["domain"],
                    "condition": condition,
                }
            )

    return rows


def content_triples(case):
    return normalize_triples(
        (
            row.subject,
            row.predicate,
            row.object,
        )
        for row in case.content
    )


def feedback_item(context, condition, spec):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown RQ3 feedback condition: {condition}")

    injected = content_triples(context["injected"])
    focus = controlled_owl_focus(context, injected)

    if focus is None:
        raise RuntimeError(
            f"{context['selected']['id']}: controlled OWL focus missing"
        )

    feedback_spec = spec["feedback"]
    row = {
        "validator": feedback_spec["validator"],
        "violation_id": (
            f"owl:inconsistent:{context['selected']['id']}"
        ),
        "error_type": feedback_spec["error_type"],
        "focus": None,
        "path": feedback_spec["path"],
        "message": feedback_spec["verdict"]["message"],
    }

    if condition in {"location", "explanation"}:
        row["focus"] = focus

    if condition == "explanation":
        domain = context["selected"]["domain"]
        row["message"] = feedback_spec["explanation"][
            "message_by_domain"
        ][domain]

    if tuple(row) != FEEDBACK_FIELDS:
        raise RuntimeError("RQ3 feedback fields changed across conditions")

    return row


def render_condition_prompt(context, condition, spec, prompt_template):
    triples = content_triples(context["injected"])
    feedback = feedback_item(context, condition, spec)
    prompt = render_repair_prompt(
        prompt_template,
        context["clean"].source_text,
        allowed_relations(context),
        triples,
        [feedback],
    )
    return prompt, feedback


def shacl_violation_ids(shacl):
    return {
        row["violation_id"] for row in shacl["violations"]
    }


def unsupported_grounding_ids(judgments):
    return {
        grounding_identity(tuple(row["triple"]))
        for row in judgments
        if row["verdict"] == "UNSUPPORTED"
    }


def initial_measurement(context, cache):
    injected = content_triples(context["injected"])
    symbolic = revalidate_symbolic(context, injected)

    if symbolic["owl_consistent"]:
        raise RuntimeError(
            f"{context['selected']['id']}: injected RQ3 graph is "
            "unexpectedly OWL consistent"
        )

    before = set(cache)
    grounding = judge_current_grounding(
        context["clean"].source_text,
        injected,
        cache,
    )

    if set(cache) != before:
        raise RuntimeError(
            f"{context['selected']['id']}: injected graph required a "
            "new grounding judgment instead of a frozen one"
        )

    return {
        "triples": [list(row) for row in injected],
        "raw_shacl": symbolic["shacl"],
        "raw_shacl_violation_ids": sorted(
            shacl_violation_ids(symbolic["shacl"])
        ),
        "owl_consistent": symbolic["owl_consistent"],
        "grounding_judgments": grounding,
        "grounding_unsupported_ids": sorted(
            unsupported_grounding_ids(grounding)
        ),
    }


def controlled_target_removed(context, repaired_triples):
    clean = content_triples(context["clean"])
    injected = content_triples(context["injected"])
    added = primary_delta(clean, injected)["added"]

    if len(added) != 1:
        raise RuntimeError(
            f"{context['selected']['id']}: expected exactly one "
            "injected disjointness triple"
        )

    return added.isdisjoint(triple_set(repaired_triples))


def edit_metrics(injected_triples, repaired_triples):
    injected = triple_set(injected_triples)
    repaired = triple_set(repaired_triples)
    removed = injected - repaired
    added = repaired - injected

    return {
        "removed_from_injected": [
            list(row) for row in sorted(removed)
        ],
        "added_to_injected": [
            list(row) for row in sorted(added)
        ],
        "symmetric_difference_from_injected": len(removed | added),
        "graph_size_before": len(injected),
        "graph_size_after": len(repaired),
        "graph_size_delta": len(repaired) - len(injected),
    }


def generate_one(context, condition, spec, prompt_template):
    prompt, feedback = render_condition_prompt(
        context,
        condition,
        spec,
        prompt_template,
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
            generation["raw_response"],
            allowed_relations(context),
        )

    return {
        "feedback": feedback,
        "rendered_prompt_sha256": sha256_bytes(
            prompt.encode("utf-8")
        ),
        "rendered_prompt": prompt,
        **generation,
        "parse": parsed,
    }


def symbolic_post_measurement(context, repaired_triples):
    clean = content_triples(context["clean"])
    injected = content_triples(context["injected"])
    repaired = normalize_triples(repaired_triples)
    symbolic = revalidate_symbolic(context, repaired)
    reference = reference_metrics(clean, injected, repaired)
    target_removed = controlled_target_removed(context, repaired)

    return {
        "triples": [list(row) for row in repaired],
        "raw_shacl": symbolic["shacl"],
        "raw_shacl_violation_ids": sorted(
            shacl_violation_ids(symbolic["shacl"])
        ),
        "owl_consistent": symbolic["owl_consistent"],
        "reference": reference,
        "edit": edit_metrics(injected, repaired),
        "controlled_target_removed": target_removed,
        "owl_inconsistent_after_target_removal": (
            target_removed and not symbolic["owl_consistent"]
        ),
    }


def finalize_row(row, context, initial, cache):
    parsed = row["repair"]["parse"]

    if not parsed["ok"]:
        row["post_repair_measurement"] = None
        row["outcome"] = {
            "controlled_target_removed": False,
            "owl_consistent": None,
            "reference_recovery": False,
            "collateral_edit": None,
            "new_raw_shacl_findings": None,
            "new_grounding_findings": None,
            "owl_inconsistent_after_target_removal": None,
            "output_failure": parsed["failure"],
            "edit_distance_from_injected": None,
            "edit_distance_from_clean_reference": None,
        }
        return row

    post = row["post_repair_measurement"]
    repaired = normalize_triples(post["triples"])
    grounding = [cache[triple] for triple in repaired]
    grounding_ids = unsupported_grounding_ids(grounding)
    initial_grounding = set(initial["grounding_unsupported_ids"])
    shacl_ids = set(post["raw_shacl_violation_ids"])
    initial_shacl = set(initial["raw_shacl_violation_ids"])
    reference = post["reference"]

    post["grounding_judgments"] = grounding
    post["grounding_unsupported_ids"] = sorted(grounding_ids)
    post["new_grounding_violation_ids"] = sorted(
        grounding_ids - initial_grounding
    )
    post["new_raw_shacl_violation_ids"] = sorted(
        shacl_ids - initial_shacl
    )

    row["outcome"] = {
        "controlled_target_removed": post[
            "controlled_target_removed"
        ],
        "owl_consistent": post["owl_consistent"],
        "reference_recovery": reference["reference_recovery"],
        "collateral_edit": (
            reference["collateral_symmetric_difference"] > 0
        ),
        "new_raw_shacl_findings": bool(
            post["new_raw_shacl_violation_ids"]
        ),
        "new_grounding_findings": bool(
            post["new_grounding_violation_ids"]
        ),
        "owl_inconsistent_after_target_removal": post[
            "owl_inconsistent_after_target_removal"
        ],
        "output_failure": None,
        "edit_distance_from_injected": post["edit"][
            "symmetric_difference_from_injected"
        ],
        "edit_distance_from_clean_reference": reference[
            "reference_symmetric_difference"
        ],
    }
    return row


def run_experiment(spec, case_rows, frozen_grounding):
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    contexts = {
        row["id"]: controlled_context(row["id"])
        for row in case_rows
    }
    state = {}

    for row in case_rows:
        case_id = row["id"]

        if case_id not in frozen_grounding:
            raise RuntimeError(
                f"{case_id}: missing frozen grounding result"
            )

        cache, _ = grounding_cache_for_case(
            frozen_grounding[case_id]
        )
        state[case_id] = {
            "cache": cache,
            "initial": initial_measurement(
                contexts[case_id],
                cache,
            ),
        }

    rows = []

    for task in execution_schedule(case_rows):
        context = contexts[task["id"]]
        print(
            f"[{task['execution_index']:02d}/30] "
            f"{task['id']} ({task['condition']})"
        )
        repair = generate_one(
            context,
            task["condition"],
            spec,
            prompt_template,
        )
        row = {
            **task,
            "initial_measurement": state[task["id"]]["initial"],
            "repair": repair,
            "post_repair_measurement": None,
        }

        if repair["parse"]["ok"]:
            row["post_repair_measurement"] = (
                symbolic_post_measurement(
                    context,
                    repair["parse"]["triples"],
                )
            )

        rows.append(row)

    novel_grounding_calls = 0

    for case in case_rows:
        case_id = case["id"]
        context = contexts[case_id]
        cache = state[case_id]["cache"]
        union = sorted(
            {
                tuple(triple)
                for row in rows
                if row["id"] == case_id
                and row["repair"]["parse"]["ok"]
                for triple in row["repair"]["parse"]["triples"]
            }
        )
        before = set(cache)
        judge_current_grounding(
            context["clean"].source_text,
            union,
            cache,
        )
        novel_grounding_calls += len(set(cache) - before)

        for row in rows:
            if row["id"] == case_id:
                finalize_row(
                    row,
                    context,
                    state[case_id]["initial"],
                    cache,
                )

    return rows, novel_grounding_calls


def preflight_files():
    required = [
        SPEC_PATH,
        PROTOCOL_PATH,
        PROMPT_PATH,
        SELECTION_PATH,
        BASELINE_PATH,
        FROZEN_GROUNDING_RESULTS,
        FROZEN_TARGET_ANALYSIS,
    ]
    missing = [path for path in required if not path.exists()]

    if missing:
        raise RuntimeError(
            "Missing required RQ3 input files:\n"
            + "\n".join(str(path) for path in missing)
        )


def build_metadata(
    output,
    spec,
    repair_model,
    grounding_model,
    case_rows,
    novel_grounding_calls,
):
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "rq2_locked_commit": spec["rq2_locked_commit"],
        "spec": str(SPEC_PATH.relative_to(ROOT)),
        "spec_sha256": sha256_file(SPEC_PATH),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "repair_prompt": str(PROMPT_PATH.relative_to(ROOT)),
        "repair_prompt_sha256": sha256_file(PROMPT_PATH),
        "controlled_selection": str(SELECTION_PATH.relative_to(ROOT)),
        "controlled_selection_sha256": sha256_file(SELECTION_PATH),
        "frozen_grounding_results": str(
            FROZEN_GROUNDING_RESULTS.relative_to(ROOT)
        ),
        "frozen_grounding_results_sha256": sha256_file(
            FROZEN_GROUNDING_RESULTS
        ),
        "frozen_target_analysis": str(
            FROZEN_TARGET_ANALYSIS.relative_to(ROOT)
        ),
        "frozen_target_analysis_sha256": sha256_file(
            FROZEN_TARGET_ANALYSIS
        ),
        "repair_model": repair_model,
        "grounding_model": grounding_model,
        "case_ids": [row["id"] for row in case_rows],
        "execution_schedule": execution_schedule(case_rows),
        "repair_generations": 30,
        "novel_grounding_assessor_calls": novel_grounding_calls,
        "result_path": str(output.resolve()),
        "invalid_output_retry": False,
        "post_repair_measurement_visible_to_model": False,
    }


def write_results(output, rows, metadata):
    metadata_path = Path(str(output) + ".meta.json")
    output_tmp = Path(str(output) + ".tmp")
    metadata_tmp = Path(str(metadata_path) + ".tmp")

    occupied = [
        path for path in (
            output,
            metadata_path,
            output_tmp,
            metadata_tmp,
        )
        if path.exists()
    ]

    if occupied:
        raise RuntimeError(
            "RQ3 output path already exists; outputs are never "
            "overwritten:\n"
            + "\n".join(str(path) for path in occupied)
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output_tmp.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_tmp.replace(output)
    metadata_tmp.replace(metadata_path)


def ensure_output_paths_available(output):
    metadata_path = Path(str(output) + ".meta.json")
    output_tmp = Path(str(output) + ".tmp")
    metadata_tmp = Path(str(metadata_path) + ".tmp")
    occupied = [
        path for path in (
            output,
            metadata_path,
            output_tmp,
            metadata_tmp,
        )
        if path.exists()
    ]

    if occupied:
        raise RuntimeError(
            "RQ3 output path already exists; outputs are never "
            "overwritten:\n"
            + "\n".join(str(path) for path in occupied)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify locked inputs and local model identities. "
            "No generation or validation is run."
        ),
    )
    args = parser.parse_args()
    output = DEFAULT_OUTPUT

    preflight_files()
    spec, baseline = load_spec()
    verify_grounding_spec()
    case_rows = selected_disjointness_cases(spec)
    repair_model = local_repair_model_metadata(spec, baseline)
    grounding_model = grounding_judge.model_metadata()

    if args.preflight_only:
        print("RQ3 OWL feedback framing preflight: OK")
        print(f"cases: {len(case_rows)}")
        print("conditions: 3")
        print("planned repair generations: 30")
        print(f"repair prompt sha256: {sha256_file(PROMPT_PATH)}")
        print(f"repair model digest: {repair_model['model_digest']}")
        print(
            "grounding model digest: "
            f"{grounding_model['model_digest']}"
        )
        print("No generation or validator was run.")
        return

    try:
        ensure_output_paths_available(output)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    frozen_grounding, _ = load_frozen_grounding()
    rows, novel_grounding_calls = run_experiment(
        spec,
        case_rows,
        frozen_grounding,
    )

    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 RQ3 rows, got {len(rows)}")

    key_count = len(
        {(row["id"], row["condition"]) for row in rows}
    )

    if key_count != 30:
        raise RuntimeError("RQ3 result keys are not unique")

    metadata = build_metadata(
        output,
        spec,
        repair_model,
        grounding_model,
        case_rows,
        novel_grounding_calls,
    )
    write_results(output, rows, metadata)
    print(f"wrote: {output}")
    print(f"wrote: {output}.meta.json")
    print("repair generations: 30")
    print(
        "novel grounding assessor calls: "
        f"{novel_grounding_calls}"
    )


if __name__ == "__main__":
    main()
