"""Analyze one locked, paired OWL feedback experiment without model execution."""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "owl_feedback_framing.jsonl"
DEFAULT_METADATA = ROOT / "results" / "owl_feedback_framing.jsonl.meta.json"
DEFAULT_OUTPUT = ROOT / "results" / "owl_feedback_framing_analysis.json"
ANALYSIS_SPEC = ROOT / "experiments" / "owl_feedback_analysis_spec.json"
RQ3_SPEC = ROOT / "experiments" / "owl_feedback_framing_spec.json"
CONDITIONS = ("verdict", "location", "explanation")
DOMAINS = ("movie", "music")
PAIRED_COMPARISONS = (
    ("explanation", "verdict"),
    ("explanation", "location"),
    ("location", "verdict"),
)
GRAPH_FIELDS = (
    "owl_consistent",
    "collateral_edit",
    "new_raw_shacl_findings",
    "new_grounding_findings",
    "owl_inconsistent_after_target_removal",
    "edit_distance_from_injected",
    "edit_distance_from_clean_reference",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": numerator / denominator if denominator else None,
    }


def read_rows(path):
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid locked JSONL at line {line_number}") from exc
    return rows


def validate_metadata(metadata, analysis_spec, rq3_spec):
    locked = analysis_spec["locked_run"]
    if metadata.get("git_head") != locked["git_head"]:
        raise RuntimeError("Locked RQ3 run commit changed")
    if metadata.get("rq2_locked_commit") != locked["rq2_locked_commit"]:
        raise RuntimeError("Locked RQ2 ancestor changed")
    if metadata.get("repair_generations") != locked["repair_generations"]:
        raise RuntimeError("Locked run must contain exactly 30 repair generations")
    if metadata.get("invalid_output_retry") is not False:
        raise RuntimeError("Locked run enabled retry after invalid output")
    if metadata.get("post_repair_measurement_visible_to_model") is not False:
        raise RuntimeError("Measurement after repair was visible to the repair model")
    if len(metadata.get("case_ids", [])) != 10:
        raise RuntimeError("Locked run must contain exactly ten paired cases")

    tracked_inputs = (
        ("spec", "spec_sha256"),
        ("protocol", "protocol_sha256"),
        ("repair_prompt", "repair_prompt_sha256"),
        ("controlled_selection", "controlled_selection_sha256"),
    )
    for path_key, hash_key in tracked_inputs:
        path = ROOT / metadata[path_key]
        if sha256_file(path) != metadata[hash_key]:
            raise RuntimeError(f"Locked tracked input changed: {metadata[path_key]}")
    for relative_path, expected_hash in rq3_spec["locked_dependencies"].items():
        if sha256_file(ROOT / relative_path) != expected_hash:
            raise RuntimeError(f"Locked RQ1 or RQ2 dependency changed: {relative_path}")

    design = rq3_spec["design"]
    if tuple(design["condition_order"]) != CONDITIONS:
        raise RuntimeError("Locked condition ordering changed")
    if design["repair_steps_per_condition"] != 1:
        raise RuntimeError("RQ3 must remain a single step repair experiment")
    repair_model = metadata["repair_model"]
    if repair_model["model"] != rq3_spec["repair_model"]["name"]:
        raise RuntimeError("Locked repair model changed")
    if repair_model["options"] != rq3_spec["repair_model"]["options"]:
        raise RuntimeError("Locked repair generation options changed")
    baseline_model = read_json(
        ROOT / "experiments" / "text2kgbench_llama31_baseline.json"
    )["model"]
    if repair_model["model_digest"] != baseline_model["digest"]:
        raise RuntimeError("Locked extraction or repair model digest changed")

    grounding_spec = read_json(ROOT / "experiments" / "grounding_judge_spec.json")
    grounding_model = metadata["grounding_model"]
    if grounding_model["model"] != grounding_spec["model"]:
        raise RuntimeError("Locked grounding assessor changed")
    if grounding_model["options"] != grounding_spec["options"]:
        raise RuntimeError("Locked grounding assessor settings changed")
    if grounding_model["judge_version"] != f"v{grounding_spec['version']}":
        raise RuntimeError("Locked grounding assessor version changed")
    if grounding_model["prompt_sha256"] != sha256_file(ROOT / grounding_spec["prompt"]):
        raise RuntimeError("Locked grounding assessor prompt changed")


def validate_feedback_and_prompt(row, rq3_spec):
    repair = row["repair"]
    feedback = repair["feedback"]
    settings = rq3_spec["feedback"]
    if set(feedback) != set(settings["fixed_fields"]):
        raise RuntimeError(f"Feedback schema changed for {row['id']}")
    if feedback["validator"] != "owl_consistency":
        raise RuntimeError("Feedback other than OWL reached the repair prompt")
    if feedback["error_type"] is not None or feedback["path"] is not None:
        raise RuntimeError("A fixed feedback field leaked extra condition information")
    if feedback["violation_id"] != f"owl:inconsistent:{row['id']}":
        raise RuntimeError("Unexpected OWL violation identity")

    condition = row["condition"]
    if condition == "verdict":
        expected_message = settings["verdict"]["message"]
        if feedback["focus"] is not None:
            raise RuntimeError("Verdict only feedback exposed the focus entity")
    elif condition == "location":
        expected_message = settings["location"]["message"]
        if not feedback["focus"]:
            raise RuntimeError("Location feedback omitted the focus entity")
    else:
        expected_message = settings["explanation"]["message_by_domain"][row["domain"]]
        if not feedback["focus"]:
            raise RuntimeError("Explanation feedback omitted the focus entity")
    if feedback["message"] != expected_message:
        raise RuntimeError(f"Locked feedback wording changed for {row['id']}")

    prompt = repair["rendered_prompt"]
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != repair[
        "rendered_prompt_sha256"
    ]:
        raise RuntimeError(f"Rendered prompt hash mismatch for {row['id']}")
    marker = "Validation feedback:\n"
    if prompt.count(marker) != 1:
        raise RuntimeError("Repair prompt must contain exactly one feedback block")
    prefix, suffix = prompt.split(marker, 1)
    feedback_json, separator, _ = suffix.partition("\n\nRepaired graph:")
    if not separator or json.loads(feedback_json) != [feedback]:
        raise RuntimeError("Rendered feedback does not match recorded OWL feedback")
    return prefix


def validate_outcome(row):
    parse = row["repair"]["parse"]
    outcome = row["outcome"]
    post = row["post_repair_measurement"]
    if parse["ok"] is False:
        if post is not None:
            raise RuntimeError("An invalid model output was assigned a validated graph")
        if not outcome["output_failure"] or outcome["output_failure"] != parse["failure"]:
            raise RuntimeError("Output failure was not recorded separately")
        if outcome["controlled_target_removed"] or outcome["reference_recovery"]:
            raise RuntimeError("Invalid model output cannot count as success for that request")
        if any(outcome[field] is not None for field in GRAPH_FIELDS):
            raise RuntimeError("Outcomes that depend on a parsed graph must be undefined after failure")
        return

    if parse["ok"] is not True or not isinstance(post, dict):
        raise RuntimeError("Usable repair output must have measurement after repair")
    if outcome["output_failure"] is not None or parse["failure"] is not None:
        raise RuntimeError("A usable repair output was labeled as a failure")
    if outcome["controlled_target_removed"] != post["controlled_target_removed"]:
        raise RuntimeError("Controlled target outcome does not match recorded measurement")
    if outcome["owl_consistent"] != post["owl_consistent"]:
        raise RuntimeError("OWL consistency outcome does not match recorded measurement")
    if outcome["reference_recovery"] != post["reference"]["reference_recovery"]:
        raise RuntimeError("Reference recovery does not match recorded measurement")
    if outcome["collateral_edit"] != bool(
        post["reference"]["collateral_symmetric_difference"]
    ):
        raise RuntimeError("Collateral edit outcome does not match recorded measurement")
    if outcome["new_raw_shacl_findings"] != bool(post["new_raw_shacl_violation_ids"]):
        raise RuntimeError("New SHACL findings do not match recorded measurement")
    if outcome["new_grounding_findings"] != bool(post["new_grounding_violation_ids"]):
        raise RuntimeError("New grounding findings do not match recorded measurement")
    residual = outcome["controlled_target_removed"] and not outcome["owl_consistent"]
    if outcome["owl_inconsistent_after_target_removal"] != residual:
        raise RuntimeError("Target removal and remaining OWL inconsistency were conflated")
    if outcome["edit_distance_from_injected"] != post["edit"][
        "symmetric_difference_from_injected"
    ]:
        raise RuntimeError("Injected graph edit distance changed")
    if outcome["edit_distance_from_clean_reference"] != post["reference"][
        "reference_symmetric_difference"
    ]:
        raise RuntimeError("Clean reference edit distance changed")


def validate_run(rows, metadata, analysis_spec, rq3_spec):
    validate_metadata(metadata, analysis_spec, rq3_spec)
    if len(rows) != 30:
        raise RuntimeError("Locked RQ3 run must contain exactly 30 observations pairing a case with a condition")
    schedule = metadata.get("execution_schedule")
    if not isinstance(schedule, list) or len(schedule) != len(rows):
        raise RuntimeError("Locked execution schedule is missing or incomplete")

    grouped = defaultdict(dict)
    prefixes = {}
    for row, expected in zip(rows, schedule):
        identity = {
            key: row.get(key)
            for key in ("execution_index", "case_index", "id", "domain", "condition")
        }
        if identity != expected:
            raise RuntimeError("Result rows do not match the locked execution schedule")
        if row["condition"] not in CONDITIONS or row["domain"] not in DOMAINS:
            raise RuntimeError("Unexpected RQ3 condition or domain")
        if row["condition"] in grouped[row["id"]]:
            raise RuntimeError("Duplicate paired observation of a case and a condition")
        if row["initial_measurement"]["owl_consistent"] is not False:
            raise RuntimeError("A selected disjointness case was not initially inconsistent")
        prefix = validate_feedback_and_prompt(row, rq3_spec)
        if row["id"] in prefixes and prefixes[row["id"]] != prefix:
            raise RuntimeError("Prompt content outside the OWL feedback block changed")
        prefixes[row["id"]] = prefix
        grouped[row["id"]][row["condition"]] = row
        validate_outcome(row)

    if list(grouped) != metadata["case_ids"]:
        raise RuntimeError("Paired case identities differ from locked run metadata")
    for case_id, condition_rows in grouped.items():
        if set(condition_rows) != set(CONDITIONS):
            raise RuntimeError(f"Incomplete paired condition set for {case_id}")
        initial = [row["initial_measurement"] for row in condition_rows.values()]
        if any(item != initial[0] for item in initial[1:]):
            raise RuntimeError("Paired conditions did not begin from the same injected state")
        focused = [condition_rows[key]["repair"]["feedback"]["focus"] for key in CONDITIONS[1:]]
        if focused[0] != focused[1]:
            raise RuntimeError("Location and explanation identified different focus entities")
    domain_counts = Counter(rows["verdict"]["domain"] for rows in grouped.values())
    if domain_counts != Counter({"movie": 5, "music": 5}):
        raise RuntimeError("RQ3 must contain exactly five Movie and five Music cases")
    return grouped


def summarize_rows(rows):
    usable = [row for row in rows if row["outcome"]["output_failure"] is None]
    outcomes = [row["outcome"] for row in rows]
    graph_outcomes = [row["outcome"] for row in usable]
    counts = {
        "n": len(rows),
        "usable_outputs": len(usable),
        "output_failures": sum(bool(item["output_failure"]) for item in outcomes),
        "controlled_target_removed": sum(item["controlled_target_removed"] for item in outcomes),
        "reference_recovery": sum(item["reference_recovery"] for item in outcomes),
        "owl_consistent": sum(item["owl_consistent"] for item in graph_outcomes),
        "collateral_edit": sum(item["collateral_edit"] for item in graph_outcomes),
        "new_raw_shacl_findings": sum(item["new_raw_shacl_findings"] for item in graph_outcomes),
        "new_grounding_findings": sum(item["new_grounding_findings"] for item in graph_outcomes),
        "owl_inconsistent_after_target_removal": sum(
            item["owl_inconsistent_after_target_removal"] for item in graph_outcomes
        ),
        "output_failure_types": dict(
            Counter(item["output_failure"] for item in outcomes if item["output_failure"])
        ),
    }
    rates = {
        field: rate(counts[field], counts["n"])
        for field in ("controlled_target_removed", "reference_recovery", "output_failures")
    }
    rates.update(
        {
            field: rate(counts[field], counts["usable_outputs"])
            for field in (
                "owl_consistent",
                "collateral_edit",
                "new_raw_shacl_findings",
                "new_grounding_findings",
            )
        }
    )
    rates["owl_inconsistent_after_target_removal"] = rate(
        counts["owl_inconsistent_after_target_removal"],
        counts["controlled_target_removed"],
    )
    injected_edits = [item["edit_distance_from_injected"] for item in graph_outcomes]
    reference_edits = [item["edit_distance_from_clean_reference"] for item in graph_outcomes]
    return {
        "counts": counts,
        "rates": rates,
        "edits": {
            "usable_output_denominator": len(usable),
            "mean_from_injected": statistics.mean(injected_edits) if injected_edits else None,
            "median_from_injected": statistics.median(injected_edits) if injected_edits else None,
            "mean_from_clean_reference": (
                statistics.mean(reference_edits) if reference_edits else None
            ),
            "median_from_clean_reference": (
                statistics.median(reference_edits) if reference_edits else None
            ),
        },
    }


def paired_target_comparison(grouped, left, right):
    counts = Counter()
    case_rows = []
    for case_id, conditions in grouped.items():
        left_value = conditions[left]["outcome"]["controlled_target_removed"]
        right_value = conditions[right]["outcome"]["controlled_target_removed"]
        if left_value and right_value:
            transition = "both_resolved"
        elif left_value:
            transition = "left_only"
        elif right_value:
            transition = "right_only"
        else:
            transition = "neither_resolved"
        counts[transition] += 1
        case_rows.append(
            {
                "id": case_id,
                "domain": conditions[left]["domain"],
                "left_resolved": left_value,
                "right_resolved": right_value,
                "transition": transition,
            }
        )
    return {
        "left": left,
        "right": right,
        "n_paired_cases": len(grouped),
        "both_resolved": counts["both_resolved"],
        "left_only": counts["left_only"],
        "right_only": counts["right_only"],
        "neither_resolved": counts["neither_resolved"],
        "same": counts["both_resolved"] + counts["neither_resolved"],
        "net_target_difference": counts["left_only"] - counts["right_only"],
        "cases": case_rows,
    }


def case_record(case_id, conditions):
    record = {"id": case_id, "domain": conditions["verdict"]["domain"], "conditions": {}}
    for condition in CONDITIONS:
        row = conditions[condition]
        outcome = row["outcome"]
        record["conditions"][condition] = {
            "execution_index": row["execution_index"],
            "usable_output": outcome["output_failure"] is None,
            **outcome,
        }
    return record


def git_head():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_analysis(rows, metadata, analysis_spec, rq3_spec, result_path, metadata_path):
    result_hash = sha256_file(result_path)
    metadata_hash = sha256_file(metadata_path)
    locked = analysis_spec["locked_run"]
    if result_hash != locked["result_sha256"]:
        raise RuntimeError("Locked RQ3 result file hash changed. Reruns are not accepted.")
    if metadata_hash != locked["metadata_sha256"]:
        raise RuntimeError("Locked RQ3 metadata hash changed")
    grouped = validate_run(rows, metadata, analysis_spec, rq3_spec)
    by_condition = {
        condition: summarize_rows([conditions[condition] for conditions in grouped.values()])
        for condition in CONDITIONS
    }
    by_domain = {
        domain: {
            condition: summarize_rows(
                [
                    conditions[condition]
                    for conditions in grouped.values()
                    if conditions[condition]["domain"] == domain
                ]
            )
            for condition in CONDITIONS
        }
        for domain in DOMAINS
    }
    paired = {
        f"{left}_vs_{right}": paired_target_comparison(grouped, left, right)
        for left, right in PAIRED_COMPARISONS
    }
    residual_cases = sorted(
        case_id
        for case_id, conditions in grouped.items()
        if any(
            row["outcome"]["owl_inconsistent_after_target_removal"] is True
            for row in conditions.values()
        )
    )
    return {
        "analysis_unit": "controlled case",
        "paired_by_case": True,
        "n_paired_cases": len(grouped),
        "case_condition_observations": len(rows),
        "conditions": list(CONDITIONS),
        "by_condition": by_condition,
        "by_domain": by_domain,
        "paired_target_comparisons": paired,
        "pooled_case_condition_observations": {
            "independent_cases": False,
            **summarize_rows(rows),
        },
        "residual_owl_case_ids": residual_cases,
        "cases": [case_record(case_id, conditions) for case_id, conditions in grouped.items()],
        "input": {
            "result_path": str(result_path),
            "result_sha256": result_hash,
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_hash,
            "run_git_head": metadata["git_head"],
            "rq2_locked_commit": metadata["rq2_locked_commit"],
            "repair_generations": metadata["repair_generations"],
            "novel_grounding_assessor_calls": metadata["novel_grounding_assessor_calls"],
        },
        "analysis_provenance": {
            "analysis_git_head": git_head(),
            "analysis_script_sha256": sha256_file(__file__),
            "analysis_spec_sha256": sha256_file(ANALYSIS_SPEC),
            "models_executed": False,
            "validators_executed": False,
        },
        "claim_boundaries": analysis_spec["claim_boundaries"],
    }


def write_cases_csv(payload, path):
    fields = [
        "id", "domain", "condition", "usable_output", "controlled_target_removed",
        "owl_consistent", "reference_recovery", "collateral_edit", "new_raw_shacl_findings",
        "new_grounding_findings", "owl_inconsistent_after_target_removal", "output_failure",
        "edit_distance_from_injected", "edit_distance_from_clean_reference",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in payload["cases"]:
            for condition in CONDITIONS:
                values = case["conditions"][condition]
                writer.writerow(
                    {"id": case["id"], "domain": case["domain"], "condition": condition,
                     **{field: values.get(field) for field in fields[3:]}}
                )


def write_paired_csv(payload, path):
    fields = ["comparison", "id", "domain", "left", "right", "left_resolved", "right_resolved", "transition"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, comparison in payload["paired_target_comparisons"].items():
            for case in comparison["cases"]:
                writer.writerow({"comparison": name, "left": comparison["left"], "right": comparison["right"], **case})


def run(result_path=DEFAULT_RESULTS, metadata_path=DEFAULT_METADATA, output_path=DEFAULT_OUTPUT):
    result_path, metadata_path, output_path = map(Path, (result_path, metadata_path, output_path))
    analysis_spec = read_json(ANALYSIS_SPEC)
    rq3_spec = read_json(RQ3_SPEC)
    rows = read_rows(result_path)
    metadata = read_json(metadata_path)
    payload = build_analysis(rows, metadata, analysis_spec, rq3_spec, result_path, metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    cases_path = output_path.with_name("owl_feedback_framing_cases.csv")
    paired_path = output_path.with_name("owl_feedback_framing_paired.csv")
    write_cases_csv(payload, cases_path)
    write_paired_csv(payload, paired_path)

    print(f"paired controlled cases: {payload['n_paired_cases']}")
    print(f"observations pairing a case with a condition: {payload['case_condition_observations']}")
    print("condition       target  owl / usable  reference  collateral / usable  failures")
    for condition in CONDITIONS:
        counts = payload["by_condition"][condition]["counts"]
        print(
            f"{condition:<14} {counts['controlled_target_removed']}/{counts['n']}"
            f"     {counts['owl_consistent']}/{counts['usable_outputs']}"
            f"           {counts['reference_recovery']}/{counts['n']}"
            f"        {counts['collateral_edit']}/{counts['usable_outputs']}"
            f"                  {counts['output_failures']}/{counts['n']}"
        )
    for path in (output_path, cases_path, paired_path):
        print(f"wrote: {path}")
    print("No language model, grounding assessor, or validator was executed.")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.results, args.metadata, args.output)


if __name__ == "__main__":
    main()
