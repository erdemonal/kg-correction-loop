from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "experiments" / "sosa_ssn_feedback_framing_spec.json"
CASES_PATH = ROOT / "experiments" / "sosa_ssn_confirmatory_cases.jsonl"
DEFAULT_RESULTS = ROOT / "results" / "sosa_ssn_feedback_framing.jsonl"
DEFAULT_METADATA = ROOT / "results" / "sosa_ssn_feedback_framing.meta.json"
DEFAULT_ANALYSIS = ROOT / "results" / "sosa_ssn_feedback_framing_analysis.json"
DEFAULT_CASES_CSV = ROOT / "results" / "sosa_ssn_feedback_framing_analysis_cases.csv"

FRAMINGS = ("verdict", "location", "explanation")
PAIRS = (
    ("explanation", "verdict"),
    ("explanation", "location"),
    ("location", "verdict"),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(count: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {
            "count": count,
            "n": n,
            "rate": None,
            "lower_95": None,
            "upper_95": None,
        }
    rate = count / n
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return {
        "count": count,
        "n": n,
        "rate": rate,
        "lower_95": max(0.0, center - half),
        "upper_95": min(1.0, center + half),
    }


def cochran_q(matrix: list[list[bool]]) -> dict:
    if not matrix or any(len(row) != len(FRAMINGS) for row in matrix):
        raise RuntimeError("Cochran Q requires a complete three-column paired matrix")
    k = len(FRAMINGS)
    column_totals = [sum(row[index] for row in matrix) for index in range(k)]
    row_totals = [sum(row) for row in matrix]
    total = sum(column_totals)
    denominator = k * total - sum(value * value for value in row_totals)
    if denominator == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (
            (k - 1)
            * (k * sum(value * value for value in column_totals) - total * total)
            / denominator
        )
        # With three conditions the chi-square reference distribution has two
        # degrees of freedom, whose survival function is exp(-x / 2).
        p_value = math.exp(-statistic / 2)
    return {
        "test": "Cochran Q",
        "statistic": statistic,
        "degrees_of_freedom": k - 1,
        "p_value": p_value,
        "column_totals": dict(zip(FRAMINGS, column_totals)),
    }


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict:
    if len(left) != len(right) or not left:
        raise RuntimeError("McNemar comparison requires complete paired outcomes")
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    both = sum(a and b for a, b in zip(left, right))
    neither = len(left) - left_only - right_only - both
    discordant = left_only + right_only
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    differences = [int(a) - int(b) for a, b in zip(left, right)]
    risk_difference = statistics.mean(differences)
    if len(differences) > 1:
        standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    else:
        standard_error = 0.0
    return {
        "n": len(left),
        "left_only": left_only,
        "right_only": right_only,
        "both": both,
        "neither": neither,
        "discordant": discordant,
        "risk_difference_left_minus_right": risk_difference,
        "risk_difference_lower_95": max(-1.0, risk_difference - 1.96 * standard_error),
        "risk_difference_upper_95": min(1.0, risk_difference + 1.96 * standard_error),
        "risk_difference_interval": "paired Wald 95% confidence interval",
        "p_value_raw": p_value,
    }


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def prompt_without_feedback(prompt: str) -> tuple[str, str]:
    marker = "Validation feedback:\n"
    if prompt.count(marker) != 1:
        raise RuntimeError("repair prompt has an unexpected feedback block")
    prefix, remainder = prompt.split(marker, 1)
    _feedback, separator, suffix = remainder.partition("\n\nRepaired graph:")
    if not separator:
        raise RuntimeError("repair prompt feedback boundary is missing")
    return prefix, suffix


def triple(row) -> tuple[str, str, str]:
    if isinstance(row, dict):
        value = (row.get("subject"), row.get("predicate"), row.get("object"))
    else:
        value = tuple(row) if isinstance(row, (list, tuple)) else ()
    if len(value) != 3 or not all(isinstance(item, str) and item for item in value):
        raise RuntimeError(f"invalid triple: {row!r}")
    return value


def validate_metadata(metadata: dict, spec: dict, results_path: Path) -> None:
    if metadata.get("status") != "complete":
        raise RuntimeError("RQ3 run is not complete")
    if metadata.get("completed_observations") != 90:
        raise RuntimeError("RQ3 metadata does not record 90 completed observations")
    if metadata.get("repair_generations") != 90:
        raise RuntimeError("RQ3 repair generation count changed")
    if metadata.get("results_sha256") != sha256_file(results_path):
        raise RuntimeError("RQ3 result hash mismatch")
    if metadata.get("spec_sha256") != sha256_file(SPEC_PATH):
        raise RuntimeError("RQ3 specification hash mismatch")
    if metadata.get("invalid_output_retry") is not False:
        raise RuntimeError("RQ3 invalid output retry policy changed")
    if metadata.get("post_repair_measurement_visible_to_model") is not False:
        raise RuntimeError("post-repair measurements were visible to the model")
    if not metadata.get("completed_at_utc"):
        raise RuntimeError("RQ3 completion time is missing")
    model = metadata.get("repair_model", {})
    generation = spec["generation"]
    for key, expected in (
        ("name", generation["model"]),
        ("digest", generation["digest"]),
        ("options", generation["options"]),
    ):
        if model.get(key) != expected:
            raise RuntimeError(f"RQ3 repair model {key} mismatch")


def expected_feedback(case: dict, framing: str, spec: dict) -> dict:
    added = case["primary_modification"]["added"]
    if len(added) != 1:
        raise RuntimeError(f"unexpected controlled target in {case['case_id']}")
    settings = spec["feedback"]
    focus = added[0]["subject"] if framing != "verdict" else None
    message = settings["explanation" if framing == "explanation" else "verdict"]["message"]
    opaque = hashlib.sha256(case["case_id"].encode("utf-8")).hexdigest()[:20]
    return {
        "validator": settings["validator"],
        "violation_id": f"owl:inconsistent:{opaque}",
        "error_type": settings["error_type"],
        "focus": focus,
        "path": settings["path"],
        "message": message,
    }


def validate_successful_outcome(row: dict) -> None:
    outcome = row["outcome"]
    post = row.get("post_repair_measurement")
    if not isinstance(post, dict):
        raise RuntimeError("usable repair output has no post-repair measurement")
    reference = post["reference"]
    edit = post["edit"]
    expected = {
        "controlled_target_removed": post["controlled_target_removed"],
        "owl_consistent": post["owl_consistent"],
        "exact_reference_recovery": reference["reference_recovery"],
        "collateral_edit": reference["collateral_symmetric_difference"] > 0,
        "new_raw_shacl_findings": bool(post["new_raw_shacl_violation_ids"]),
        "new_grounding_findings": bool(post["grounding"]["new_actionable_violation_ids"]),
        "owl_inconsistent_after_target_removal": (
            post["controlled_target_removed"] and not post["owl_consistent"]
        ),
        "output_failure": None,
        "edit_distance_from_injected": edit["symmetric_difference_from_injected"],
        "edit_distance_from_clean_reference": reference["reference_symmetric_difference"],
    }
    if outcome != expected:
        raise RuntimeError(f"recorded outcome is inconsistent for {row['case_id']} {row['framing']}")


def validate_rows(rows: list[dict], metadata: dict, spec: dict, cases: list[dict]) -> None:
    if len(rows) != 90:
        raise RuntimeError(f"expected 90 RQ3 observations, found {len(rows)}")
    selected = [case for case in cases if case.get("condition") == "disjointness"]
    if len(selected) != 30 or len({case["case_id"] for case in selected}) != 30:
        raise RuntimeError("locked disjointness sample must contain 30 unique cases")
    case_by_id = {case["case_id"]: case for case in selected}
    if set(metadata.get("case_ids", [])) != set(case_by_id):
        raise RuntimeError("RQ3 metadata case set mismatch")
    schedule = metadata.get("execution_schedule")
    if not isinstance(schedule, list) or len(schedule) != 90:
        raise RuntimeError("RQ3 execution schedule is missing")
    seen = set()
    prompt_shells = defaultdict(set)
    initial_states = defaultdict(set)
    for row, expected in zip(rows, schedule):
        scheduled = {key: row.get(key) for key in expected}
        if scheduled != expected:
            raise RuntimeError("RQ3 rows are not in the locked execution schedule")
        key = (row["case_id"], row["framing"])
        if key in seen:
            raise RuntimeError(f"duplicate RQ3 observation: {key}")
        seen.add(key)
        if row["framing"] not in FRAMINGS or row["case_id"] not in case_by_id:
            raise RuntimeError("RQ3 row is outside the locked sample")
        repair = row.get("repair", {})
        prompt = repair.get("rendered_prompt")
        if not isinstance(prompt, str) or repair.get("rendered_prompt_sha256") != sha256_text(prompt):
            raise RuntimeError("RQ3 rendered prompt hash mismatch")
        if repair.get("feedback") != expected_feedback(case_by_id[row["case_id"]], row["framing"], spec):
            raise RuntimeError("RQ3 feedback differs from the specified framing")
        prompt_shells[row["case_id"]].add(prompt_without_feedback(prompt))
        initial_states[row["case_id"]].add(
            json.dumps(row.get("initial_measurement"), sort_keys=True, separators=(",", ":"))
        )
        if row.get("initial_measurement", {}).get("owl_consistent") is not False:
            raise RuntimeError("RQ3 initial graph is not OWL inconsistent")
        parsed = repair.get("parse", {})
        if parsed.get("ok"):
            if parsed.get("failure") is not None:
                raise RuntimeError("usable repair output records a parse failure")
            validate_successful_outcome(row)
        else:
            outcome = row.get("outcome", {})
            if row.get("post_repair_measurement") is not None:
                raise RuntimeError("failed repair output has a post-repair measurement")
            if outcome.get("output_failure") != parsed.get("failure"):
                raise RuntimeError("output failure does not match the parser result")
            if outcome.get("controlled_target_removed") is not False:
                raise RuntimeError("failed output claims controlled target removal")
            if outcome.get("exact_reference_recovery") is not False:
                raise RuntimeError("failed output claims exact reference recovery")
            for field in (
                "owl_consistent",
                "collateral_edit",
                "new_raw_shacl_findings",
                "new_grounding_findings",
                "owl_inconsistent_after_target_removal",
                "edit_distance_from_injected",
                "edit_distance_from_clean_reference",
            ):
                if outcome.get(field) is not None:
                    raise RuntimeError(f"failed output fabricates {field}")
    expected_pairs = {(case_id, framing) for case_id in case_by_id for framing in FRAMINGS}
    if seen != expected_pairs:
        raise RuntimeError("RQ3 paired matrix is incomplete")
    if any(len(values) != 1 for values in prompt_shells.values()):
        raise RuntimeError("prompt content outside feedback differs within a case")
    if any(len(values) != 1 for values in initial_states.values()):
        raise RuntimeError("initial measurement differs across framings")


def mean(values) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def median(values) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def framing_summary(rows: list[dict]) -> dict:
    n = len(rows)
    usable = [row for row in rows if row["outcome"]["output_failure"] is None]
    return {
        "n": n,
        "usable_outputs": wilson_interval(len(usable), n),
        "controlled_target_removed": wilson_interval(
            sum(bool(row["outcome"]["controlled_target_removed"]) for row in rows), n
        ),
        "owl_consistent": wilson_interval(
            sum(bool(row["outcome"]["owl_consistent"]) for row in rows), n
        ),
        "exact_reference_recovery": wilson_interval(
            sum(bool(row["outcome"]["exact_reference_recovery"]) for row in rows), n
        ),
        "output_failure": wilson_interval(n - len(usable), n),
        "output_failure_types": dict(sorted(Counter(
            row["outcome"]["output_failure"] for row in rows
            if row["outcome"]["output_failure"] is not None
        ).items())),
        "among_usable_outputs": {
            "collateral_edit": wilson_interval(
                sum(bool(row["outcome"]["collateral_edit"]) for row in usable), len(usable)
            ),
            "new_raw_shacl_findings": wilson_interval(
                sum(bool(row["outcome"]["new_raw_shacl_findings"]) for row in usable), len(usable)
            ),
            "new_grounding_findings": wilson_interval(
                sum(bool(row["outcome"]["new_grounding_findings"]) for row in usable), len(usable)
            ),
            "owl_inconsistent_after_target_removal": wilson_interval(
                sum(bool(row["outcome"]["owl_inconsistent_after_target_removal"]) for row in usable),
                len(usable),
            ),
        },
        "edit_distance_from_injected": {
            "mean": mean(row["outcome"]["edit_distance_from_injected"] for row in usable),
            "median": median(row["outcome"]["edit_distance_from_injected"] for row in usable),
        },
        "edit_distance_from_clean_reference": {
            "mean": mean(row["outcome"]["edit_distance_from_clean_reference"] for row in usable),
            "median": median(row["outcome"]["edit_distance_from_clean_reference"] for row in usable),
        },
    }


def cost_summary(rows: list[dict], metadata: dict) -> dict:
    live_grounding = {}
    for row in rows:
        post = row.get("post_repair_measurement")
        if not isinstance(post, dict):
            continue
        for judgment in post.get("grounding", {}).get("judgments", []):
            if judgment.get("source") != "repair_round":
                continue
            key = (row["case_id"], triple(judgment["triple"]))
            previous = live_grounding.get(key)
            if previous is not None and previous.get("verdict") != judgment.get("verdict"):
                raise RuntimeError("shared grounding cache contains conflicting verdicts")
            live_grounding[key] = judgment
    repair = [row["repair"] for row in rows]
    started = datetime.fromisoformat(metadata["created_at_utc"])
    completed = datetime.fromisoformat(metadata["completed_at_utc"])
    return {
        "repair_generation": {
            "calls": len(repair),
            "prompt_tokens": sum(item.get("prompt_eval_count") or 0 for item in repair),
            "generated_tokens": sum(item.get("eval_count") or 0 for item in repair),
            "recorded_duration_seconds": sum(item.get("total_duration_ns") or 0 for item in repair) / 1e9,
        },
        "live_grounding": {
            "unique_calls_within_case": len(live_grounding),
            "prompt_tokens": sum(item.get("prompt_eval_count") or 0 for item in live_grounding.values()),
            "generated_tokens": sum(item.get("eval_count") or 0 for item in live_grounding.values()),
            "recorded_duration_seconds": sum(item.get("total_duration_ns") or 0 for item in live_grounding.values()) / 1e9,
        },
        "wall_seconds": (completed - started).total_seconds(),
    }


def case_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in sorted(rows, key=lambda item: (item["case_index"], FRAMINGS.index(item["framing"]))):
        outcome = row["outcome"]
        output.append({
            "case_id": row["case_id"],
            "case_index": row["case_index"],
            "source_family": row["source_family"],
            "framing": row["framing"],
            "controlled_target_removed": outcome["controlled_target_removed"],
            "owl_consistent": outcome["owl_consistent"],
            "exact_reference_recovery": outcome["exact_reference_recovery"],
            "output_failure": outcome["output_failure"] or "",
            "collateral_edit": outcome["collateral_edit"],
            "new_raw_shacl_findings": outcome["new_raw_shacl_findings"],
            "new_grounding_findings": outcome["new_grounding_findings"],
            "edit_distance_from_injected": outcome["edit_distance_from_injected"],
            "edit_distance_from_clean_reference": outcome["edit_distance_from_clean_reference"],
        })
    return output


def analyze(rows: list[dict], metadata: dict, spec: dict) -> dict:
    grouped = {name: [row for row in rows if row["framing"] == name] for name in FRAMINGS}
    paired = defaultdict(dict)
    for row in rows:
        paired[row["case_id"]][row["framing"]] = bool(row["outcome"]["controlled_target_removed"])
    ordered_cases = sorted(paired)
    matrix = [[paired[case_id][name] for name in FRAMINGS] for case_id in ordered_cases]
    comparisons = []
    for left_name, right_name in PAIRS:
        value = exact_mcnemar(
            [paired[case_id][left_name] for case_id in ordered_cases],
            [paired[case_id][right_name] for case_id in ordered_cases],
        )
        comparisons.append({"left": left_name, "right": right_name, **value})
    adjusted = holm_adjust([row["p_value_raw"] for row in comparisons])
    for row, p_value in zip(comparisons, adjusted):
        row["p_value_holm"] = p_value
        row["reject_at_alpha_0_05"] = p_value < spec["analysis_plan"]["alpha"]
    return {
        "version": 1,
        "created_at_utc": datetime.now().astimezone().isoformat(),
        "integrity": {
            "observations": len(rows),
            "paired_cases": len(paired),
            "observations_per_framing": {name: len(grouped[name]) for name in FRAMINGS},
            "results_sha256": metadata["results_sha256"],
            "metadata_sha256": None,
            "spec_sha256": metadata["spec_sha256"],
            "run_git_head": metadata["git_head"],
            "complete_paired_matrix": all(set(value) == set(FRAMINGS) for value in paired.values()),
        },
        "primary_outcome": {
            "name": "controlled_target_removed",
            "by_framing": {name: framing_summary(grouped[name])["controlled_target_removed"] for name in FRAMINGS},
            "omnibus": cochran_q(matrix),
            "pairwise": comparisons,
            "multiplicity_correction": "Holm correction across three exact two-sided McNemar tests",
        },
        "secondary_outcomes": {name: framing_summary(grouped[name]) for name in FRAMINGS},
        "cost": cost_summary(rows, metadata),
        "models": {
            "repair": metadata["repair_model"],
            "grounding": metadata["grounding_model"],
        },
        "claim_boundaries": spec["claim_boundaries"],
        "execution": {
            "models_or_validators_run": False,
            "preliminary_results_pooled": False,
        },
    }


def run(results_path: Path, metadata_path: Path, output_path: Path, cases_csv_path: Path) -> dict:
    spec = read_json(SPEC_PATH)
    if spec.get("version") != 1:
        raise RuntimeError("unsupported RQ3 specification version")
    if sha256_file(CASES_PATH) != spec["sample"]["cases_sha256"]:
        raise RuntimeError("RQ3 case file hash mismatch")
    rows = read_jsonl(results_path)
    metadata = read_json(metadata_path)
    cases = read_jsonl(CASES_PATH)
    validate_metadata(metadata, spec, results_path)
    validate_rows(rows, metadata, spec, cases)
    payload = analyze(rows, metadata, spec)
    # Preserve the actual paths used when tests or external analysis supply alternatives.
    payload["integrity"]["metadata_sha256"] = sha256_file(metadata_path)
    write_json(output_path, payload)
    case_output = case_rows(rows)
    write_csv(cases_csv_path, case_output, list(case_output[0]))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--cases-csv", type=Path, default=DEFAULT_CASES_CSV)
    args = parser.parse_args()
    payload = run(args.results, args.metadata, args.output, args.cases_csv)
    primary = payload["primary_outcome"]
    print(f"paired cases analyzed: {payload['integrity']['paired_cases']}")
    for framing in FRAMINGS:
        value = primary["by_framing"][framing]
        failure = payload["secondary_outcomes"][framing]["output_failure"]
        print(
            f"{framing}: target {value['count']}/{value['n']}, "
            f"output failure {failure['count']}/{failure['n']}"
        )
    omnibus = primary["omnibus"]
    print(f"Cochran Q: {omnibus['statistic']:.6g}, p={omnibus['p_value']:.6g}")
    print(f"analysis: {args.output.resolve()}")
    print("No model, grounding assessor, validator, reasoner, or repair was executed.")


if __name__ == "__main__":
    main()
