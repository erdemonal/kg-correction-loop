#!/usr/bin/env python3
"""Analyze the completed SOSA and SSN confirmatory grounding and repair runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "experiments" / "sosa_ssn_confirmatory_cases.jsonl"
RUNNER_SPEC_PATH = ROOT / "experiments" / "sosa_ssn_confirmatory_runner_spec.json"

GROUNDING_NAME = "sosa_ssn_confirmatory_grounding.jsonl"
GROUNDING_META_NAME = "sosa_ssn_confirmatory_grounding.meta.json"
REPAIR_NAME = "sosa_ssn_confirmatory_repair.jsonl"
REPAIR_META_NAME = "sosa_ssn_confirmatory_repair.meta.json"

ANALYSIS_NAME = "sosa_ssn_confirmatory_analysis.json"
CASES_CSV_NAME = "sosa_ssn_confirmatory_analysis_cases.csv"
ROUNDS_CSV_NAME = "sosa_ssn_confirmatory_analysis_rounds.csv"

CONDITIONS = (
    "cardinality",
    "disjointness",
    "domain_range",
    "functional_property_conflict",
    "grounding",
    "temporal",
)

STOP_REASONS = {
    "validated",
    "stalled",
    "oscillation",
    "max_rounds",
    "no_feedback",
    "output_failure",
}

VALIDATORS = {"raw_shacl", "owl_consistency", "grounding_v3"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def index_unique(rows: list[dict], label: str) -> dict[str, dict]:
    output = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"{label}: row has no case_id")
        if case_id in output:
            raise RuntimeError(f"{label}: duplicate case_id {case_id}")
        output[case_id] = row
    return output


def triple_key(row) -> tuple[str, str, str]:
    if isinstance(row, dict):
        value = (row.get("subject"), row.get("predicate"), row.get("object"))
    else:
        value = tuple(row) if isinstance(row, (list, tuple)) else ()
    if len(value) != 3 or not all(isinstance(item, str) and item for item in value):
        raise RuntimeError(f"invalid triple: {row!r}")
    return value


def triple_set(rows) -> set[tuple[str, str, str]]:
    return {triple_key(row) for row in rows}


def mean(values) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def median(values) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def wilson_interval(count: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"count": count, "n": n, "rate": None, "lower_95": None, "upper_95": None}
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


def prf(true_positive: int, false_positive: int, false_negative: int) -> dict:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else (1.0 if false_negative == 0 else 0.0)
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else (1.0 if false_positive == 0 else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def scores_from_round(round_row: dict) -> dict:
    triples = round_row.get("triples")
    validation = round_row.get("validation")
    if not isinstance(triples, list) or not isinstance(validation, dict):
        raise RuntimeError("cannot score a round without a validated graph")
    reference = validation.get("reference")
    if not isinstance(reference, dict):
        raise RuntimeError("validated round has no reference comparison")
    predicted = triple_set(triples)
    extra = triple_set(reference.get("new_not_in_clean_reference", []))
    missing = triple_set(reference.get("clean_reference_removed", []))
    if not extra <= predicted:
        raise RuntimeError("recorded extra triples are not in the graph")
    if predicted & missing:
        raise RuntimeError("recorded missing reference triple remains in the graph")
    result = prf(len(predicted - extra), len(extra), len(missing))
    result.update(
        {
            "graph_size": len(predicted),
            "reference_size": len(predicted - extra) + len(missing),
            "reference_recovery": bool(reference.get("reference_recovery")),
            "reference_difference": reference.get("reference_symmetric_difference"),
            "collateral_difference": reference.get("collateral_symmetric_difference"),
            "collateral_added": len(reference.get("collateral_added") or []),
            "collateral_removed": len(reference.get("collateral_removed") or []),
            "extra_triples": len(extra),
            "missing_reference_triples": len(missing),
        }
    )
    if result["reference_difference"] != len(extra) + len(missing):
        raise RuntimeError("reference symmetric difference is inconsistent")
    if result["reference_recovery"] != (not extra and not missing):
        raise RuntimeError("reference recovery flag is inconsistent")
    return result


def validate_metadata(
    metadata: dict,
    results_path: Path,
    runner_spec_path: Path,
    expected_cases: int,
    label: str,
) -> None:
    if metadata.get("status") != "complete":
        raise RuntimeError(f"{label}: run is not complete")
    if metadata.get("case_count") != expected_cases:
        raise RuntimeError(f"{label}: wrong case_count")
    if metadata.get("completed_case_count") != expected_cases:
        raise RuntimeError(f"{label}: wrong completed_case_count")
    if metadata.get("results_sha256") != sha256_file(results_path):
        raise RuntimeError(f"{label}: result hash mismatch")
    if metadata.get("runner_spec_sha256") != sha256_file(runner_spec_path):
        raise RuntimeError(f"{label}: runner spec hash mismatch")
    if not metadata.get("completed_at_utc"):
        raise RuntimeError(f"{label}: missing completion time")


def validate_grounding_row(case: dict, row: dict, spec: dict) -> None:
    case_id = case["case_id"]
    if row.get("case_id") != case_id:
        raise RuntimeError(f"{case_id}: grounding case id mismatch")
    for field in ("condition", "source_family", "source_text_sha256"):
        if row.get(field) != case.get(field):
            raise RuntimeError(f"{case_id}: grounding {field} mismatch")
    clean = triple_set(case["clean_content_triples"])
    injected = triple_set(case["injected_content_triples"])
    union = clean | injected
    judgments = row.get("judgments")
    if not isinstance(judgments, list):
        raise RuntimeError(f"{case_id}: no grounding judgments")
    judged = [triple_key(item.get("triple")) for item in judgments]
    if len(judged) != len(set(judged)) or set(judged) != union:
        raise RuntimeError(f"{case_id}: grounding union coverage mismatch")
    verdict = {triple_key(item["triple"]): item.get("verdict") for item in judgments}
    if set(verdict.values()) - {"SUPPORTED", "UNSUPPORTED"}:
        raise RuntimeError(f"{case_id}: invalid grounding verdict")
    for state, expected in (("clean", clean), ("injected", injected)):
        payload = row.get(state)
        if not isinstance(payload, dict):
            raise RuntimeError(f"{case_id}: missing {state} grounding summary")
        state_judged = {triple_key(item["triple"]) for item in payload.get("judgments", [])}
        unsupported = sum(verdict[value] == "UNSUPPORTED" for value in expected)
        if state_judged != expected:
            raise RuntimeError(f"{case_id}: {state} grounding coverage mismatch")
        if payload.get("triple_count") != len(expected):
            raise RuntimeError(f"{case_id}: {state} triple count mismatch")
        if payload.get("unsupported_count") != unsupported:
            raise RuntimeError(f"{case_id}: {state} unsupported count mismatch")
        if payload.get("grounding_error") != bool(unsupported):
            raise RuntimeError(f"{case_id}: {state} error flag mismatch")
    target = row.get("target")
    target_triples = triple_set(case["primary_modification"]["added"])
    expected_error = bool(spec["grounding"]["expected_target_error"][case["condition"]])
    observed_error = any(verdict[value] == "UNSUPPORTED" for value in target_triples)
    if triple_set(target.get("triples", [])) != target_triples:
        raise RuntimeError(f"{case_id}: grounding target mismatch")
    if target.get("expected_grounding_error") != expected_error:
        raise RuntimeError(f"{case_id}: expected grounding target mismatch")
    if target.get("observed_grounding_error") != observed_error:
        raise RuntimeError(f"{case_id}: observed grounding target mismatch")
    if target.get("matches_expected") != (observed_error == expected_error):
        raise RuntimeError(f"{case_id}: grounding match flag mismatch")


def feedback_lookup(round_row: dict) -> dict[str, dict]:
    validation = round_row.get("validation")
    if not isinstance(validation, dict):
        return {}
    output = {}
    for item in validation.get("actionable_feedback", []):
        violation_id = item.get("violation_id")
        if not isinstance(violation_id, str) or not violation_id:
            raise RuntimeError("feedback item has no violation_id")
        if violation_id in output:
            raise RuntimeError("duplicate feedback violation_id")
        output[violation_id] = item
    return output


def validate_repair_row(case: dict, row: dict) -> None:
    case_id = case["case_id"]
    if row.get("case_id") != case_id:
        raise RuntimeError(f"{case_id}: repair case id mismatch")
    for field in ("condition", "source_family"):
        if row.get(field) != case.get(field):
            raise RuntimeError(f"{case_id}: repair {field} mismatch")
    rounds = row.get("rounds")
    final = row.get("final")
    if not isinstance(rounds, list) or not rounds or not isinstance(final, dict):
        raise RuntimeError(f"{case_id}: incomplete repair trajectory")
    if [item.get("round") for item in rounds] != list(range(len(rounds))):
        raise RuntimeError(f"{case_id}: nonsequential rounds")
    if not isinstance(rounds[0].get("validation"), dict):
        raise RuntimeError(f"{case_id}: round 0 is not validated")
    if rounds[0]["validation"].get("target_resolved") is not False:
        raise RuntimeError(f"{case_id}: injected target begins resolved")
    if final.get("repair_rounds") != len(rounds) - 1:
        raise RuntimeError(f"{case_id}: repair round count mismatch")
    if final.get("stop_reason") not in STOP_REASONS:
        raise RuntimeError(f"{case_id}: invalid stop reason")
    initial_feedback = rounds[0]["validation"].get("actionable_feedback")
    if not isinstance(initial_feedback, list):
        raise RuntimeError(f"{case_id}: no initial feedback list")
    sources = sorted({item.get("validator") for item in initial_feedback})
    if set(sources) - VALIDATORS:
        raise RuntimeError(f"{case_id}: unknown initial feedback source")
    if row.get("received_initial_feedback") != bool(initial_feedback):
        raise RuntimeError(f"{case_id}: feedback flag mismatch")
    if row.get("initial_feedback_sources") != sources:
        raise RuntimeError(f"{case_id}: feedback source mismatch")
    valid_rounds = []
    for index, round_row in enumerate(rounds):
        validation = round_row.get("validation")
        triples = round_row.get("triples")
        if index > 0 and not isinstance(round_row.get("repair"), dict):
            raise RuntimeError(f"{case_id}: repair round has no generation")
        if isinstance(validation, dict):
            if not isinstance(triples, list):
                raise RuntimeError(f"{case_id}: validated round has no graph")
            scores_from_round(round_row)
            valid_rounds.append(round_row)
            lookup = feedback_lookup(round_row)
            for violation_id in round_row.get("new_violation_ids", []):
                if violation_id not in lookup:
                    raise RuntimeError(f"{case_id}: new violation not in feedback")
        elif triples is not None:
            raise RuntimeError(f"{case_id}: unvalidated round contains a graph")
    resolution_rounds = [
        item["round"]
        for item in valid_rounds
        if item["round"] > 0 and item["validation"].get("target_resolved") is True
    ]
    first_resolution = min(resolution_rounds) if resolution_rounds else None
    if final.get("rounds_to_resolution") != first_resolution:
        raise RuntimeError(f"{case_id}: first resolution round mismatch")
    last_valid = valid_rounds[-1]
    last_scores = scores_from_round(last_valid)
    if final["stop_reason"] == "output_failure":
        if not isinstance(final.get("output_failure"), str):
            raise RuntimeError(f"{case_id}: output failure has no type")
        if rounds[-1].get("validation") is not None:
            raise RuntimeError(f"{case_id}: output failure round was validated")
        if final.get("target_resolved") or final.get("validated_state") or final.get("reference_recovery"):
            raise RuntimeError(f"{case_id}: output failure recorded as successful")
    else:
        if final.get("output_failure") is not None:
            raise RuntimeError(f"{case_id}: nonfailure has output error")
        if final.get("target_resolved") != bool(last_valid["validation"]["target_resolved"]):
            raise RuntimeError(f"{case_id}: final target differs from final graph")
        if final.get("reference_recovery") != last_scores["reference_recovery"]:
            raise RuntimeError(f"{case_id}: final reference recovery mismatch")


def round_cost(round_row: dict, seen_grounding: set[tuple[str, str, str]]) -> dict:
    repair = round_row.get("repair")
    if isinstance(repair, dict):
        for field in ("prompt_eval_count", "eval_count", "total_duration_ns"):
            if repair.get(field) is None:
                raise RuntimeError(f"repair generation has no {field}")
        parse = repair.get("parse") or {}
        repair_values = {
            "repair_calls": 1,
            "repair_prompt_tokens": repair["prompt_eval_count"],
            "repair_generated_tokens": repair["eval_count"],
            "repair_duration_ns": repair["total_duration_ns"],
            "parse_ok": bool(parse.get("ok")),
            "parse_failure": parse.get("failure") or "",
        }
    else:
        repair_values = {
            "repair_calls": 0,
            "repair_prompt_tokens": 0,
            "repair_generated_tokens": 0,
            "repair_duration_ns": 0,
            "parse_ok": True,
            "parse_failure": "",
        }
    calls = prompt = generated = duration = 0
    validation = round_row.get("validation")
    if isinstance(validation, dict):
        for judgment in validation.get("grounding", {}).get("judgments", []):
            if judgment.get("source") != "repair_round":
                continue
            key = triple_key(judgment.get("triple"))
            if key in seen_grounding:
                continue
            seen_grounding.add(key)
            for field in ("prompt_eval_count", "eval_count", "total_duration_ns"):
                if judgment.get(field) is None:
                    raise RuntimeError(f"live grounding judgment has no {field}")
            calls += 1
            prompt += judgment["prompt_eval_count"]
            generated += judgment["eval_count"]
            duration += judgment["total_duration_ns"]
    return {
        **repair_values,
        "live_grounding_calls": calls,
        "live_grounding_prompt_tokens": prompt,
        "live_grounding_generated_tokens": generated,
        "live_grounding_duration_ns": duration,
    }


def new_violation_summary(row: dict) -> Counter:
    found = {}
    for round_row in row["rounds"][1:]:
        lookup = feedback_lookup(round_row)
        for violation_id in round_row.get("new_violation_ids", []):
            item = lookup.get(violation_id)
            if item is None:
                raise RuntimeError(f"{row['case_id']}: new violation missing from feedback")
            found.setdefault(violation_id, item["validator"])
    return Counter(found.values())


def analyze_case(
    case: dict,
    grounding: dict,
    repair: dict,
    spec: dict,
) -> tuple[dict, list[dict]]:
    validate_grounding_row(case, grounding, spec)
    validate_repair_row(case, repair)
    valid_rounds = [item for item in repair["rounds"] if isinstance(item.get("validation"), dict)]
    initial = valid_rounds[0]
    last_valid = valid_rounds[-1]
    initial_scores = scores_from_round(initial)
    last_scores = scores_from_round(last_valid)
    final = repair["final"]
    ever_resolved = final["rounds_to_resolution"] is not None
    last_target = bool(last_valid["validation"]["target_resolved"])
    new_violations = new_violation_summary(repair)
    collateral_values = [
        scores_from_round(item)["collateral_difference"]
        for item in valid_rounds
        if item["round"] > 0
    ]
    seen_grounding = set()
    analyzed_rounds = []
    totals = Counter()
    previous_graph = None
    for round_row in repair["rounds"]:
        cost = round_cost(round_row, seen_grounding)
        totals.update({key: value for key, value in cost.items() if isinstance(value, int)})
        validation = round_row.get("validation")
        has_graph = isinstance(validation, dict) and isinstance(round_row.get("triples"), list)
        if has_graph:
            scores = scores_from_round(round_row)
            current_graph = triple_set(round_row["triples"])
            edit_distance = (
                len(current_graph ^ previous_graph) if previous_graph is not None else None
            )
            previous_graph = current_graph
            grounding_payload = validation.get("grounding", {})
            analyzed_rounds.append(
                {
                    "case_id": case["case_id"],
                    "condition": case["condition"],
                    "source_family": case["source_family"],
                    "round": round_row["round"],
                    "has_validated_graph": True,
                    "parse_ok": cost["parse_ok"],
                    "parse_failure": cost["parse_failure"],
                    "target_resolved": bool(validation["target_resolved"]),
                    "reference_recovery": scores["reference_recovery"],
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                    "reference_difference": scores["reference_difference"],
                    "collateral_difference": scores["collateral_difference"],
                    "edit_distance_from_previous_graph": edit_distance,
                    "actionable_feedback_count": len(validation["actionable_feedback"]),
                    "new_violation_count": len(round_row.get("new_violation_ids", [])),
                    "shacl_violation_count": len(validation["symbolic"]["shacl"]["violations"]),
                    "owl_consistent": bool(validation["symbolic"]["owl_consistent"]),
                    "grounding_unsupported_count": sum(
                        item["verdict"] == "UNSUPPORTED"
                        for item in grounding_payload.get("judgments", [])
                    ),
                    **{key: value for key, value in cost.items() if key not in {"parse_ok", "parse_failure"}},
                }
            )
        else:
            analyzed_rounds.append(
                {
                    "case_id": case["case_id"],
                    "condition": case["condition"],
                    "source_family": case["source_family"],
                    "round": round_row["round"],
                    "has_validated_graph": False,
                    "parse_ok": cost["parse_ok"],
                    "parse_failure": cost["parse_failure"],
                    "target_resolved": "",
                    "reference_recovery": "",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "reference_difference": "",
                    "collateral_difference": "",
                    "edit_distance_from_previous_graph": "",
                    "actionable_feedback_count": "",
                    "new_violation_count": "",
                    "shacl_violation_count": "",
                    "owl_consistent": "",
                    "grounding_unsupported_count": "",
                    **{key: value for key, value in cost.items() if key not in {"parse_ok", "parse_failure"}},
                }
            )
    target = grounding["target"]
    f1_delta = last_scores["f1"] - initial_scores["f1"]
    case_row = {
        "case_id": case["case_id"],
        "condition": case["condition"],
        "source_family": case["source_family"],
        "grounding_expected_target_error": bool(target["expected_grounding_error"]),
        "grounding_observed_target_error": bool(target["observed_grounding_error"]),
        "grounding_target_matches_expected": bool(target["matches_expected"]),
        "clean_grounding_error": bool(grounding["clean"]["grounding_error"]),
        "clean_grounding_unsupported_count": grounding["clean"]["unsupported_count"],
        "injected_grounding_error": bool(grounding["injected"]["grounding_error"]),
        "injected_grounding_unsupported_count": grounding["injected"]["unsupported_count"],
        "received_initial_feedback": bool(repair["received_initial_feedback"]),
        "initial_feedback_sources": "+".join(repair["initial_feedback_sources"]),
        "stop_reason": final["stop_reason"],
        "repair_rounds": final["repair_rounds"],
        "end_to_end_target_resolved": bool(final["target_resolved"]),
        "ever_target_resolved": ever_resolved,
        "first_resolution_round": final["rounds_to_resolution"] if ever_resolved else "",
        "last_validated_round": last_valid["round"],
        "last_validated_target_resolved": last_target,
        "target_regressed_after_resolution": ever_resolved and not last_target,
        "output_failure_after_resolution": ever_resolved and final["stop_reason"] == "output_failure",
        "validated_state": bool(final["validated_state"]),
        "end_to_end_reference_recovery": bool(final["reference_recovery"]),
        "last_validated_reference_recovery": last_scores["reference_recovery"],
        "output_failure": final["output_failure"] or "",
        "initial_precision": initial_scores["precision"],
        "initial_recall": initial_scores["recall"],
        "initial_f1": initial_scores["f1"],
        "last_validated_precision": last_scores["precision"],
        "last_validated_recall": last_scores["recall"],
        "last_validated_f1": last_scores["f1"],
        "f1_delta": f1_delta,
        "f1_change": "improved" if f1_delta > 0 else "worsened" if f1_delta < 0 else "unchanged",
        "initial_reference_difference": initial_scores["reference_difference"],
        "last_validated_reference_difference": last_scores["reference_difference"],
        "last_validated_collateral_difference": last_scores["collateral_difference"],
        "any_collateral_edit": any(value > 0 for value in collateral_values),
        "peak_collateral_difference": max(collateral_values) if collateral_values else 0,
        "any_new_violation": bool(new_violations),
        "distinct_new_violation_count": sum(new_violations.values()),
        "new_shacl_violation_count": new_violations["raw_shacl"],
        "new_owl_violation_count": new_violations["owl_consistency"],
        "new_grounding_violation_count": new_violations["grounding_v3"],
        "repair_calls": totals["repair_calls"],
        "repair_prompt_tokens": totals["repair_prompt_tokens"],
        "repair_generated_tokens": totals["repair_generated_tokens"],
        "repair_duration_ns": totals["repair_duration_ns"],
        "live_grounding_calls": totals["live_grounding_calls"],
        "live_grounding_prompt_tokens": totals["live_grounding_prompt_tokens"],
        "live_grounding_generated_tokens": totals["live_grounding_generated_tokens"],
        "live_grounding_duration_ns": totals["live_grounding_duration_ns"],
    }
    return case_row, analyzed_rounds


def confusion(rows: list[dict]) -> dict:
    tp = sum(row["grounding_expected_target_error"] and row["grounding_observed_target_error"] for row in rows)
    fn = sum(row["grounding_expected_target_error"] and not row["grounding_observed_target_error"] for row in rows)
    fp = sum(not row["grounding_expected_target_error"] and row["grounding_observed_target_error"] for row in rows)
    tn = sum(not row["grounding_expected_target_error"] and not row["grounding_observed_target_error"] for row in rows)
    return {
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "accuracy": (tp + tn) / len(rows) if rows else None,
    }


def grounding_group_summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "target_confusion": confusion(rows),
        "target_matches_expected": wilson_interval(
            sum(row["grounding_target_matches_expected"] for row in rows), len(rows)
        ),
        "expected_target_error": sum(row["grounding_expected_target_error"] for row in rows),
        "observed_target_error": sum(row["grounding_observed_target_error"] for row in rows),
        "clean_graph_flagged": wilson_interval(sum(row["clean_grounding_error"] for row in rows), len(rows)),
        "injected_graph_flagged": wilson_interval(sum(row["injected_grounding_error"] for row in rows), len(rows)),
        "mean_clean_unsupported_assertions": mean(row["clean_grounding_unsupported_count"] for row in rows),
        "mean_injected_unsupported_assertions": mean(row["injected_grounding_unsupported_count"] for row in rows),
    }


def validator_coverage_summary(rows: list[dict]) -> dict:
    patterns = Counter(row["initial_feedback_sources"] or "none" for row in rows)
    return {
        "n": len(rows),
        "raw_shacl": wilson_interval(
            sum("raw_shacl" in row["initial_feedback_sources"].split("+") for row in rows),
            len(rows),
        ),
        "owl_consistency": wilson_interval(
            sum("owl_consistency" in row["initial_feedback_sources"].split("+") for row in rows),
            len(rows),
        ),
        "grounding_v3": wilson_interval(
            sum("grounding_v3" in row["initial_feedback_sources"].split("+") for row in rows),
            len(rows),
        ),
        "overlap_patterns": dict(sorted(patterns.items())),
    }


def change_counts(rows: list[dict]) -> dict:
    counts = Counter(row["f1_change"] for row in rows)
    return {"improved": counts["improved"], "unchanged": counts["unchanged"], "worsened": counts["worsened"]}


def repair_group_summary(rows: list[dict]) -> dict:
    ever = [row for row in rows if row["ever_target_resolved"]]
    return {
        "n": len(rows),
        "received_initial_feedback": wilson_interval(sum(row["received_initial_feedback"] for row in rows), len(rows)),
        "end_to_end_target_resolution": wilson_interval(sum(row["end_to_end_target_resolved"] for row in rows), len(rows)),
        "ever_target_resolution": wilson_interval(sum(row["ever_target_resolved"] for row in rows), len(rows)),
        "last_validated_target_resolution": wilson_interval(sum(row["last_validated_target_resolved"] for row in rows), len(rows)),
        "validated_state": wilson_interval(sum(row["validated_state"] for row in rows), len(rows)),
        "end_to_end_reference_recovery": wilson_interval(sum(row["end_to_end_reference_recovery"] for row in rows), len(rows)),
        "last_validated_reference_recovery": wilson_interval(sum(row["last_validated_reference_recovery"] for row in rows), len(rows)),
        "output_failure": wilson_interval(sum(bool(row["output_failure"]) for row in rows), len(rows)),
        "target_regression_given_ever_resolved": wilson_interval(
            sum(row["target_regressed_after_resolution"] for row in ever), len(ever)
        ),
        "output_failure_given_ever_resolved": wilson_interval(
            sum(row["output_failure_after_resolution"] for row in ever), len(ever)
        ),
        "any_collateral_edit": wilson_interval(sum(row["any_collateral_edit"] for row in rows), len(rows)),
        "last_validated_collateral_edit": wilson_interval(
            sum(row["last_validated_collateral_difference"] > 0 for row in rows), len(rows)
        ),
        "any_new_violation": wilson_interval(sum(row["any_new_violation"] for row in rows), len(rows)),
        "mean_repair_rounds": mean(row["repair_rounds"] for row in rows),
        "median_repair_rounds": median(row["repair_rounds"] for row in rows),
        "stop_reasons": dict(sorted(Counter(row["stop_reason"] for row in rows).items())),
        "output_failure_types": dict(sorted(Counter(row["output_failure"] for row in rows if row["output_failure"]).items())),
        "first_resolution_rounds": dict(
            sorted(Counter(str(row["first_resolution_round"]) if row["first_resolution_round"] != "" else "never" for row in rows).items())
        ),
        "mean_initial_f1": mean(row["initial_f1"] for row in rows),
        "mean_last_validated_f1": mean(row["last_validated_f1"] for row in rows),
        "mean_paired_f1_change": mean(row["f1_delta"] for row in rows),
        "median_paired_f1_change": median(row["f1_delta"] for row in rows),
        "paired_f1_changes": change_counts(rows),
        "mean_last_validated_reference_difference": mean(row["last_validated_reference_difference"] for row in rows),
        "mean_last_validated_collateral_difference": mean(row["last_validated_collateral_difference"] for row in rows),
        "distinct_new_violations": sum(row["distinct_new_violation_count"] for row in rows),
    }


def wall_seconds(metadata: dict) -> float:
    start = datetime.fromisoformat(metadata["created_at_utc"])
    end = datetime.fromisoformat(metadata["completed_at_utc"])
    return (end - start).total_seconds()


def initial_grounding_cost(grounding_rows: list[dict]) -> dict:
    judgments = [item for row in grounding_rows for item in row["judgments"]]
    return {
        "calls": len(judgments),
        "prompt_tokens": sum(item["prompt_eval_count"] for item in judgments),
        "generated_tokens": sum(item["eval_count"] for item in judgments),
        "duration_ns": sum(item["total_duration_ns"] for item in judgments),
    }


def analysis_payload(
    case_rows: list[dict],
    grounding_rows: list[dict],
    grounding_meta: dict,
    repair_meta: dict,
    input_paths: dict[str, Path],
) -> dict:
    by_condition = {
        condition: [row for row in case_rows if row["condition"] == condition]
        for condition in CONDITIONS
    }
    by_source = defaultdict(list)
    for row in case_rows:
        by_source[row["source_family"]].append(row)
    initial_cost = initial_grounding_cost(grounding_rows)
    repair_cost = {
        "repair_calls": sum(row["repair_calls"] for row in case_rows),
        "repair_prompt_tokens": sum(row["repair_prompt_tokens"] for row in case_rows),
        "repair_generated_tokens": sum(row["repair_generated_tokens"] for row in case_rows),
        "repair_duration_ns": sum(row["repair_duration_ns"] for row in case_rows),
        "live_grounding_calls": sum(row["live_grounding_calls"] for row in case_rows),
        "live_grounding_prompt_tokens": sum(row["live_grounding_prompt_tokens"] for row in case_rows),
        "live_grounding_generated_tokens": sum(row["live_grounding_generated_tokens"] for row in case_rows),
        "live_grounding_duration_ns": sum(row["live_grounding_duration_ns"] for row in case_rows),
    }
    repair_cost["recorded_model_duration_ns"] = repair_cost["repair_duration_ns"] + repair_cost["live_grounding_duration_ns"]
    return {
        "version": 1,
        "analysis_git_head": git_head(),
        "scope": (
            "Controlled confirmatory characterization of the locked 180-case SOSA and SSN sample. "
            "Source-family summaries are descriptive and do not establish ecosystem-wide generalization."
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "integrity": {
            "cases": len(case_rows),
            "unique_case_ids": len({row["case_id"] for row in case_rows}),
            "cases_per_condition": dict(sorted(Counter(row["condition"] for row in case_rows).items())),
            "source_families": dict(sorted(Counter(row["source_family"] for row in case_rows).items())),
            "run_git_head": repair_meta["git_head"],
            "audited_commit": repair_meta["audited_commit"],
            "grounding_and_repair_git_head_match": grounding_meta["git_head"] == repair_meta["git_head"],
            "grounding_hash_bound_into_repair": repair_meta["grounding_results_sha256"] == grounding_meta["results_sha256"],
        },
        "grounding": {
            "interpretation": (
                "Grounding target outcomes are assessed against controlled injection metadata. "
                "Clean-graph flags are retained as assessor behavior and are not human-adjudicated false positives."
            ),
            "overall": grounding_group_summary(case_rows),
            "by_condition": {condition: grounding_group_summary(rows) for condition, rows in by_condition.items()},
            "by_source_family_descriptive_only": {
                source: grounding_group_summary(rows) for source, rows in sorted(by_source.items())
            },
            "initial_cost": {
                **initial_cost,
                "duration_seconds": initial_cost["duration_ns"] / 1_000_000_000,
                "wall_seconds": wall_seconds(grounding_meta),
            },
        },
        "validator_coverage_at_round_zero": {
            "overall": validator_coverage_summary(case_rows),
            "by_condition": {
                condition: validator_coverage_summary(rows)
                for condition, rows in by_condition.items()
            },
        },
        "repair": {
            "interpretation": (
                "End-to-end outcome, ever resolved, last validated state, exact reference recovery, "
                "and output failure are separate measures."
            ),
            "overall": repair_group_summary(case_rows),
            "by_condition": {condition: repair_group_summary(rows) for condition, rows in by_condition.items()},
            "by_source_family_descriptive_only": {
                source: repair_group_summary(rows) for source, rows in sorted(by_source.items())
            },
            "cost": {
                **repair_cost,
                "repair_duration_seconds": repair_cost["repair_duration_ns"] / 1_000_000_000,
                "live_grounding_duration_seconds": repair_cost["live_grounding_duration_ns"] / 1_000_000_000,
                "recorded_model_duration_seconds": repair_cost["recorded_model_duration_ns"] / 1_000_000_000,
                "wall_seconds": wall_seconds(repair_meta),
                "monetary_cost": None,
            },
        },
    }


def run_analysis(results_dir: Path, output_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    paths = {
        "cases": CASES_PATH,
        "runner_spec": RUNNER_SPEC_PATH,
        "grounding_results": results_dir / GROUNDING_NAME,
        "grounding_metadata": results_dir / GROUNDING_META_NAME,
        "repair_results": results_dir / REPAIR_NAME,
        "repair_metadata": results_dir / REPAIR_META_NAME,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing {name}: {path}")
    spec = read_json(RUNNER_SPEC_PATH)
    cases = read_jsonl(CASES_PATH)
    grounding_rows = read_jsonl(paths["grounding_results"])
    repair_rows = read_jsonl(paths["repair_results"])
    grounding_meta = read_json(paths["grounding_metadata"])
    repair_meta = read_json(paths["repair_metadata"])
    expected_order = [row["case_id"] for row in sorted(cases, key=lambda row: (row["condition"], row["case_id"]))]
    if [row.get("case_id") for row in grounding_rows] != expected_order:
        raise RuntimeError("grounding results are not the complete fixed-order run")
    if [row.get("case_id") for row in repair_rows] != expected_order:
        raise RuntimeError("repair results are not the complete fixed-order run")
    validate_metadata(grounding_meta, paths["grounding_results"], RUNNER_SPEC_PATH, len(cases), "grounding")
    validate_metadata(repair_meta, paths["repair_results"], RUNNER_SPEC_PATH, len(cases), "repair")
    for item in spec["inputs"].values():
        frozen_path = ROOT / item["path"]
        if not frozen_path.is_file() or sha256_file(frozen_path) != item["sha256"]:
            raise RuntimeError(f"frozen runner input changed: {item['path']}")
    if grounding_meta.get("git_head") != repair_meta.get("git_head"):
        raise RuntimeError("grounding and repair used different execution commits")
    if grounding_meta.get("audited_commit") != repair_meta.get("audited_commit"):
        raise RuntimeError("grounding and repair used different audited commits")
    if repair_meta.get("grounding_results_sha256") != grounding_meta.get("results_sha256"):
        raise RuntimeError("repair metadata is not bound to this grounding run")
    if grounding_meta.get("model", {}).get("digest") != spec["models"]["grounding"]["digest"]:
        raise RuntimeError("grounding model digest mismatch")
    if repair_meta.get("repair_model", {}).get("digest") != spec["models"]["repair"]["digest"]:
        raise RuntimeError("repair model digest mismatch")
    if repair_meta.get("grounding_model", {}).get("digest") != spec["models"]["grounding"]["digest"]:
        raise RuntimeError("repair-time grounding model digest mismatch")
    if grounding_meta.get("model", {}).get("options") != spec["models"]["grounding"]["options"]:
        raise RuntimeError("grounding model options mismatch")
    if repair_meta.get("repair_model", {}).get("options") != spec["models"]["repair"]["options"]:
        raise RuntimeError("repair model options mismatch")
    if repair_meta.get("grounding_model", {}).get("options") != spec["models"]["grounding"]["options"]:
        raise RuntimeError("repair-time grounding options mismatch")
    if grounding_meta.get("prompt") != spec["inputs"]["grounding_prompt"]:
        raise RuntimeError("grounding prompt metadata mismatch")
    if repair_meta.get("repair_prompt") != spec["inputs"]["repair_prompt"]:
        raise RuntimeError("repair prompt metadata mismatch")
    if repair_meta.get("grounding_prompt") != spec["inputs"]["grounding_prompt"]:
        raise RuntimeError("repair-time grounding prompt metadata mismatch")
    case_index = index_unique(cases, "cases")
    grounding_index = index_unique(grounding_rows, "grounding")
    repair_index = index_unique(repair_rows, "repair")
    if set(case_index) != set(grounding_index) or set(case_index) != set(repair_index):
        raise RuntimeError("case IDs differ across frozen cases and results")
    case_rows = []
    round_rows = []
    for case_id in expected_order:
        case_row, rows = analyze_case(
            case_index[case_id],
            grounding_index[case_id],
            repair_index[case_id],
            spec,
        )
        case_rows.append(case_row)
        round_rows.extend(rows)
    payload = analysis_payload(case_rows, grounding_rows, grounding_meta, repair_meta, paths)
    write_json(output_dir / ANALYSIS_NAME, payload)
    write_csv(output_dir / CASES_CSV_NAME, case_rows, list(case_rows[0]))
    write_csv(output_dir / ROUNDS_CSV_NAME, round_rows, list(round_rows[0]))
    return payload, case_rows, round_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    payload, _cases, _rounds = run_analysis(args.results_dir, args.output_dir)
    grounding = payload["grounding"]["overall"]["target_confusion"]
    repair = payload["repair"]["overall"]
    print(f"confirmatory cases analyzed: {payload['integrity']['cases']}")
    print(
        "grounding target TP/FN/TN/FP: "
        f"{grounding['true_positive']}/{grounding['false_negative']}/"
        f"{grounding['true_negative']}/{grounding['false_positive']}"
    )
    print(
        "repair target final/ever/last-validated: "
        f"{repair['end_to_end_target_resolution']['count']}/"
        f"{repair['ever_target_resolution']['count']}/"
        f"{repair['last_validated_target_resolution']['count']}"
    )
    print(
        "validated/exact/output-failure: "
        f"{repair['validated_state']['count']}/"
        f"{repair['end_to_end_reference_recovery']['count']}/"
        f"{repair['output_failure']['count']}"
    )
    print(f"analysis: {args.output_dir / ANALYSIS_NAME}")
    print("No model, grounding assessor, validator, reasoner, or repair was executed.")


if __name__ == "__main__":
    main()
