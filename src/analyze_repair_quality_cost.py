"""Clean reference F1 and recorded model cost from the completed RQ2 repair run."""

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.analyze_repair_dynamics import (
    CONDITIONS,
    RUN_METADATA,
    TRAJECTORIES,
    index_unique,
    last_validated_round,
    percentile,
    read_json,
    read_jsonl,
    sha256_file,
    validate_full_run_metadata,
    validate_trajectory,
    verify_recorded_inputs,
)
from src.repair_engine import triple_set


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SPEC = ROOT / "experiments" / "repair_quality_cost_spec.json"
OUTPUT_JSON = ROOT / "results" / "repair_quality_cost.json"
OUTPUT_CASES_CSV = ROOT / "results" / "repair_quality_cost_cases.csv"
OUTPUT_ROUNDS_CSV = ROOT / "results" / "repair_quality_cost_rounds.csv"
DOMAINS = ("movie", "music")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42
SURVIVOR_ROUND_LABEL = (
    "Round means include only the cases that still had a validated graph at that round. "
    "They do not track the same 50 cases at every round."
)
SURVIVOR_ROUND_WARNING = (
    "A lower mean F1 at later rounds does not mean that extra rounds made the graphs worse. "
    "The later means include only the cases that were still running."
)


def load_spec(path=ANALYSIS_SPEC):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("models_or_validators_run_by_analysis") is not False:
        raise RuntimeError("This analysis must not run models or validators")
    if spec.get("analysis_unit") != "controlled case":
        raise RuntimeError("Quality and cost analysis uses the controlled case as the unit")
    return spec


def triple_key(triple):
    return tuple(triple)


def prf(true_positive, false_positive, false_negative):
    if true_positive + false_positive:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 1.0 if false_negative == 0 else 0.0
    if true_positive + false_negative:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 1.0 if false_positive == 0 else 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def scores_from_reference(triples, reference):
    predicted = triple_set(triples)
    extra = triple_set(reference["new_not_in_clean_reference"])
    missing = triple_set(reference["clean_reference_removed"])
    if not extra <= predicted:
        raise RuntimeError("Extra triples are not a subset of the current graph")
    if predicted & missing:
        raise RuntimeError("Removed clean reference triples are still in the current graph")
    scores = prf(len(predicted - extra), len(extra), len(missing))
    scores["graph_size"] = len(predicted)
    scores["reference_size"] = len(predicted - extra) + len(missing)
    scores["reference_recovery"] = bool(reference["reference_recovery"])
    scores["collateral_removed"] = len(reference.get("collateral_removed") or [])
    scores["collateral_added"] = len(reference.get("collateral_added") or [])
    scores["clean_reference_removed"] = len(missing)
    scores["new_not_in_clean_reference"] = len(extra)
    return scores


def edit_from_previous(previous_triples, current_triples):
    if previous_triples is None:
        return {
            "previous_graph_available": False,
            "triples_added": None,
            "triples_removed": None,
            "symmetric_edit_distance": None,
            "net_size_change": None,
        }
    previous = triple_set(previous_triples)
    current = triple_set(current_triples)
    added = current - previous
    removed = previous - current
    return {
        "previous_graph_available": True,
        "triples_added": len(added),
        "triples_removed": len(removed),
        "symmetric_edit_distance": len(added) + len(removed),
        "net_size_change": len(current) - len(previous),
    }


def live_grounding_judgments(round_row):
    validation = round_row.get("validation")
    if not isinstance(validation, dict):
        return []
    grounding = validation.get("grounding") or {}
    output = []
    for judgment in grounding.get("judgments") or []:
        if judgment.get("source") != "repair_round":
            continue
        if "prompt_eval_count" not in judgment or "eval_count" not in judgment:
            continue
        if "total_duration_ns" not in judgment:
            continue
        output.append(judgment)
    return output


def new_grounding_cost(round_row, seen):
    prompt = generated = duration = calls = 0
    for judgment in live_grounding_judgments(round_row):
        key = triple_key(judgment["triple"])
        if key in seen:
            continue
        seen.add(key)
        calls += 1
        prompt += judgment["prompt_eval_count"]
        generated += judgment["eval_count"]
        duration += judgment["total_duration_ns"]
    return {
        "grounding_assessor_calls": calls,
        "grounding_prompt_eval_count": prompt,
        "grounding_eval_count": generated,
        "grounding_duration_ns": duration,
    }


def repair_cost(round_row):
    repair = round_row.get("repair")
    if not isinstance(repair, dict):
        return {
            "repair_calls": 0,
            "repair_prompt_eval_count": 0,
            "repair_eval_count": 0,
            "repair_duration_ns": 0,
            "parse_ok": None,
            "parse_failure": None,
        }
    for field in ("prompt_eval_count", "eval_count", "total_duration_ns"):
        if repair.get(field) is None:
            raise RuntimeError("Repair generation is missing recorded token or duration fields")
    parse = repair.get("parse") or {}
    return {
        "repair_calls": 1,
        "repair_prompt_eval_count": repair["prompt_eval_count"],
        "repair_eval_count": repair["eval_count"],
        "repair_duration_ns": repair["total_duration_ns"],
        "parse_ok": parse.get("ok"),
        "parse_failure": parse.get("failure"),
    }


def analyze_round(round_row, previous_triples, seen_grounding):
    repair = repair_cost(round_row)
    grounding = new_grounding_cost(round_row, seen_grounding)
    validation = round_row.get("validation")
    has_graph = isinstance(round_row.get("triples"), list) and isinstance(validation, dict)
    row = {
        "round": round_row["round"],
        "has_validated_graph": has_graph,
        **repair,
        **grounding,
        "recorded_model_duration_ns": (
            repair["repair_duration_ns"] + grounding["grounding_duration_ns"]
        ),
    }
    if not has_graph:
        row.update(
            {
                "target_resolved": None,
                "graph_size": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "true_positive": None,
                "false_positive": None,
                "false_negative": None,
                "reference_recovery": None,
                "collateral_removed": None,
                "collateral_added": None,
                "clean_reference_removed": None,
                "new_not_in_clean_reference": None,
                "previous_graph_available": previous_triples is not None,
                "triples_added": None,
                "triples_removed": None,
                "symmetric_edit_distance": None,
                "net_size_change": None,
            }
        )
        return row, previous_triples
    scores = scores_from_reference(round_row["triples"], validation["reference"])
    edits = edit_from_previous(previous_triples, round_row["triples"])
    row.update(scores)
    row.update(edits)
    row["target_resolved"] = bool(validation["target_resolved"])
    return row, round_row["triples"]


def analyze_trajectory(trajectory):
    validate_trajectory(trajectory)
    seen = set()
    previous = None
    rounds = []
    for round_row in trajectory["rounds"]:
        analyzed, previous = analyze_round(round_row, previous, seen)
        rounds.append(analyzed)
    last_valid = last_validated_round(trajectory)
    last_metrics = next(row for row in rounds if row["round"] == last_valid["round"])
    initial = rounds[0]
    if not initial["has_validated_graph"]:
        raise RuntimeError("Round 0 must have a validated graph")
    final = trajectory["final"]
    totals = {
        "repair_calls": sum(row["repair_calls"] for row in rounds),
        "repair_prompt_eval_count": sum(row["repair_prompt_eval_count"] for row in rounds),
        "repair_eval_count": sum(row["repair_eval_count"] for row in rounds),
        "repair_duration_ns": sum(row["repair_duration_ns"] for row in rounds),
        "grounding_assessor_calls": sum(row["grounding_assessor_calls"] for row in rounds),
        "grounding_prompt_eval_count": sum(row["grounding_prompt_eval_count"] for row in rounds),
        "grounding_eval_count": sum(row["grounding_eval_count"] for row in rounds),
        "grounding_duration_ns": sum(row["grounding_duration_ns"] for row in rounds),
    }
    totals["recorded_model_duration_ns"] = (
        totals["repair_duration_ns"] + totals["grounding_duration_ns"]
    )
    return {
        "id": trajectory["id"],
        "domain": trajectory["domain"],
        "condition": trajectory["condition"],
        "stop_reason": final["stop_reason"],
        "output_failure": final["output_failure"],
        "end_to_end_target_resolved": bool(final["target_resolved"]),
        "validated_state": bool(final["validated_state"]),
        "end_to_end_reference_recovery": bool(final["reference_recovery"]),
        "last_validated_round": last_metrics["round"],
        "last_validated_target_resolved": last_metrics["target_resolved"],
        "last_validated_reference_recovery": last_metrics["reference_recovery"],
        "last_validated_precision": last_metrics["precision"],
        "last_validated_recall": last_metrics["recall"],
        "last_validated_f1": last_metrics["f1"],
        "last_validated_true_positive": last_metrics["true_positive"],
        "last_validated_false_positive": last_metrics["false_positive"],
        "last_validated_false_negative": last_metrics["false_negative"],
        "last_validated_graph_size": last_metrics["graph_size"],
        "last_validated_collateral_removed": last_metrics["collateral_removed"],
        "last_validated_collateral_added": last_metrics["collateral_added"],
        "last_validated_symmetric_edit_distance": last_metrics["symmetric_edit_distance"],
        "initial_precision": initial["precision"],
        "initial_recall": initial["recall"],
        "initial_f1": initial["f1"],
        "initial_true_positive": initial["true_positive"],
        "initial_false_positive": initial["false_positive"],
        "initial_false_negative": initial["false_negative"],
        "initial_graph_size": initial["graph_size"],
        "f1_delta": last_metrics["f1"] - initial["f1"],
        "rounds": rounds,
        "transitions": graph_transitions(rounds),
        **totals,
    }


def mean(values):
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def median(values):
    present = [value for value in values if value is not None]
    if not present:
        return None
    return statistics.median(present)


def classify_delta(delta):
    if delta > 0:
        return "improved"
    if delta < 0:
        return "worsened"
    return "unchanged"


def count_changes(deltas):
    improved = unchanged = worsened = 0
    for delta in deltas:
        label = classify_delta(delta)
        if label == "improved":
            improved += 1
        elif label == "worsened":
            worsened += 1
        else:
            unchanged += 1
    return {
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
    }


def graph_transitions(rounds):
    usable = [row for row in rounds if row["has_validated_graph"]]
    transitions = []
    for before, after in zip(usable, usable[1:]):
        delta = after["f1"] - before["f1"]
        transitions.append(
            {
                "from_round": before["round"],
                "to_round": after["round"],
                "repair_transition": after["round"],
                "f1_before": before["f1"],
                "f1_after": after["f1"],
                "delta": delta,
                "change": classify_delta(delta),
            }
        )
    return transitions


def bootstrap_mean(values, *, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, groups=None):
    if not values:
        return {
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "samples": samples,
            "seed": seed,
            "unit": "controlled case",
        }
    if groups is None:
        grouped = [list(values)]
    else:
        grouped = [list(group) for group in groups if group]
    rng = random.Random(seed)
    observed = []
    for _ in range(samples):
        resampled = []
        for group in grouped:
            resampled.extend(rng.choice(group) for _ in range(len(group)))
        observed.append(sum(resampled) / len(resampled))
    return {
        "estimate": sum(values) / len(values),
        "lower_95": percentile(observed, 0.025),
        "upper_95": percentile(observed, 0.975),
        "samples": samples,
        "seed": seed,
        "unit": "controlled case",
    }


def paired_change_summary(cases, *, stratify_by=None, stratify_order=None):
    deltas = [row["f1_delta"] for row in cases]
    if stratify_by:
        groups = []
        for name in stratify_order:
            selected = [row["f1_delta"] for row in cases if row[stratify_by] == name]
            if selected:
                groups.append(selected)
    else:
        groups = None
    return {
        "n": len(cases),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        **count_changes(deltas),
        "bootstrap_mean_delta": bootstrap_mean(deltas, groups=groups),
    }


def grouped_paired(cases, key, expected):
    groups = defaultdict(list)
    for row in cases:
        groups[row[key]].append(row)
    if set(groups) != set(expected):
        raise RuntimeError(f"Unexpected {key} groups: {sorted(groups)}")
    return {name: paired_change_summary(groups[name]) for name in expected}


def summarize_transitions(transitions):
    deltas = [row["delta"] for row in transitions]
    by_number = defaultdict(list)
    for row in transitions:
        by_number[row["repair_transition"]].append(row)
    by_repair_transition = {}
    for number in sorted(by_number):
        group_deltas = [row["delta"] for row in by_number[number]]
        by_repair_transition[number] = {
            "n": len(group_deltas),
            "mean_delta": mean(group_deltas),
            "median_delta": median(group_deltas),
            **count_changes(group_deltas),
        }
    return {
        "n": len(transitions),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        **count_changes(deltas),
        "by_repair_transition": by_repair_transition,
    }


def survivor_round_summaries(cases):
    by_round = defaultdict(list)
    for case in cases:
        for round_row in case["rounds"]:
            if round_row["has_validated_graph"]:
                by_round[round_row["round"]].append(round_row["f1"])
    return {
        "label": SURVIVOR_ROUND_LABEL,
        "warning": SURVIVOR_ROUND_WARNING,
        "by_round": {
            round_n: {
                "n": len(values),
                "mean_f1": mean(values),
                "median_f1": median(values),
            }
            for round_n, values in sorted(by_round.items())
        },
    }


def domain_range_diagnostic(cases):
    selected = [row for row in cases if row["condition"] == "domain_range"]
    return {
        "purpose": (
            "Initial and last validated precision, recall, and F1 for each "
            "domain_range case. Use this list to check a mean last validated F1 "
            "of 0 against the case scores."
        ),
        "n": len(selected),
        "mean_initial_f1": mean(row["initial_f1"] for row in selected),
        "mean_last_validated_f1": mean(row["last_validated_f1"] for row in selected),
        "cases": [
            {
                "id": row["id"],
                "domain": row["domain"],
                "initial_true_positive": row["initial_true_positive"],
                "initial_false_positive": row["initial_false_positive"],
                "initial_false_negative": row["initial_false_negative"],
                "initial_precision": row["initial_precision"],
                "initial_recall": row["initial_recall"],
                "initial_f1": row["initial_f1"],
                "last_validated_true_positive": row["last_validated_true_positive"],
                "last_validated_false_positive": row["last_validated_false_positive"],
                "last_validated_false_negative": row["last_validated_false_negative"],
                "last_validated_precision": row["last_validated_precision"],
                "last_validated_recall": row["last_validated_recall"],
                "last_validated_f1": row["last_validated_f1"],
                "last_validated_graph_size": row["last_validated_graph_size"],
            }
            for row in selected
        ],
    }


def summarize_cases(cases):
    n = len(cases)
    return {
        "n": n,
        "aggregation": "one last validated value per controlled case",
        "end_to_end_target_resolved": sum(row["end_to_end_target_resolved"] for row in cases),
        "last_validated_target_resolved": sum(row["last_validated_target_resolved"] for row in cases),
        "output_failure": sum(row["output_failure"] is not None for row in cases),
        "validated_state": sum(row["validated_state"] for row in cases),
        "end_to_end_reference_recovery": sum(row["end_to_end_reference_recovery"] for row in cases),
        "last_validated_reference_recovery": sum(row["last_validated_reference_recovery"] for row in cases),
        "mean_last_validated_precision": mean(row["last_validated_precision"] for row in cases),
        "mean_last_validated_recall": mean(row["last_validated_recall"] for row in cases),
        "mean_last_validated_f1": mean(row["last_validated_f1"] for row in cases),
        "mean_last_validated_collateral_removed": mean(
            row["last_validated_collateral_removed"] for row in cases
        ),
        "mean_repair_calls": mean(row["repair_calls"] for row in cases),
        "mean_repair_prompt_eval_count": mean(row["repair_prompt_eval_count"] for row in cases),
        "mean_repair_eval_count": mean(row["repair_eval_count"] for row in cases),
        "mean_repair_duration_ns": mean(row["repair_duration_ns"] for row in cases),
        "mean_grounding_assessor_calls": mean(row["grounding_assessor_calls"] for row in cases),
        "mean_grounding_prompt_eval_count": mean(row["grounding_prompt_eval_count"] for row in cases),
        "mean_grounding_eval_count": mean(row["grounding_eval_count"] for row in cases),
        "mean_grounding_duration_ns": mean(row["grounding_duration_ns"] for row in cases),
        "mean_recorded_model_duration_ns": mean(row["recorded_model_duration_ns"] for row in cases),
        "sum_repair_calls": sum(row["repair_calls"] for row in cases),
        "sum_repair_prompt_eval_count": sum(row["repair_prompt_eval_count"] for row in cases),
        "sum_repair_eval_count": sum(row["repair_eval_count"] for row in cases),
        "sum_repair_duration_ns": sum(row["repair_duration_ns"] for row in cases),
        "sum_grounding_assessor_calls": sum(row["grounding_assessor_calls"] for row in cases),
        "sum_grounding_prompt_eval_count": sum(row["grounding_prompt_eval_count"] for row in cases),
        "sum_grounding_eval_count": sum(row["grounding_eval_count"] for row in cases),
        "sum_grounding_duration_ns": sum(row["grounding_duration_ns"] for row in cases),
        "sum_recorded_model_duration_ns": sum(row["recorded_model_duration_ns"] for row in cases),
    }


def grouped_summary(cases, key, expected):
    groups = defaultdict(list)
    for row in cases:
        groups[row[key]].append(row)
    if set(groups) != set(expected):
        raise RuntimeError(f"Unexpected {key} groups: {sorted(groups)}")
    return {name: summarize_cases(groups[name]) for name in expected}


def analyze_records(trajectories, metadata, spec, verify_inputs=True):
    if verify_inputs:
        validate_full_run_metadata(metadata)
        verify_recorded_inputs(metadata)
    indexed = index_unique(trajectories)
    if verify_inputs and len(indexed) != 50:
        raise RuntimeError("Quality and cost analysis requires the 50 case RQ2 run")
    cases = [analyze_trajectory(indexed[case_id]) for case_id in sorted(indexed)]
    condition_names = CONDITIONS if verify_inputs else tuple(sorted({row["condition"] for row in cases}))
    domain_names = DOMAINS if verify_inputs else tuple(sorted({row["domain"] for row in cases}))
    transitions = [row for case in cases for row in case["transitions"]]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_unit": "controlled case",
        "reference_f1_interpretation": spec["reference_f1_interpretation"],
        "unavailable": spec["unavailable"],
        "survivor_round_warning": SURVIVOR_ROUND_WARNING,
        "input": {
            "trajectory_path": "results/controlled_repair_trajectories.jsonl",
            "trajectory_sha256": sha256_file(TRAJECTORIES) if TRAJECTORIES.exists() else None,
        },
        "overall": summarize_cases(cases),
        "by_condition": grouped_summary(cases, "condition", condition_names),
        "by_domain": grouped_summary(cases, "domain", domain_names),
        "paired_f1_change": {
            "overall": paired_change_summary(
                cases,
                stratify_by="condition",
                stratify_order=condition_names,
            ),
            "by_condition": grouped_paired(cases, "condition", condition_names),
            "by_domain": grouped_paired(cases, "domain", domain_names),
        },
        "transitions": summarize_transitions(transitions),
        "survivor_round_summaries": survivor_round_summaries(cases),
        "domain_range_diagnostic": domain_range_diagnostic(cases),
        "cases": cases,
    }


CASE_FIELDS = [
    "id",
    "domain",
    "condition",
    "stop_reason",
    "output_failure",
    "end_to_end_target_resolved",
    "validated_state",
    "end_to_end_reference_recovery",
    "last_validated_round",
    "last_validated_target_resolved",
    "last_validated_reference_recovery",
    "last_validated_precision",
    "last_validated_recall",
    "last_validated_f1",
    "last_validated_true_positive",
    "last_validated_false_positive",
    "last_validated_false_negative",
    "last_validated_graph_size",
    "last_validated_collateral_removed",
    "last_validated_collateral_added",
    "initial_precision",
    "initial_recall",
    "initial_f1",
    "initial_true_positive",
    "initial_false_positive",
    "initial_false_negative",
    "initial_graph_size",
    "f1_delta",
    "repair_calls",
    "repair_prompt_eval_count",
    "repair_eval_count",
    "repair_duration_ns",
    "grounding_assessor_calls",
    "grounding_prompt_eval_count",
    "grounding_eval_count",
    "grounding_duration_ns",
    "recorded_model_duration_ns",
]

ROUND_FIELDS = [
    "id",
    "domain",
    "condition",
    "round",
    "has_validated_graph",
    "target_resolved",
    "graph_size",
    "precision",
    "recall",
    "f1",
    "true_positive",
    "false_positive",
    "false_negative",
    "reference_recovery",
    "collateral_removed",
    "collateral_added",
    "triples_added",
    "triples_removed",
    "symmetric_edit_distance",
    "net_size_change",
    "repair_calls",
    "repair_prompt_eval_count",
    "repair_eval_count",
    "repair_duration_ns",
    "parse_ok",
    "parse_failure",
    "grounding_assessor_calls",
    "grounding_prompt_eval_count",
    "grounding_eval_count",
    "grounding_duration_ns",
    "recorded_model_duration_ns",
]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_outputs(payload, json_path=OUTPUT_JSON, cases_path=OUTPUT_CASES_CSV, rounds_path=OUTPUT_ROUNDS_CSV):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    json_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(cases_path, CASE_FIELDS, payload["cases"])
    round_rows = []
    for case in payload["cases"]:
        for round_row in case["rounds"]:
            round_rows.append(
                {
                    "id": case["id"],
                    "domain": case["domain"],
                    "condition": case["condition"],
                    **round_row,
                }
            )
    write_csv(rounds_path, ROUND_FIELDS, round_rows)
    return json_path, cases_path, rounds_path


def run_analysis(
    trajectory_path=TRAJECTORIES,
    metadata_path=RUN_METADATA,
    spec_path=ANALYSIS_SPEC,
    json_path=OUTPUT_JSON,
    cases_path=OUTPUT_CASES_CSV,
    rounds_path=OUTPUT_ROUNDS_CSV,
    verify_inputs=True,
):
    spec = load_spec(spec_path)
    payload = analyze_records(
        read_jsonl(trajectory_path),
        read_json(metadata_path),
        spec,
        verify_inputs=verify_inputs,
    )
    write_outputs(payload, json_path, cases_path, rounds_path)
    overall = payload["overall"]
    paired = payload["paired_f1_change"]["overall"]
    bootstrap = paired["bootstrap_mean_delta"]
    transitions = payload["transitions"]
    diagnostic = payload["domain_range_diagnostic"]
    print(f"cases: {overall['n']}")
    print(f"mean last validated F1: {overall['mean_last_validated_f1']}")
    print("paired F1 change from the initial graph to the last validated graph:")
    print(
        f"  mean delta={paired['mean_delta']} median delta={paired['median_delta']} "
        f"improved={paired['improved']} unchanged={paired['unchanged']} "
        f"worsened={paired['worsened']}"
    )
    print(
        f"  bootstrap mean delta 95% interval "
        f"[{bootstrap['lower_95']}, {bootstrap['upper_95']}] "
        f"samples={bootstrap['samples']} seed={bootstrap['seed']}"
    )
    print("consecutive validated graphs in the same case:")
    print(
        f"  n={transitions['n']} mean delta={transitions['mean_delta']} "
        f"median delta={transitions['median_delta']} "
        f"improved={transitions['improved']} unchanged={transitions['unchanged']} "
        f"worsened={transitions['worsened']}"
    )
    for number, row in transitions["by_repair_transition"].items():
        print(
            f"  repair transition {number}: n={row['n']} mean={row['mean_delta']} "
            f"median={row['median_delta']} improved={row['improved']} "
            f"unchanged={row['unchanged']} worsened={row['worsened']}"
        )
    print(payload["survivor_round_summaries"]["label"])
    print(payload["survivor_round_summaries"]["warning"])
    print("domain_range diagnostic:")
    print(
        f"  n={diagnostic['n']} mean initial F1={diagnostic['mean_initial_f1']} "
        f"mean last validated F1={diagnostic['mean_last_validated_f1']}"
    )
    for row in diagnostic["cases"]:
        print(
            f"  {row['id']} {row['domain']} "
            f"initial P/R/F1={row['initial_precision']:.3f}/"
            f"{row['initial_recall']:.3f}/{row['initial_f1']:.3f} "
            f"(tp={row['initial_true_positive']} fp={row['initial_false_positive']} "
            f"fn={row['initial_false_negative']}) "
            f"last P/R/F1={row['last_validated_precision']:.3f}/"
            f"{row['last_validated_recall']:.3f}/{row['last_validated_f1']:.3f} "
            f"(tp={row['last_validated_true_positive']} "
            f"fp={row['last_validated_false_positive']} "
            f"fn={row['last_validated_false_negative']} "
            f"size={row['last_validated_graph_size']})"
        )
    print(f"repair generations: {overall['sum_repair_calls']}")
    print(f"live grounding assessor calls: {overall['sum_grounding_assessor_calls']}")
    print(f"wrote: {json_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_analysis()


if __name__ == "__main__":
    main()
