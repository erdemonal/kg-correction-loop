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
PRIMARY_CONDITIONS = (
    "disjointness",
    "cardinality",
    "temporal",
    "grounding",
)
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42
BOOTSTRAP_INTERPRETATION = (
    "This interval describes sensitivity to resampling the controlled cases. "
    "It is not a population confidence interval."
)
SURVIVOR_ROUND_LABEL = (
    "Round means include only the nonempty reference cases that still had a validated graph at that round. "
    "They do not track a fixed set of cases at every round."
)
SURVIVOR_ROUND_WARNING = (
    "A lower mean F1 at later rounds does not mean that extra rounds made the graphs worse. "
    "The later means include only the cases that were still running."
)
PRIMARY_F1_NOTE = (
    "Primary clean reference F1 uses only the cases whose initial clean reference contains at least one triple."
)
CONVENTION_SUMMARY_NOTE = (
    "This summary uses all 50 cases. It includes the 10 empty reference domain_range cases. "
    "F1 of 0 for a nonempty prediction against an empty reference is only an explicit computational convention. "
    "It is not the primary graph quality estimate."
)
EMPTY_REFERENCE_NOTE = (
    "These cases have an empty clean reference at the initial graph. "
    "The quality metric is the number of extra triples, not F1. "
    "F1 of 0 for a nonempty prediction against an empty reference is only an explicit computational convention."
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
                "reference_size": None,
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
        "last_validated_reference_size": last_metrics["reference_size"],
        "last_validated_extra_triples": last_metrics["new_not_in_clean_reference"],
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
        "initial_reference_size": initial["reference_size"],
        "initial_extra_triples": initial["new_not_in_clean_reference"],
        "empty_reference": initial["reference_size"] == 0,
        "extra_delta": (
            last_metrics["new_not_in_clean_reference"] - initial["new_not_in_clean_reference"]
        ),
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
            "interpretation": BOOTSTRAP_INTERPRETATION,
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
        "interpretation": BOOTSTRAP_INTERPRETATION,
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
    bootstrap = bootstrap_mean(deltas, groups=groups)
    return {
        "n": len(cases),
        "mean_initial_precision": mean(row["initial_precision"] for row in cases),
        "mean_initial_recall": mean(row["initial_recall"] for row in cases),
        "mean_initial_f1": mean(row["initial_f1"] for row in cases),
        "mean_last_validated_precision": mean(row["last_validated_precision"] for row in cases),
        "mean_last_validated_recall": mean(row["last_validated_recall"] for row in cases),
        "mean_last_validated_f1": mean(row["last_validated_f1"] for row in cases),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        **count_changes(deltas),
        "bootstrap_mean_delta": bootstrap,
        "end_to_end_target_resolved": sum(row["end_to_end_target_resolved"] for row in cases),
        "last_validated_target_resolved": sum(row["last_validated_target_resolved"] for row in cases),
        "output_failure": sum(row["output_failure"] is not None for row in cases),
        "validated_state": sum(row["validated_state"] for row in cases),
    }


def grouped_paired(cases, key, expected):
    groups = defaultdict(list)
    for row in cases:
        groups[row[key]].append(row)
    if set(groups) != set(expected):
        raise RuntimeError(f"Unexpected {key} groups: {sorted(groups)}")
    return {name: paired_change_summary(groups[name]) for name in expected}


def nonempty_reference_cases(cases):
    return [row for row in cases if row["initial_reference_size"] > 0]


def empty_reference_cases(cases):
    return [row for row in cases if row["initial_reference_size"] == 0]


def classify_extra_change(delta):
    if delta < 0:
        return "improved"
    if delta > 0:
        return "worsened"
    return "unchanged"


def count_extra_changes(deltas):
    improved = unchanged = worsened = 0
    for delta in deltas:
        label = classify_extra_change(delta)
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


def present_names(cases, key, expected):
    found = {row[key] for row in cases}
    return tuple(name for name in expected if name in found)


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
            "Case table for each domain_range trajectory, including precision, recall, and F1. "
            "All 10 have an empty initial clean reference. Extra triple counts in the empty "
            "reference summary are the quality metric for these cases."
        ),
        "n": len(selected),
        "mean_initial_f1": mean(row["initial_f1"] for row in selected),
        "mean_last_validated_f1": mean(row["last_validated_f1"] for row in selected),
        "cases": [
            {
                "id": row["id"],
                "domain": row["domain"],
                "initial_reference_size": row["initial_reference_size"],
                "initial_extra_triples": row["initial_extra_triples"],
                "initial_true_positive": row["initial_true_positive"],
                "initial_false_positive": row["initial_false_positive"],
                "initial_false_negative": row["initial_false_negative"],
                "initial_precision": row["initial_precision"],
                "initial_recall": row["initial_recall"],
                "initial_f1": row["initial_f1"],
                "last_validated_extra_triples": row["last_validated_extra_triples"],
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


def empty_reference_summary(cases):
    selected = empty_reference_cases(cases)
    extras_initial = [row["initial_extra_triples"] for row in selected]
    extras_last = [row["last_validated_extra_triples"] for row in selected]
    extra_deltas = [row["extra_delta"] for row in selected]
    recovered = [row["last_validated_graph_size"] == 0 for row in selected]
    return {
        "note": EMPTY_REFERENCE_NOTE,
        "primary_metric": "extra triples, not F1",
        "n": len(selected),
        "exact_empty_graph_recovery": sum(recovered),
        "exact_empty_graph_recovery_rate": (
            sum(recovered) / len(selected) if selected else None
        ),
        "mean_initial_extra_triples": mean(extras_initial),
        "mean_last_validated_extra_triples": mean(extras_last),
        "median_initial_extra_triples": median(extras_initial),
        "median_last_validated_extra_triples": median(extras_last),
        "mean_extra_delta": mean(extra_deltas),
        "median_extra_delta": median(extra_deltas),
        **count_extra_changes(extra_deltas),
        "end_to_end_target_resolved": sum(row["end_to_end_target_resolved"] for row in selected),
        "last_validated_target_resolved": sum(
            row["last_validated_target_resolved"] for row in selected
        ),
        "output_failure": sum(row["output_failure"] is not None for row in selected),
        "validated_state": sum(row["validated_state"] for row in selected),
        "cases": [
            {
                "id": row["id"],
                "domain": row["domain"],
                "condition": row["condition"],
                "initial_reference_size": row["initial_reference_size"],
                "initial_extra_triples": row["initial_extra_triples"],
                "last_validated_extra_triples": row["last_validated_extra_triples"],
                "extra_delta": row["extra_delta"],
                "extra_change": classify_extra_change(row["extra_delta"]),
                "last_validated_graph_size": row["last_validated_graph_size"],
                "exact_empty_graph_recovery": row["last_validated_graph_size"] == 0,
                "end_to_end_target_resolved": row["end_to_end_target_resolved"],
                "last_validated_target_resolved": row["last_validated_target_resolved"],
                "output_failure": row["output_failure"],
                "validated_state": row["validated_state"],
            }
            for row in selected
        ],
    }


def f1_cohort_block(cases, *, note, primary, condition_names, domain_names, stratify_overall):
    overall = paired_change_summary(
        cases,
        stratify_by="condition" if stratify_overall else None,
        stratify_order=condition_names if stratify_overall else None,
    )
    transitions = [row for case in cases for row in case["transitions"]]
    return {
        "note": note,
        "primary": primary,
        **overall,
        "by_condition": grouped_paired(cases, "condition", condition_names) if cases else {},
        "by_domain": grouped_paired(cases, "domain", domain_names) if cases else {},
        "transitions": summarize_transitions(transitions),
        "survivor_round_summaries": survivor_round_summaries(cases),
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
    primary = nonempty_reference_cases(cases)
    empty = empty_reference_cases(cases)
    if verify_inputs:
        if len(cases) != 50:
            raise RuntimeError("Quality and cost analysis requires the 50 case RQ2 run")
        if len(primary) != 40:
            raise RuntimeError("Primary F1 requires 40 nonempty reference cases")
        if len(empty) != 10:
            raise RuntimeError("Empty reference summary requires 10 empty reference cases")
        if {row["condition"] for row in primary} != set(PRIMARY_CONDITIONS):
            raise RuntimeError("Primary F1 conditions do not match the four nonempty reference conditions")
        condition_names = CONDITIONS
        domain_names = DOMAINS
        primary_conditions = PRIMARY_CONDITIONS
        primary_domains = DOMAINS
    else:
        condition_names = tuple(sorted({row["condition"] for row in cases}))
        domain_names = tuple(sorted({row["domain"] for row in cases}))
        primary_conditions = present_names(primary, "condition", condition_names)
        primary_domains = present_names(primary, "domain", domain_names)
    convention_conditions = present_names(cases, "condition", condition_names)
    convention_domains = present_names(cases, "domain", domain_names)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_unit": "controlled case",
        "reference_f1_interpretation": spec["reference_f1_interpretation"],
        "unavailable": spec["unavailable"],
        "survivor_round_warning": SURVIVOR_ROUND_WARNING,
        "bootstrap_interpretation": BOOTSTRAP_INTERPRETATION,
        "input": {
            "trajectory_path": "results/controlled_repair_trajectories.jsonl",
            "trajectory_sha256": sha256_file(TRAJECTORIES) if TRAJECTORIES.exists() else None,
        },
        "overall": summarize_cases(cases),
        "primary_f1": f1_cohort_block(
            primary,
            note=PRIMARY_F1_NOTE,
            primary=True,
            condition_names=primary_conditions,
            domain_names=primary_domains,
            stratify_overall=True,
        ),
        "all_case_convention_based_summary": {
            "includes_empty_reference_domain_range_cases": len(empty),
            **f1_cohort_block(
                cases,
                note=CONVENTION_SUMMARY_NOTE if verify_inputs else (
                    f"This summary uses all {len(cases)} cases. "
                    f"It includes {len(empty)} empty reference cases. "
                    "F1 of 0 for a nonempty prediction against an empty reference is only an explicit computational convention. "
                    "It is not the primary graph quality estimate."
                ),
                primary=False,
                condition_names=convention_conditions,
                domain_names=convention_domains,
                stratify_overall=True,
            ),
        },
        "empty_reference": empty_reference_summary(cases),
        "by_condition": grouped_summary(cases, "condition", convention_conditions) if cases else {},
        "by_domain": grouped_summary(cases, "domain", convention_domains) if cases else {},
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
    "last_validated_reference_size",
    "last_validated_extra_triples",
    "last_validated_collateral_removed",
    "last_validated_collateral_added",
    "initial_precision",
    "initial_recall",
    "initial_f1",
    "initial_true_positive",
    "initial_false_positive",
    "initial_false_negative",
    "initial_graph_size",
    "initial_reference_size",
    "initial_extra_triples",
    "empty_reference",
    "extra_delta",
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
    "reference_size",
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


def print_f1_block(title, block):
    bootstrap = block["bootstrap_mean_delta"]
    print(title)
    print(block["note"])
    print(f"  n={block['n']}")
    print(
        f"  initial mean P/R/F1={block['mean_initial_precision']}/"
        f"{block['mean_initial_recall']}/{block['mean_initial_f1']}"
    )
    print(
        f"  last validated mean P/R/F1={block['mean_last_validated_precision']}/"
        f"{block['mean_last_validated_recall']}/{block['mean_last_validated_f1']}"
    )
    print(
        f"  paired F1 mean delta={block['mean_delta']} median delta={block['median_delta']} "
        f"improved={block['improved']} unchanged={block['unchanged']} "
        f"worsened={block['worsened']}"
    )
    print(
        f"  controlled case resampling sensitivity "
        f"[{bootstrap['lower_95']}, {bootstrap['upper_95']}] "
        f"samples={bootstrap['samples']} seed={bootstrap['seed']}"
    )
    print(f"  {bootstrap['interpretation']}")
    print(
        f"  end to end target resolved={block['end_to_end_target_resolved']} "
        f"last validated target resolved={block['last_validated_target_resolved']} "
        f"output failure={block['output_failure']} "
        f"validated state={block['validated_state']}"
    )
    for name, row in block["by_condition"].items():
        print(
            f"  condition {name}: n={row['n']} mean delta={row['mean_delta']} "
            f"+{row['improved']} ={row['unchanged']} -{row['worsened']}"
        )
    for name, row in block["by_domain"].items():
        print(
            f"  domain {name}: n={row['n']} mean delta={row['mean_delta']} "
            f"+{row['improved']} ={row['unchanged']} -{row['worsened']}"
        )


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
    primary = payload["primary_f1"]
    convention = payload["all_case_convention_based_summary"]
    empty = payload["empty_reference"]
    diagnostic = payload["domain_range_diagnostic"]
    print(f"cases: {overall['n']}")
    print_f1_block("primary clean reference F1 (nonempty reference cases):", primary)
    print_f1_block("all case convention based summary (secondary):", convention)
    print("empty reference diagnostic (extra triples, not F1):")
    print(empty["note"])
    print(f"  n={empty['n']}")
    print(
        f"  exact empty graph recovery={empty['exact_empty_graph_recovery']}/"
        f"{empty['n']} rate={empty['exact_empty_graph_recovery_rate']}"
    )
    print(
        f"  extra triples initial mean/median="
        f"{empty['mean_initial_extra_triples']}/{empty['median_initial_extra_triples']} "
        f"last validated mean/median="
        f"{empty['mean_last_validated_extra_triples']}/{empty['median_last_validated_extra_triples']}"
    )
    print(
        f"  extra delta mean={empty['mean_extra_delta']} median={empty['median_extra_delta']} "
        f"improved={empty['improved']} unchanged={empty['unchanged']} "
        f"worsened={empty['worsened']}"
    )
    print(
        f"  end to end target resolved={empty['end_to_end_target_resolved']} "
        f"last validated target resolved={empty['last_validated_target_resolved']} "
        f"output failure={empty['output_failure']} "
        f"validated state={empty['validated_state']}"
    )
    for row in empty["cases"]:
        print(
            f"  {row['id']} {row['domain']} extras {row['initial_extra_triples']} -> "
            f"{row['last_validated_extra_triples']} ({row['extra_change']}) "
            f"size={row['last_validated_graph_size']} "
            f"empty_graph={row['exact_empty_graph_recovery']} "
            f"e2e_target={row['end_to_end_target_resolved']} "
            f"output_failure={row['output_failure']}"
        )
    print(primary["survivor_round_summaries"]["label"])
    print(primary["survivor_round_summaries"]["warning"])
    print("domain_range diagnostic:")
    print(
        f"  n={diagnostic['n']} mean initial F1={diagnostic['mean_initial_f1']} "
        f"mean last validated F1={diagnostic['mean_last_validated_f1']}"
    )
    for row in diagnostic["cases"]:
        print(
            f"  {row['id']} {row['domain']} "
            f"ref_size={row['initial_reference_size']} "
            f"initial P/R/F1={row['initial_precision']:.3f}/"
            f"{row['initial_recall']:.3f}/{row['initial_f1']:.3f} "
            f"(tp={row['initial_true_positive']} fp={row['initial_false_positive']} "
            f"fn={row['initial_false_negative']}) "
            f"last P/R/F1={row['last_validated_precision']:.3f}/"
            f"{row['last_validated_recall']:.3f}/{row['last_validated_f1']:.3f} "
            f"(tp={row['last_validated_true_positive']} "
            f"fp={row['last_validated_false_positive']} "
            f"fn={row['last_validated_false_negative']} "
            f"size={row['last_validated_graph_size']} "
            f"extras={row['last_validated_extra_triples']})"
        )
    print(f"repair generations: {overall['sum_repair_calls']}")
    print(f"live grounding assessor calls: {overall['sum_grounding_assessor_calls']}")
    print(f"wrote: {json_path}")
    print(f"wrote: {cases_path}")
    print(f"wrote: {rounds_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_analysis()


if __name__ == "__main__":
    main()
