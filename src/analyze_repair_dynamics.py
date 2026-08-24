import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRAJECTORIES = (
    ROOT / "results" / "controlled_repair_trajectories.jsonl"
)
RUN_METADATA = (
    ROOT / "results" / "controlled_repair_trajectories.jsonl.meta.json"
)
ANALYSIS_SPEC = (
    ROOT / "experiments" / "repair_analysis_spec.json"
)

OUTPUT_JSON = ROOT / "results" / "repair_dynamics_analysis.json"
OUTPUT_CASES_CSV = ROOT / "results" / "repair_dynamics_cases.csv"
OUTPUT_ROUNDS_CSV = ROOT / "results" / "repair_dynamics_rounds.csv"

CONDITIONS = (
    "disjointness",
    "domain_range",
    "cardinality",
    "temporal",
    "grounding",
)

STOP_REASONS = {
    "validated",
    "stalled",
    "oscillation",
    "max_rounds",
    "output_failure",
    "no_feedback",
}


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc

    return rows


def validate_full_run_metadata(metadata):
    if metadata.get("cases") != 50:
        raise RuntimeError(
            "Repair metadata does not describe a 50-case run"
        )

    if metadata.get("start") != 1:
        raise RuntimeError(
            "Repair metadata does not start from case 1"
        )

    if metadata.get("limit") is not None:
        raise RuntimeError(
            "Repair metadata contains a case limit"
        )

    if metadata.get("case_id") is not None:
        raise RuntimeError(
            "Repair metadata describes a single-case run"
        )

    if metadata.get("max_repair_rounds") != 5:
        raise RuntimeError(
            "Repair metadata does not use the fixed five-round cap"
        )

    if metadata.get("invalid_model_output_retry") is not False:
        raise RuntimeError(
            "Repair metadata does not record no retry after invalid output"
        )


def verify_recorded_inputs(metadata):
    checks = (
        ("repair_spec", "repair_spec_sha256"),
        ("repair_prompt", "repair_prompt_sha256"),
        ("baseline_manifest", "baseline_manifest_sha256"),
        (
            "frozen_grounding_results",
            "frozen_grounding_results_sha256",
        ),
        (
            "frozen_target_analysis",
            "frozen_target_analysis_sha256",
        ),
    )

    for path_key, sha_key in checks:
        relative = metadata.get(path_key)
        expected = metadata.get(sha_key)

        if not isinstance(relative, str) or not relative:
            raise RuntimeError(
                f"Repair metadata has no {path_key}"
            )

        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(
                f"Repair metadata has no valid {sha_key}"
            )

        path = ROOT / relative

        if not path.exists():
            raise RuntimeError(
                f"Recorded repair input is missing: {relative}"
            )

        actual = sha256_file(path)

        if actual != expected:
            raise RuntimeError(
                f"Recorded repair input changed: {relative}"
            )


def index_unique(rows):
    output = {}

    for row in rows:
        case_id = row.get("id")

        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError("Repair trajectory has no case id")

        if case_id in output:
            raise RuntimeError(
                f"Duplicate repair trajectory: {case_id}"
            )

        output[case_id] = row

    return output


def validated_rounds(trajectory):
    return [
        row
        for row in trajectory["rounds"]
        if isinstance(row.get("validation"), dict)
    ]


def last_validated_round(trajectory):
    rows = validated_rounds(trajectory)

    if not rows:
        raise RuntimeError(
            f"{trajectory.get('id')}: no validated state"
        )

    return rows[-1]


def feedback_by_id(round_row):
    validation = round_row.get("validation")

    if not isinstance(validation, dict):
        return {}

    feedback = validation.get("actionable_feedback")

    if not isinstance(feedback, list):
        raise RuntimeError("Validated round has no actionable feedback")

    output = {}

    for item in feedback:
        violation_id = item.get("violation_id")

        if not isinstance(violation_id, str) or not violation_id:
            raise RuntimeError(
                "Feedback item has no violation identity"
            )

        output[violation_id] = item

    return output


def distinct_new_violations(trajectory):
    found = {}

    for round_row in trajectory["rounds"][1:]:
        identities = round_row.get("new_violation_ids", [])

        if not isinstance(identities, list):
            raise RuntimeError(
                f"{trajectory['id']}: invalid new violation list"
            )

        if not identities:
            continue

        lookup = feedback_by_id(round_row)

        for violation_id in identities:
            item = lookup.get(violation_id)

            if item is None:
                raise RuntimeError(
                    f"{trajectory['id']}: new violation is not in "
                    f"the round feedback: {violation_id}"
                )

            validator = item.get("validator")

            if not isinstance(validator, str) or not validator:
                raise RuntimeError(
                    f"{trajectory['id']}: new violation has no validator"
                )

            found.setdefault(violation_id, validator)

    return found


def validate_trajectory(trajectory):
    case_id = trajectory.get("id")
    domain = trajectory.get("domain")
    condition = trajectory.get("condition")
    rounds = trajectory.get("rounds")
    final = trajectory.get("final")

    if not isinstance(case_id, str) or not case_id:
        raise RuntimeError("Trajectory has no id")

    if domain not in {"movie", "music"}:
        raise RuntimeError(f"{case_id}: invalid domain")

    if condition not in CONDITIONS:
        raise RuntimeError(f"{case_id}: invalid condition")

    if not isinstance(rounds, list) or not rounds:
        raise RuntimeError(f"{case_id}: no rounds")

    expected_numbers = list(range(len(rounds)))
    observed_numbers = [row.get("round") for row in rounds]

    if observed_numbers != expected_numbers:
        raise RuntimeError(
            f"{case_id}: repair rounds are not sequential"
        )

    if not isinstance(rounds[0].get("validation"), dict):
        raise RuntimeError(
            f"{case_id}: round 0 has no validation"
        )

    if not isinstance(final, dict):
        raise RuntimeError(f"{case_id}: no final outcome")

    stop_reason = final.get("stop_reason")

    if stop_reason not in STOP_REASONS:
        raise RuntimeError(
            f"{case_id}: invalid stop reason {stop_reason!r}"
        )

    if final.get("repair_rounds") != len(rounds) - 1:
        raise RuntimeError(
            f"{case_id}: final repair round count is inconsistent"
        )

    if stop_reason == "output_failure":
        if not isinstance(final.get("output_failure"), str):
            raise RuntimeError(
                f"{case_id}: output failure has no failure type"
            )
    elif final.get("output_failure") is not None:
        raise RuntimeError(
            f"{case_id}: non-output failure has an output error"
        )

    initial_feedback = rounds[0]["validation"].get(
        "actionable_feedback"
    )

    if not isinstance(initial_feedback, list):
        raise RuntimeError(
            f"{case_id}: round 0 has no feedback list"
        )

    received = trajectory.get("received_initial_feedback")

    if received != bool(initial_feedback):
        raise RuntimeError(
            f"{case_id}: initial feedback flag is inconsistent"
        )

    sources = sorted(
        {
            item["validator"]
            for item in initial_feedback
        }
    )

    if trajectory.get("initial_feedback_sources") != sources:
        raise RuntimeError(
            f"{case_id}: initial feedback sources are inconsistent"
        )

    first_resolution = final.get("rounds_to_resolution")
    observed_resolution_rounds = [
        row["round"]
        for row in validated_rounds(trajectory)
        if row["round"] > 0
        and row["validation"].get("target_resolved") is True
    ]
    observed_first = (
        min(observed_resolution_rounds)
        if observed_resolution_rounds
        else None
    )

    if first_resolution != observed_first:
        raise RuntimeError(
            f"{case_id}: first resolution round is inconsistent"
        )

    distinct_new_violations(trajectory)


def case_row(trajectory):
    validate_trajectory(trajectory)

    final = trajectory["final"]
    valid_rows = validated_rounds(trajectory)
    last_valid = valid_rows[-1]
    last_reference = last_valid["validation"]["reference"]

    collateral_values = [
        row["validation"]["reference"][
            "collateral_symmetric_difference"
        ]
        for row in valid_rows
        if row["round"] > 0
    ]
    new_violations = distinct_new_violations(trajectory)
    new_source_counts = Counter(new_violations.values())

    first_resolution = final["rounds_to_resolution"]
    ever_resolved = first_resolution is not None
    final_resolved = bool(final["target_resolved"])
    last_validated_resolved = bool(
        last_valid["validation"]["target_resolved"]
    )
    graph_regressed = (
        ever_resolved and not last_validated_resolved
    )
    output_failure_after_resolution = (
        ever_resolved
        and final["stop_reason"] == "output_failure"
    )

    return {
        "id": trajectory["id"],
        "domain": trajectory["domain"],
        "condition": trajectory["condition"],
        "received_initial_feedback": bool(
            trajectory["received_initial_feedback"]
        ),
        "initial_feedback_sources": list(
            trajectory["initial_feedback_sources"]
        ),
        "stop_reason": final["stop_reason"],
        "repair_rounds": final["repair_rounds"],
        "final_target_resolved": final_resolved,
        "ever_target_resolved": ever_resolved,
        "first_resolution_round": first_resolution,
        "graph_regressed_after_resolution": graph_regressed,
        "output_failure_after_resolution": (
            output_failure_after_resolution
        ),
        "resolution_retained_to_end": (
            ever_resolved and final_resolved
        ),
        "validated_state": bool(final["validated_state"]),
        "validated_stop": final["stop_reason"] == "validated",
        "reference_recovery": bool(
            final["reference_recovery"]
        ),
        "output_failure": final["output_failure"] or "",
        "last_validated_round": last_valid["round"],
        "last_validated_target_resolved": bool(
            last_valid["validation"]["target_resolved"]
        ),
        "last_validated_reference_recovery": bool(
            last_reference["reference_recovery"]
        ),
        "last_validated_reference_difference": (
            last_reference["reference_symmetric_difference"]
        ),
        "last_validated_collateral_difference": (
            last_reference["collateral_symmetric_difference"]
        ),
        "any_collateral_edit": any(
            value > 0 for value in collateral_values
        ),
        "peak_collateral_difference": (
            max(collateral_values)
            if collateral_values
            else 0
        ),
        "any_new_violation": bool(new_violations),
        "distinct_new_violation_count": len(new_violations),
        "new_grounding_violation_count": (
            new_source_counts["grounding_v3"]
        ),
        "new_shacl_violation_count": (
            new_source_counts["raw_shacl"]
        ),
        "new_owl_violation_count": (
            new_source_counts["owl_consistency"]
        ),
    }


def round_rows(trajectory):
    output = []

    for row in trajectory["rounds"]:
        validation = row.get("validation")
        repair = row.get("repair")

        if isinstance(validation, dict):
            reference = validation["reference"]
            symbolic = validation["symbolic"]
            grounding = validation["grounding"]
            actionable = validation["actionable_feedback"]

            output.append(
                {
                    "id": trajectory["id"],
                    "domain": trajectory["domain"],
                    "condition": trajectory["condition"],
                    "round": row["round"],
                    "has_validated_state": True,
                    "parse_ok": (
                        repair["parse"]["ok"]
                        if isinstance(repair, dict)
                        else True
                    ),
                    "parse_failure": (
                        repair["parse"]["failure"] or ""
                        if isinstance(repair, dict)
                        else ""
                    ),
                    "target_resolved": bool(
                        validation["target_resolved"]
                    ),
                    "reference_recovery": bool(
                        reference["reference_recovery"]
                    ),
                    "reference_difference": (
                        reference[
                            "reference_symmetric_difference"
                        ]
                    ),
                    "collateral_difference": (
                        reference[
                            "collateral_symmetric_difference"
                        ]
                    ),
                    "actionable_violation_count": len(actionable),
                    "new_violation_count": len(
                        row.get("new_violation_ids", [])
                    ),
                    "shacl_violation_count": len(
                        symbolic["shacl"]["violations"]
                    ),
                    "owl_consistent": bool(
                        symbolic["owl_consistent"]
                    ),
                    "grounding_unsupported_count": sum(
                        item["verdict"] == "UNSUPPORTED"
                        for item in grounding["judgments"]
                    ),
                }
            )
        else:
            parse = (
                repair.get("parse", {})
                if isinstance(repair, dict)
                else {}
            )
            output.append(
                {
                    "id": trajectory["id"],
                    "domain": trajectory["domain"],
                    "condition": trajectory["condition"],
                    "round": row["round"],
                    "has_validated_state": False,
                    "parse_ok": bool(parse.get("ok", False)),
                    "parse_failure": (
                        parse.get("failure")
                        or trajectory["final"].get(
                            "output_failure"
                        )
                        or ""
                    ),
                    "target_resolved": "",
                    "reference_recovery": "",
                    "reference_difference": "",
                    "collateral_difference": "",
                    "actionable_violation_count": "",
                    "new_violation_count": "",
                    "shacl_violation_count": "",
                    "owl_consistent": "",
                    "grounding_unsupported_count": "",
                }
            )

    return output


def rate(rows, field):
    if not rows:
        return None

    return sum(bool(row[field]) for row in rows) / len(rows)


def conditional_target_rate(rows):
    selected = [
        row for row in rows
        if row["received_initial_feedback"]
    ]

    if not selected:
        return None

    return rate(selected, "final_target_resolved")


def graph_regression_rate_given_resolution(rows):
    selected = [
        row for row in rows
        if row["ever_target_resolved"]
    ]

    if not selected:
        return None

    return rate(
        selected,
        "graph_regressed_after_resolution",
    )


def output_failure_after_resolution_rate(rows):
    selected = [
        row for row in rows
        if row["ever_target_resolved"]
    ]

    if not selected:
        return None

    return rate(
        selected,
        "output_failure_after_resolution",
    )


def mean_repair_rounds(rows):
    if not rows:
        return None

    return statistics.mean(
        row["repair_rounds"] for row in rows
    )


BOOTSTRAP_METRICS = {
    "final_target_resolution": (
        lambda rows: rate(rows, "final_target_resolved")
    ),
    "target_resolution_given_feedback": (
        conditional_target_rate
    ),
    "ever_target_resolution": (
        lambda rows: rate(rows, "ever_target_resolved")
    ),
    "graph_regression_given_ever_resolved": (
        graph_regression_rate_given_resolution
    ),
    "output_failure_after_resolution": (
        output_failure_after_resolution_rate
    ),
    "reference_recovery": (
        lambda rows: rate(rows, "reference_recovery")
    ),
    "validated_state": (
        lambda rows: rate(rows, "validated_state")
    ),
    "validated_stop": (
        lambda rows: rate(rows, "validated_stop")
    ),
    "any_collateral_edit": (
        lambda rows: rate(rows, "any_collateral_edit")
    ),
    "any_new_violation": (
        lambda rows: rate(rows, "any_new_violation")
    ),
    "output_failure": (
        lambda rows: sum(bool(row["output_failure"]) for row in rows)
        / len(rows)
        if rows
        else None
    ),
    "mean_repair_rounds": mean_repair_rounds,
}


def percentile(values, probability):
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def bootstrap_intervals(
    rows,
    *,
    samples,
    seed,
    stratify_by_condition,
):
    groups = []

    if stratify_by_condition:
        for condition in CONDITIONS:
            selected = [
                row for row in rows
                if row["condition"] == condition
            ]

            if selected:
                groups.append(selected)
    else:
        groups = [list(rows)]

    rng = random.Random(seed)
    values = {
        name: []
        for name in BOOTSTRAP_METRICS
    }

    for _ in range(samples):
        resampled = []

        for group in groups:
            resampled.extend(
                rng.choice(group)
                for _ in range(len(group))
            )

        for name, function in BOOTSTRAP_METRICS.items():
            value = function(resampled)

            if value is not None:
                values[name].append(value)

    output = {}

    for name, function in BOOTSTRAP_METRICS.items():
        point = function(rows)
        observed = values[name]

        output[name] = {
            "estimate": point,
            "lower_95": percentile(observed, 0.025),
            "upper_95": percentile(observed, 0.975),
        }

    return output


def counts_by_validator(case_rows):
    identity_counts = Counter()
    case_sets = defaultdict(set)

    mapping = {
        "grounding_v3": "new_grounding_violation_count",
        "raw_shacl": "new_shacl_violation_count",
        "owl_consistency": "new_owl_violation_count",
    }

    for row in case_rows:
        for validator, field in mapping.items():
            count = row[field]
            identity_counts[validator] += count

            if count:
                case_sets[validator].add(row["id"])

    return {
        validator: {
            "distinct_violation_identities": (
                identity_counts[validator]
            ),
            "cases": len(case_sets[validator]),
        }
        for validator in mapping
    }


def summary_counts(case_rows):
    first_rounds = Counter(
        (
            str(row["first_resolution_round"])
            if row["first_resolution_round"] is not None
            else "never"
        )
        for row in case_rows
    )
    stop_reasons = Counter(
        row["stop_reason"] for row in case_rows
    )
    output_failures = Counter(
        row["output_failure"]
        for row in case_rows
        if row["output_failure"]
    )

    final_outcomes = Counter(
        (
            row["final_target_resolved"],
            row["validated_state"],
            row["reference_recovery"],
        )
        for row in case_rows
    )

    return {
        "n": len(case_rows),
        "received_initial_feedback": sum(
            row["received_initial_feedback"]
            for row in case_rows
        ),
        "final_target_resolved": sum(
            row["final_target_resolved"]
            for row in case_rows
        ),
        "final_target_resolved_given_feedback": sum(
            (
                row["received_initial_feedback"]
                and row["final_target_resolved"]
            )
            for row in case_rows
        ),
        "ever_target_resolved": sum(
            row["ever_target_resolved"]
            for row in case_rows
        ),
        "last_validated_target_resolved": sum(
            row["last_validated_target_resolved"]
            for row in case_rows
        ),
        "graph_regressed_after_resolution": sum(
            row["graph_regressed_after_resolution"]
            for row in case_rows
        ),
        "output_failure_after_resolution": sum(
            row["output_failure_after_resolution"]
            for row in case_rows
        ),
        "reference_recovery": sum(
            row["reference_recovery"]
            for row in case_rows
        ),
        "validated_state": sum(
            row["validated_state"]
            for row in case_rows
        ),
        "validated_stop": sum(
            row["validated_stop"]
            for row in case_rows
        ),
        "any_collateral_edit": sum(
            row["any_collateral_edit"]
            for row in case_rows
        ),
        "collateral_at_last_validated_state": sum(
            row["last_validated_collateral_difference"] > 0
            for row in case_rows
        ),
        "any_new_violation": sum(
            row["any_new_violation"]
            for row in case_rows
        ),
        "distinct_new_violation_identities": sum(
            row["distinct_new_violation_count"]
            for row in case_rows
        ),
        "new_violations_by_validator": counts_by_validator(
            case_rows
        ),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "output_failure_types": dict(
            sorted(output_failures.items())
        ),
        "first_resolution_round": dict(
            sorted(first_rounds.items())
        ),
        "mean_repair_rounds": mean_repair_rounds(case_rows),
        "median_repair_rounds": statistics.median(
            row["repair_rounds"] for row in case_rows
        )
        if case_rows
        else None,
        "final_state_cross_table": {
            (
                f"target={target},validated={validated},"
                f"reference={reference}"
            ): count
            for (
                target,
                validated,
                reference,
            ), count in sorted(final_outcomes.items())
        },
    }


def build_analysis(case_rows, spec, metadata):
    bootstrap = spec["bootstrap"]
    samples = bootstrap["samples"]
    seed = bootstrap["seed"]

    by_condition = {}

    for condition in CONDITIONS:
        selected = [
            row for row in case_rows
            if row["condition"] == condition
        ]

        by_condition[condition] = {
            "counts": summary_counts(selected),
            "intervals": bootstrap_intervals(
                selected,
                samples=samples,
                seed=seed,
                stratify_by_condition=False,
            ),
        }

    return {
        "analysis": (
            "Descriptive repair dynamics for the completed "
            "50-case controlled repair run."
        ),
        "analysis_unit": "controlled case",
        "input": {
            "trajectory_path": str(
                TRAJECTORIES.relative_to(ROOT)
            ),
            "trajectory_sha256": sha256_file(TRAJECTORIES),
            "run_metadata_path": str(
                RUN_METADATA.relative_to(ROOT)
            ),
            "run_metadata_sha256": sha256_file(RUN_METADATA),
            "run_git_head": metadata["git_head"],
        },
        "analysis_provenance": {
            "analysis_git_head": git_head(),
            "analysis_script": str(
                Path(__file__).resolve().relative_to(ROOT)
            ),
            "analysis_script_sha256": sha256_file(
                Path(__file__).resolve()
            ),
            "analysis_spec": str(
                ANALYSIS_SPEC.relative_to(ROOT)
            ),
            "analysis_spec_sha256": sha256_file(
                ANALYSIS_SPEC
            ),
        },
        "bootstrap": {
            "samples": samples,
            "seed": seed,
            "confidence_level": bootstrap[
                "confidence_level"
            ],
            "overall_sampling": (
                "resample controlled cases with replacement "
                "within each error category"
            ),
            "condition_sampling": (
                "resample controlled cases with replacement "
                "within that error category"
            ),
            "interpretation": (
                "Intervals describe sensitivity to resampling "
                "the controlled cases. They are not population "
                "performance intervals."
            ),
        },
        "overall": {
            "counts": summary_counts(case_rows),
            "intervals": bootstrap_intervals(
                case_rows,
                samples=samples,
                seed=seed,
                stratify_by_condition=True,
            ),
        },
        "by_condition": by_condition,
        "cases": case_rows,
    }


def write_cases_csv(case_rows, path):
    fieldnames = [
        "id",
        "domain",
        "condition",
        "received_initial_feedback",
        "initial_feedback_sources",
        "stop_reason",
        "repair_rounds",
        "final_target_resolved",
        "ever_target_resolved",
        "first_resolution_round",
        "graph_regressed_after_resolution",
        "output_failure_after_resolution",
        "resolution_retained_to_end",
        "validated_state",
        "validated_stop",
        "reference_recovery",
        "output_failure",
        "last_validated_round",
        "last_validated_target_resolved",
        "last_validated_reference_recovery",
        "last_validated_reference_difference",
        "last_validated_collateral_difference",
        "any_collateral_edit",
        "peak_collateral_difference",
        "any_new_violation",
        "distinct_new_violation_count",
        "new_grounding_violation_count",
        "new_shacl_violation_count",
        "new_owl_violation_count",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in case_rows:
            value = dict(row)
            value["initial_feedback_sources"] = ";".join(
                row["initial_feedback_sources"]
            )
            writer.writerow(value)


def write_rounds_csv(rows, path):
    if not rows:
        raise RuntimeError("No repair rounds to write")

    fieldnames = list(rows[0])

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(payload):
    overall = payload["overall"]["counts"]
    print(f"cases: {overall['n']}")
    print(
        "end-to-end target resolution: "
        f"{overall['final_target_resolved']}/{overall['n']}"
    )
    print(
        "end-to-end target resolution given feedback: "
        f"{overall['final_target_resolved_given_feedback']}/"
        f"{overall['received_initial_feedback']}"
    )
    print(
        "ever resolved: "
        f"{overall['ever_target_resolved']}/{overall['n']}"
    )
    print(
        "target resolved at last validated state: "
        f"{overall['last_validated_target_resolved']}/"
        f"{overall['n']}"
    )
    print(
        "valid graph regression after first resolution: "
        f"{overall['graph_regressed_after_resolution']}/"
        f"{overall['ever_target_resolved']}"
    )
    print(
        "output failure after first resolution: "
        f"{overall['output_failure_after_resolution']}/"
        f"{overall['ever_target_resolved']}"
    )
    print(
        "reference recovery: "
        f"{overall['reference_recovery']}/{overall['n']}"
    )
    print(
        "validated stop: "
        f"{overall['validated_stop']}/{overall['n']}"
    )
    print(
        "validated state: "
        f"{overall['validated_state']}/{overall['n']}"
    )
    print(
        "cases with collateral edits: "
        f"{overall['any_collateral_edit']}/{overall['n']}"
    )
    print(
        "cases with new violations: "
        f"{overall['any_new_violation']}/{overall['n']}"
    )
    print()
    print("by condition")
    print(
        "  condition          n  final  ever  reference  "
        "validated  collateral  new violations"
    )

    for condition in CONDITIONS:
        row = payload["by_condition"][condition]["counts"]
        print(
            f"  {condition:<17} "
            f"{row['n']:>2} "
            f"{row['final_target_resolved']:>6} "
            f"{row['ever_target_resolved']:>5} "
            f"{row['reference_recovery']:>10} "
            f"{row['validated_stop']:>10} "
            f"{row['any_collateral_edit']:>11} "
            f"{row['any_new_violation']:>14}"
        )

    print()
    print("stop reasons")

    for name, count in overall["stop_reasons"].items():
        print(f"  {name}: {count}")

    print()
    print("output failures")

    for name, count in overall["output_failure_types"].items():
        print(f"  {name}: {count}")

    print()
    print("first target resolution")

    for name, count in overall["first_resolution_round"].items():
        print(f"  {name}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-input-hash-check",
        action="store_true",
        help=(
            "Skip checking recorded repair input hashes. "
            "Use only for isolated analysis development, not "
            "for the main result."
        ),
    )
    args = parser.parse_args()

    required = (
        TRAJECTORIES,
        RUN_METADATA,
        ANALYSIS_SPEC,
    )
    missing = [
        path for path in required
        if not path.exists()
    ]

    if missing:
        names = "\n  ".join(
            str(path.relative_to(ROOT))
            for path in missing
        )
        raise SystemExit(
            "Missing repair analysis inputs:\n  " + names
        )

    metadata = read_json(RUN_METADATA)
    validate_full_run_metadata(metadata)

    if not args.skip_input_hash_check:
        verify_recorded_inputs(metadata)

    trajectories = read_jsonl(TRAJECTORIES)
    indexed = index_unique(trajectories)

    if len(indexed) != 50:
        raise RuntimeError(
            f"Expected 50 trajectories, found {len(indexed)}"
        )

    condition_counts = Counter(
        row["condition"] for row in trajectories
    )

    if condition_counts != Counter(
        {condition: 10 for condition in CONDITIONS}
    ):
        raise RuntimeError(
            "Repair trajectories do not contain 10 cases "
            "for each controlled error category"
        )

    domain_counts = Counter(
        row["domain"] for row in trajectories
    )

    if domain_counts != Counter({"movie": 25, "music": 25}):
        raise RuntimeError(
            "Repair trajectories do not contain 25 cases "
            "for each domain"
        )

    case_rows = [
        case_row(row)
        for row in trajectories
    ]
    all_round_rows = [
        value
        for trajectory in trajectories
        for value in round_rows(trajectory)
    ]

    spec = read_json(ANALYSIS_SPEC)
    payload = build_analysis(
        case_rows,
        spec,
        metadata,
    )

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_cases_csv(
        case_rows,
        OUTPUT_CASES_CSV,
    )
    write_rounds_csv(
        all_round_rows,
        OUTPUT_ROUNDS_CSV,
    )

    print_summary(payload)
    print()
    print(
        f"wrote: {OUTPUT_JSON.relative_to(ROOT)}"
    )
    print(
        f"wrote: {OUTPUT_CASES_CSV.relative_to(ROOT)}"
    )
    print(
        f"wrote: {OUTPUT_ROUNDS_CSV.relative_to(ROOT)}"
    )
    print("No language model or validator was executed.")


if __name__ == "__main__":
    main()
