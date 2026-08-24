
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "results" / "repair_dynamics_analysis.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "repair_report"

CONDITIONS = (
    "disjointness",
    "domain_range",
    "cardinality",
    "temporal",
    "grounding",
)

CONDITION_LABELS = {
    "disjointness": "Disjointness",
    "domain_range": "Domain/range",
    "cardinality": "Cardinality",
    "temporal": "Temporal",
    "grounding": "Grounding",
}

STOP_ORDER = (
    "validated",
    "output_failure",
    "max_rounds",
    "oscillation",
    "stalled",
    "no_feedback",
)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_analysis(payload):
    if payload.get("analysis_unit") != "controlled case":
        raise RuntimeError("Unexpected repair analysis unit")

    overall = payload.get("overall")
    by_condition = payload.get("by_condition")
    cases = payload.get("cases")

    if not isinstance(overall, dict):
        raise RuntimeError("Repair analysis has no overall summary")

    if not isinstance(by_condition, dict):
        raise RuntimeError("Repair analysis has no condition summary")

    if not isinstance(cases, list) or len(cases) != 50:
        raise RuntimeError("Repair analysis must contain 50 cases")

    counts = overall.get("counts")
    if not isinstance(counts, dict) or counts.get("n") != 50:
        raise RuntimeError("Overall repair summary must contain 50 cases")

    for condition in CONDITIONS:
        item = by_condition.get(condition)
        if not isinstance(item, dict):
            raise RuntimeError(f"Missing condition: {condition}")

        condition_counts = item.get("counts")
        if (
            not isinstance(condition_counts, dict)
            or condition_counts.get("n") != 10
        ):
            raise RuntimeError(
                f"Condition {condition} must contain 10 cases"
            )

    required_overall = (
        "received_initial_feedback",
        "final_target_resolved",
        "ever_target_resolved",
        "last_validated_target_resolved",
        "reference_recovery",
        "validated_state",
        "validated_stop",
        "any_collateral_edit",
        "any_new_violation",
        "stop_reasons",
        "first_resolution_round",
    )

    missing = [
        key for key in required_overall
        if key not in counts
    ]
    if missing:
        raise RuntimeError(
            "Repair analysis is missing required fields: "
            + ", ".join(missing)
        )


def condition_rows(payload):
    rows = []

    for condition in CONDITIONS:
        counts = payload["by_condition"][condition]["counts"]
        rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "n": counts["n"],
                "final_target_resolved": counts[
                    "final_target_resolved"
                ],
                "ever_target_resolved": counts[
                    "ever_target_resolved"
                ],
                "last_validated_target_resolved": counts[
                    "last_validated_target_resolved"
                ],
                "reference_recovery": counts[
                    "reference_recovery"
                ],
                "validated_stop": counts["validated_stop"],
                "validated_state": counts["validated_state"],
                "any_collateral_edit": counts[
                    "any_collateral_edit"
                ],
                "any_new_violation": counts[
                    "any_new_violation"
                ],
            }
        )

    return rows


def first_resolution_rows(payload):
    values = payload["overall"]["counts"]["first_resolution_round"]
    order = ("1", "2", "3", "4", "5", "never")

    return [
        {
            "round": key,
            "label": "Never" if key == "never" else f"Round {key}",
            "count": int(values.get(key, 0)),
        }
        for key in order
        if int(values.get(key, 0)) > 0
    ]


def stop_rows(payload):
    values = payload["overall"]["counts"]["stop_reasons"]
    labels = {
        "validated": "Validated",
        "output_failure": "Output failure",
        "max_rounds": "Maximum rounds",
        "oscillation": "Oscillation",
        "stalled": "Stalled",
        "no_feedback": "No feedback",
    }

    return [
        {
            "stop_reason": key,
            "label": labels[key],
            "count": int(values.get(key, 0)),
        }
        for key in STOP_ORDER
        if int(values.get(key, 0)) > 0
    ]


def latex_escape(value):
    value = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )

    for old, new in replacements:
        value = value.replace(old, new)

    return value


def write_condition_csv(rows, path):
    fields = [
        "condition",
        "n",
        "final_target_resolved",
        "ever_target_resolved",
        "last_validated_target_resolved",
        "reference_recovery",
        "validated_stop",
        "validated_state",
        "any_collateral_edit",
        "any_new_violation",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_condition_tex(rows, path):
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Condition & Final target & Reference & Validated & "
            r"Collateral & New violations & $n$ \\"
        ),
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            "{} & {}/{} & {}/{} & {}/{} & {}/{} & {}/{} & {} \\\\".format(
                latex_escape(row["label"]),
                row["final_target_resolved"],
                row["n"],
                row["reference_recovery"],
                row["n"],
                row["validated_stop"],
                row["n"],
                row["any_collateral_edit"],
                row["n"],
                row["any_new_violation"],
                row["n"],
                row["n"],
            )
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_overall_tex(payload, path):
    counts = payload["overall"]["counts"]
    feedback_n = counts["received_initial_feedback"]

    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Measure & Result \\",
        r"\midrule",
        (
            "End to end target resolution & "
            f"{counts['final_target_resolved']}/{counts['n']} \\\\"
        ),
        (
            "Target resolution given feedback & "
            f"{counts['final_target_resolved_given_feedback']}/"
            f"{feedback_n} \\\\"
        ),
        (
            "Target resolved at least once & "
            f"{counts['ever_target_resolved']}/{counts['n']} \\\\"
        ),
        (
            "Target resolved at last validated state & "
            f"{counts['last_validated_target_resolved']}/"
            f"{counts['n']} \\\\"
        ),
        (
            "Exact clean reference recovery & "
            f"{counts['reference_recovery']}/{counts['n']} \\\\"
        ),
        (
            "Validated stop & "
            f"{counts['validated_stop']}/{counts['n']} \\\\"
        ),
        (
            "Any collateral edit & "
            f"{counts['any_collateral_edit']}/{counts['n']} \\\\"
        ),
        (
            "Any new violation & "
            f"{counts['any_new_violation']}/{counts['n']} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def draft_results_notes(payload):
    counts = payload["overall"]["counts"]
    conditions = {
        row["condition"]: row
        for row in condition_rows(payload)
    }

    ever = counts["ever_target_resolved"]
    regressed = counts.get("graph_regressed_after_resolution", 0)
    failed_after = counts.get("output_failure_after_resolution", 0)

    return "\n\n".join(
        [
            (
                "Target resolution was more common than exact reference "
                "recovery. The controlled target was resolved at the end "
                f"of {counts['final_target_resolved']}/{counts['n']} "
                "trajectories, while "
                f"{counts['reference_recovery']}/{counts['n']} exactly "
                "recovered the clean reference graph."
            ),
            (
                "Most target repairs happened in the first round. "
                f"{counts['first_resolution_round'].get('1', 0)} cases "
                "first resolved the controlled target in round 1."
            ),
            (
                "A resolved target was rarely reintroduced by a later "
                f"valid graph. Among the {ever} cases that resolved the "
                f"target at least once, {regressed} later lost that "
                "resolution in another valid graph, while "
                f"{failed_after} later ended with an unusable model output."
            ),
            (
                "Repair behavior differed by error type. Domain/range "
                "cases had "
                f"{conditions['domain_range']['final_target_resolved']}/"
                f"{conditions['domain_range']['n']} final target "
                "resolutions and "
                f"{conditions['domain_range']['reference_recovery']}/"
                f"{conditions['domain_range']['n']} exact reference "
                "recoveries."
            ),
            (
                "Repair often changed statements beyond the controlled "
                f"target: {counts['any_collateral_edit']}/{counts['n']} "
                "cases had collateral edits and "
                f"{counts['any_new_violation']}/{counts['n']} produced "
                "at least one violation identity that was absent from "
                "the initial feedback."
            ),
        ]
    ) + "\n"


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to make the figures. "
            "Install it with: pip install matplotlib"
        ) from exc
    return plt


def save_figure(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))


def plot_condition_resolution(rows, output_dir):
    plt = _matplotlib()
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        [value - width for value in x],
        [row["final_target_resolved"] for row in rows],
        width,
        label="Final target resolution",
    )
    ax.bar(
        x,
        [row["reference_recovery"] for row in rows],
        width,
        label="Exact reference recovery",
    )
    ax.bar(
        [value + width for value in x],
        [row["validated_stop"] for row in rows],
        width,
        label="Validated stop",
    )
    ax.set_ylabel("Cases")
    ax.set_ylim(0, 10.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend()
    ax.set_title("Repair outcomes by controlled error category")
    save_figure(fig, output_dir / "condition_resolution")
    plt.close(fig)


def plot_condition_side_effects(rows, output_dir):
    plt = _matplotlib()
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        [value - width / 2 for value in x],
        [row["any_collateral_edit"] for row in rows],
        width,
        label="Collateral edit",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [row["any_new_violation"] for row in rows],
        width,
        label="New violation",
    )
    ax.set_ylabel("Cases")
    ax.set_ylim(0, 10.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend()
    ax.set_title("Repair side effects by controlled error category")
    save_figure(fig, output_dir / "condition_side_effects")
    plt.close(fig)


def plot_resolution_retention(rows, output_dir):
    plt = _matplotlib()
    labels = [row["label"] for row in rows]
    x = list(range(len(labels)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        [value - width for value in x],
        [row["ever_target_resolved"] for row in rows],
        width,
        label="Resolved at least once",
    )
    ax.bar(
        x,
        [row["last_validated_target_resolved"] for row in rows],
        width,
        label="Resolved at last validated state",
    )
    ax.bar(
        [value + width for value in x],
        [row["final_target_resolved"] for row in rows],
        width,
        label="End to end resolution",
    )
    ax.set_ylabel("Cases")
    ax.set_ylim(0, 10.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend()
    ax.set_title("Target resolution and retention")
    save_figure(fig, output_dir / "resolution_retention")
    plt.close(fig)


def plot_first_resolution(payload, output_dir):
    plt = _matplotlib()
    rows = first_resolution_rows(payload)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(
        [row["label"] for row in rows],
        [row["count"] for row in rows],
    )
    ax.set_ylabel("Cases")
    ax.set_title("First round in which the controlled target was resolved")
    save_figure(fig, output_dir / "first_resolution")
    plt.close(fig)


def plot_stop_reasons(payload, output_dir):
    plt = _matplotlib()
    rows = stop_rows(payload)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(
        [row["label"] for row in rows],
        [row["count"] for row in rows],
    )
    ax.set_ylabel("Cases")
    ax.set_xticks(
        range(len(rows)),
        [row["label"] for row in rows],
        rotation=20,
        ha="right",
    )
    ax.set_title("Repair trajectory stop reasons")
    save_figure(fig, output_dir / "stop_reasons")
    plt.close(fig)


def build_manifest(payload, analysis_path, output_dir):
    provenance = payload.get("analysis_provenance", {})
    source = payload.get("input", {})

    return {
        "report": "RQ2 repair dynamics figures and tables",
        "analysis_path": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "analysis_git_head": provenance.get("analysis_git_head"),
        "analysis_script_sha256": provenance.get(
            "analysis_script_sha256"
        ),
        "trajectory_sha256": source.get("trajectory_sha256"),
        "run_git_head": source.get("run_git_head"),
        "outputs": sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file()
            and path.name != "repair_reporting_manifest.json"
        ),
        "models_or_validators_run": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        type=Path,
        default=DEFAULT_ANALYSIS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Write tables and notes without importing matplotlib.",
    )
    args = parser.parse_args()

    if not args.analysis.exists():
        raise SystemExit(f"Missing repair analysis: {args.analysis}")

    payload = read_json(args.analysis)
    validate_analysis(payload)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = condition_rows(payload)

    write_condition_csv(
        rows,
        output_dir / "condition_summary.csv",
    )
    write_condition_tex(
        rows,
        output_dir / "condition_summary.tex",
    )
    write_overall_tex(
        payload,
        output_dir / "overall_summary.tex",
    )
    (output_dir / "rq2_results_notes.md").write_text(
        draft_results_notes(payload),
        encoding="utf-8",
    )

    if not args.tables_only:
        plot_condition_resolution(rows, output_dir)
        plot_condition_side_effects(rows, output_dir)
        plot_resolution_retention(rows, output_dir)
        plot_first_resolution(payload, output_dir)
        plot_stop_reasons(payload, output_dir)

    manifest = build_manifest(
        payload,
        args.analysis,
        output_dir,
    )
    (output_dir / "repair_reporting_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote: {output_dir.relative_to(ROOT)}")
    print("No language model or validator was executed.")


if __name__ == "__main__":
    main()
