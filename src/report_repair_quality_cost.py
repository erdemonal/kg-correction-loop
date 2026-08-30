import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.analyze_repair_quality_cost import PRIMARY_CONDITIONS


ROOT = Path(__file__).resolve().parents[1]
REPORTING_SPEC = ROOT / "experiments" / "repair_quality_cost_reporting_spec.json"
DEFAULT_ANALYSIS = ROOT / "results" / "repair_quality_cost.json"
DEFAULT_DYNAMICS = ROOT / "results" / "repair_dynamics_analysis.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "repair_quality_cost_report"

CONDITION_LABELS = {
    "disjointness": "Disjointness",
    "domain_range": "Domain or range",
    "cardinality": "Cardinality",
    "temporal": "Temporal",
    "grounding": "Grounding",
}
FIGURE_STEMS = (
    "paired_f1_change",
    "condition_f1_change",
    "empty_reference_extras",
    "quality_side_effect_tradeoff",
    "recorded_model_effort",
)
TABLE_FILES = (
    "quality_cost_summary.csv",
    "quality_cost_summary.tex",
    "empty_reference_summary.csv",
    "empty_reference_summary.tex",
)
FAILURE_LABELS = {
    "unparseable_output": "unparseable output",
    "generation_truncated": "truncated generation",
    "relation_outside_allowed_set": "relation outside the allowed set",
}
NOTES_FILE = "repair_quality_cost_notes.md"
MANIFEST_FILE = "repair_quality_cost_reporting_manifest.json"
EXPECTED_ARTIFACTS = (
    tuple(f"{stem}.png" for stem in FIGURE_STEMS)
    + tuple(f"{stem}.pdf" for stem in FIGURE_STEMS)
    + TABLE_FILES
    + (NOTES_FILE,)
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec(path=REPORTING_SPEC):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    execution = spec.get("execution") or {}
    if execution.get("runs_repair_model") or execution.get("runs_validator"):
        raise RuntimeError("Reporting must not run a model or validator")
    if spec.get("view") != "clean reference F1 and repair side effects":
        raise RuntimeError("Reporting must keep clean reference F1 and repair side effects as the view")
    return spec


def validate_analysis(payload):
    if payload.get("analysis_unit") != "controlled case":
        raise RuntimeError("Quality cost analysis uses the controlled case as the unit")
    primary = payload.get("primary_f1") or {}
    empty = payload.get("empty_reference") or {}
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 50:
        raise RuntimeError("Quality cost analysis must contain 50 cases")
    if primary.get("n") != 40 or primary.get("primary") is not True:
        raise RuntimeError("Primary F1 must use the 40 cases with a nonempty clean reference")
    if empty.get("n") != 10:
        raise RuntimeError("Empty reference summary must contain 10 cases")
    if "domain_range" in (primary.get("by_condition") or {}):
        raise RuntimeError("Primary F1 must not include domain_range")
    nonempty = [row for row in cases if not row["empty_reference"]]
    vacant = [row for row in cases if row["empty_reference"]]
    if len(nonempty) != 40 or len(vacant) != 10:
        raise RuntimeError("Empty reference split does not match 40 and 10")
    if any(row["initial_reference_size"] <= 0 for row in nonempty):
        raise RuntimeError("A primary case has an empty initial reference")
    if any(row["initial_reference_size"] != 0 for row in vacant):
        raise RuntimeError("An empty reference case has a nonempty initial reference")


def join_dynamics(payload, dynamics):
    if not isinstance(dynamics, dict) or dynamics.get("analysis_unit") != "controlled case":
        raise RuntimeError("RQ2 dynamics summary is missing or has the wrong unit")
    by_id = {row["id"]: row for row in dynamics.get("cases") or []}
    if len(by_id) != 50:
        raise RuntimeError("RQ2 dynamics summary must contain 50 cases")
    joined = []
    for row in payload["cases"]:
        extra = by_id.get(row["id"])
        if extra is None:
            raise RuntimeError(f"Missing RQ2 dynamics row for {row['id']}")
        merged = dict(row)
        merged["any_collateral_edit"] = bool(extra["any_collateral_edit"])
        merged["any_new_violation"] = bool(extra["any_new_violation"])
        merged["distinct_new_violation_count"] = extra["distinct_new_violation_count"]
        joined.append(merged)
    return joined


def nonempty_cases(cases):
    selected = [row for row in cases if not row["empty_reference"]]
    if any(row["initial_reference_size"] <= 0 for row in selected):
        raise RuntimeError("F1 reporting received an empty reference case")
    return selected


def empty_cases(cases):
    return [row for row in cases if row["empty_reference"]]


def seconds(ns):
    return None if ns is None else ns / 1_000_000_000


def fmt(value, digits=3):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def latex_escape(value):
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(old, new)
    return text


def f1_row(label, condition, block, cases):
    bootstrap = block["bootstrap_mean_delta"]
    return {
        "group": label,
        "condition": condition,
        "n": block["n"],
        "mean_initial_precision": block["mean_initial_precision"],
        "mean_initial_recall": block["mean_initial_recall"],
        "mean_initial_f1": block["mean_initial_f1"],
        "mean_last_validated_precision": block["mean_last_validated_precision"],
        "mean_last_validated_recall": block["mean_last_validated_recall"],
        "mean_last_validated_f1": block["mean_last_validated_f1"],
        "mean_f1_delta": block["mean_delta"],
        "median_f1_delta": block["median_delta"],
        "improved": block["improved"],
        "unchanged": block["unchanged"],
        "worsened": block["worsened"],
        "bootstrap_lower_95": bootstrap["lower_95"],
        "bootstrap_upper_95": bootstrap["upper_95"],
        "bootstrap_samples": bootstrap["samples"],
        "bootstrap_seed": bootstrap["seed"],
        "end_to_end_target_resolved": block["end_to_end_target_resolved"],
        "last_validated_target_resolved": block["last_validated_target_resolved"],
        "output_failure": block["output_failure"],
        "validated_state": block["validated_state"],
        "sum_repair_calls": sum(row["repair_calls"] for row in cases),
        "sum_grounding_assessor_calls": sum(row["grounding_assessor_calls"] for row in cases),
        "repair_duration_s": seconds(sum(row["repair_duration_ns"] for row in cases)),
        "grounding_duration_s": seconds(sum(row["grounding_duration_ns"] for row in cases)),
        "any_collateral_edit": sum(row.get("any_collateral_edit", 0) for row in cases),
        "any_new_violation": sum(row.get("any_new_violation", 0) for row in cases),
    }


def quality_cost_rows(payload, cases):
    primary = nonempty_cases(cases)
    block = payload["primary_f1"]
    rows = [f1_row("Primary overall", "primary", block, primary)]
    for condition in PRIMARY_CONDITIONS:
        selected = [row for row in primary if row["condition"] == condition]
        rows.append(
            f1_row(
                CONDITION_LABELS[condition],
                condition,
                block["by_condition"][condition],
                selected,
            )
        )
    return rows


def empty_reference_rows(payload):
    empty = payload["empty_reference"]
    return [
        {
            "n": empty["n"],
            "primary_metric": empty["primary_metric"],
            "exact_empty_graph_recovery": empty["exact_empty_graph_recovery"],
            "exact_empty_graph_recovery_rate": empty["exact_empty_graph_recovery_rate"],
            "mean_initial_extra_triples": empty["mean_initial_extra_triples"],
            "mean_last_validated_extra_triples": empty["mean_last_validated_extra_triples"],
            "median_initial_extra_triples": empty["median_initial_extra_triples"],
            "median_last_validated_extra_triples": empty["median_last_validated_extra_triples"],
            "mean_extra_delta": empty["mean_extra_delta"],
            "median_extra_delta": empty["median_extra_delta"],
            "improved": empty["improved"],
            "unchanged": empty["unchanged"],
            "worsened": empty["worsened"],
            "end_to_end_target_resolved": empty["end_to_end_target_resolved"],
            "last_validated_target_resolved": empty["last_validated_target_resolved"],
            "output_failure": empty["output_failure"],
            "validated_state": empty["validated_state"],
        }
    ]


def write_csv(rows, path):
    if not rows:
        raise RuntimeError("Cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_quality_cost_tex(rows, path):
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        (
            r"Group & $n$ & Initial F1 & Last F1 & Mean $\Delta$ & "
            r"Improved & Unchanged & Worsened & Target \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {}/{} \\\\".format(
                latex_escape(row["group"]),
                row["n"],
                fmt(row["mean_initial_f1"]),
                fmt(row["mean_last_validated_f1"]),
                fmt(row["mean_f1_delta"]),
                row["improved"],
                row["unchanged"],
                row["worsened"],
                row["end_to_end_target_resolved"],
                row["n"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_empty_reference_tex(rows, path):
    row = rows[0]
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Measure & Result \\",
        r"\midrule",
        f"Empty reference cases & {row['n']} \\\\",
        (
            "Exact empty graph recovery & "
            f"{row['exact_empty_graph_recovery']}/{row['n']} \\\\"
        ),
        f"Mean extra triples at start & {fmt(row['mean_initial_extra_triples'], 1)} \\\\",
        (
            "Mean extra triples at last validated graph & "
            f"{fmt(row['mean_last_validated_extra_triples'], 1)} \\\\"
        ),
        f"Mean extra triple change & {fmt(row['mean_extra_delta'], 1)} \\\\",
        (
            "Improved / unchanged / worsened & "
            f"{row['improved']} / {row['unchanged']} / {row['worsened']} \\\\"
        ),
        (
            "End to end target resolution & "
            f"{row['end_to_end_target_resolved']}/{row['n']} \\\\"
        ),
        f"Output failure & {row['output_failure']}/{row['n']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def draft_results_notes(payload, cases):
    primary = payload["primary_f1"]
    empty = payload["empty_reference"]
    convention = payload["all_case_convention_based_summary"]
    bootstrap = primary["bootstrap_mean_delta"]
    overall = payload["overall"]
    repair_s = seconds(overall["sum_repair_duration_ns"])
    grounding_s = seconds(overall["sum_grounding_duration_ns"])
    recorded_s = seconds(overall["sum_recorded_model_duration_ns"])
    failures = Counter(
        row["output_failure"] for row in cases if row["output_failure"]
    )
    failure_text = ", ".join(
        f"{FAILURE_LABELS.get(name, name)}: {count}"
        for name, count in sorted(failures.items())
    )
    conditions = primary["by_condition"]
    return "\n".join(
        [
            "# Repair quality and recorded model cost",
            "",
            "The figures report how close the repaired graph is to the controlled clean graph. This is not a human judgment of source faithfulness. It is not Text2KGBench F1.",
            "",
            "The main F1 results use 40 cases. Each of those cases has at least one triple in the clean reference at the start. The 10 domain_range cases have an empty clean reference. Those 10 are shown with extra triple counts, not with F1. They do not appear in the main F1 figures or the main F1 table.",
            "",
            (
                f"On the 40 cases, mean F1 rose from {fmt(primary['mean_initial_f1'])} "
                f"at the injected start to {fmt(primary['mean_last_validated_f1'])} "
                f"at the last validated graph. The mean paired change was "
                f"{fmt(primary['mean_delta'])} (median {fmt(primary['median_delta'])}). "
                f"{primary['improved']} cases improved, {primary['unchanged']} were unchanged, "
                f"and {primary['worsened']} worsened. A case bootstrap with "
                f"{bootstrap['samples']} resamples and seed {bootstrap['seed']} gives the interval "
                f"[{fmt(bootstrap['lower_95'])}, {fmt(bootstrap['upper_95'])}] for that mean change. "
                "The interval describes resampling of these controlled cases. It is not a population confidence interval."
            ),
            "",
            (
                "On the same 40 cases, disjointness fell from "
                f"{fmt(conditions['disjointness']['mean_initial_f1'])} to "
                f"{fmt(conditions['disjointness']['mean_last_validated_f1'])}. Cardinality rose from "
                f"{fmt(conditions['cardinality']['mean_initial_f1'])} to "
                f"{fmt(conditions['cardinality']['mean_last_validated_f1'])}. Temporal rose from "
                f"{fmt(conditions['temporal']['mean_initial_f1'])} to "
                f"{fmt(conditions['temporal']['mean_last_validated_f1'])}. Grounding rose from "
                f"{fmt(conditions['grounding']['mean_initial_f1'])} to "
                f"{fmt(conditions['grounding']['mean_last_validated_f1'])}. "
                "Disjointness is the only one of these four conditions whose mean F1 fell."
            ),
            "",
            (
                f"The 10 domain_range cases began with "
                f"{fmt(empty['mean_initial_extra_triples'], 1)} extra triples on average "
                f"and ended with {fmt(empty['mean_last_validated_extra_triples'], 1)}. "
                f"None recovered the empty graph ({empty['exact_empty_graph_recovery']}/10). "
                f"Extra counts fell in {empty['improved']} cases, stayed the same in "
                f"{empty['unchanged']}, and rose in {empty['worsened']}. "
                "Assigning F1 of 0 when the prediction is nonempty and the reference is empty is a scoring rule, not a quality result for those cases."
            ),
            "",
            (
                f"The file also stores a 50 case summary that still mixes in those 10 empty reference cases "
                f"(mean F1 {fmt(convention['mean_initial_f1'])} to "
                f"{fmt(convention['mean_last_validated_f1'])}). That summary is not the main F1 result."
            ),
            "",
            (
                "Collateral edits and new violations come from the locked RQ2 case records. "
                "The scatter of F1 change against collateral edits is descriptive. "
                "It does not define a Pareto front."
            ),
            "",
            (
                f"The 50 case run used {overall['sum_repair_calls']} repair generations "
                f"({fmt(repair_s, 1)} s) and {overall['sum_grounding_assessor_calls']} "
                f"live grounding assessor calls ({fmt(grounding_s, 1)} s). "
                f"Combined Ollama time was {fmt(recorded_s, 1)} s, or about "
                f"{fmt(recorded_s / 50, 1)} s per case. "
                "These times are Ollama durations. They are not wall clock time, "
                "reasoner time, SHACL time, or money."
            ),
            "",
            (
                f"{overall['output_failure']} of 50 trajectories ended in an output failure"
                + (f" ({failure_text})" if failure_text else "")
                + ". A failed round has no graph. F1 is omitted for that round. It is not stored as 0."
            ),
            "",
            "Means for later rounds include only the cases still running. They are not a follow up of the same 40 cases at every round.",
            "",
        ]
    )


def matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to make the figures") from exc
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10})
    return plt


def save_figure(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))


def condition_colors(plt):
    cmap = plt.get_cmap("tab10")
    return {
        "disjointness": cmap(0),
        "cardinality": cmap(1),
        "temporal": cmap(2),
        "grounding": cmap(3),
        "domain_range": cmap(4),
    }


def plot_paired_f1_change(cases, payload, output_dir):
    plt = matplotlib_pyplot()
    selected = nonempty_cases(cases)
    colors = condition_colors(plt)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    ax = axes[0]
    for condition in PRIMARY_CONDITIONS:
        points = [row for row in selected if row["condition"] == condition]
        ax.scatter(
            [row["initial_f1"] for row in points],
            [row["last_validated_f1"] for row in points],
            label=CONDITION_LABELS[condition],
            color=colors[condition],
            alpha=0.85,
            edgecolors="none",
        )
    ax.plot([0, 1], [0, 1], color="0.4", linewidth=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Initial F1")
    ax.set_ylabel("Last validated F1")
    ax.set_title("Initial and last F1 for 40 cases with a clean reference")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    deltas = [row["f1_delta"] for row in selected]
    ax.hist(deltas, bins=12, color="0.35", edgecolor="white")
    ax.axvline(0, color="0.2", linewidth=1)
    primary = payload["primary_f1"]
    ax.set_xlabel("Last F1 minus initial F1")
    ax.set_ylabel("Cases")
    ax.set_title(
        f"{primary['improved']} improved, {primary['unchanged']} unchanged, "
        f"{primary['worsened']} worsened"
    )
    save_figure(fig, output_dir / "paired_f1_change")
    plt.close(fig)
    return selected


def plot_condition_f1_change(payload, output_dir):
    plt = matplotlib_pyplot()
    block = payload["primary_f1"]["by_condition"]
    labels = [CONDITION_LABELS[name] for name in PRIMARY_CONDITIONS]
    initial = [block[name]["mean_initial_f1"] for name in PRIMARY_CONDITIONS]
    last = [block[name]["mean_last_validated_f1"] for name in PRIMARY_CONDITIONS]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar([value - width / 2 for value in x], initial, width, label="Initial")
    ax.bar([value + width / 2 for value in x], last, width, label="Last validated")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean F1")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.set_title("Mean F1 by error type")
    save_figure(fig, output_dir / "condition_f1_change")
    plt.close(fig)
    return list(PRIMARY_CONDITIONS)


def plot_empty_reference_extras(payload, cases, output_dir):
    plt = matplotlib_pyplot()
    selected = empty_cases(cases)
    empty = payload["empty_reference"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    ax = axes[0]
    ax.scatter(
        [row["initial_extra_triples"] for row in selected],
        [row["last_validated_extra_triples"] for row in selected],
        color="0.2",
    )
    lim = max(
        1,
        max(row["initial_extra_triples"] for row in selected),
        max(row["last_validated_extra_triples"] for row in selected),
    ) + 0.5
    ax.plot([0, lim], [0, lim], color="0.5", linewidth=1)
    ax.set_xlim(-0.3, lim)
    ax.set_ylim(-0.3, lim)
    ax.set_xlabel("Initial extra triples")
    ax.set_ylabel("Last validated extra triples")
    ax.set_title("Domain or range cases with an empty clean reference")
    ax = axes[1]
    ax.bar(
        ["Fell", "Stayed", "Rose"],
        [empty["improved"], empty["unchanged"], empty["worsened"]],
        color=["0.45", "0.65", "0.25"],
    )
    ax.set_ylim(0, 10)
    ax.set_ylabel("Cases")
    ax.set_title("Change in extra triples")
    save_figure(fig, output_dir / "empty_reference_extras")
    plt.close(fig)
    return selected


def plot_quality_side_effect_tradeoff(cases, output_dir):
    plt = matplotlib_pyplot()
    selected = nonempty_cases(cases)
    colors = condition_colors(plt)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    ax = axes[0]
    for condition in PRIMARY_CONDITIONS:
        points = [row for row in selected if row["condition"] == condition]
        ax.scatter(
            [
                row["last_validated_collateral_removed"] + row["last_validated_collateral_added"]
                for row in points
            ],
            [row["f1_delta"] for row in points],
            label=CONDITION_LABELS[condition],
            color=colors[condition],
            alpha=0.85,
            edgecolors="none",
        )
    ax.axhline(0, color="0.4", linewidth=1)
    ax.set_xlabel("Collateral triples at the last validated graph")
    ax.set_ylabel("F1 change")
    ax.set_title("F1 change and collateral edits")
    ax.legend(fontsize=8)
    ax = axes[1]
    labels = [CONDITION_LABELS[name] for name in PRIMARY_CONDITIONS]
    x = list(range(len(labels)))
    width = 0.24
    collateral = []
    new_viol = []
    target = []
    for condition in PRIMARY_CONDITIONS:
        points = [row for row in selected if row["condition"] == condition]
        collateral.append(sum(row["any_collateral_edit"] for row in points))
        new_viol.append(sum(row["any_new_violation"] for row in points))
        target.append(sum(row["end_to_end_target_resolved"] for row in points))
    ax.bar([value - width for value in x], collateral, width, label="Any collateral edit")
    ax.bar(x, new_viol, width, label="Any new violation")
    ax.bar([value + width for value in x], target, width, label="End to end target resolution")
    ax.set_ylim(0, 10.8)
    ax.set_ylabel("Cases")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.legend(fontsize=8)
    ax.set_title("Collateral edits, new violations, and target resolution")
    save_figure(fig, output_dir / "quality_side_effect_tradeoff")
    plt.close(fig)
    return selected


def plot_recorded_model_effort(cases, output_dir):
    plt = matplotlib_pyplot()
    order = PRIMARY_CONDITIONS + ("domain_range",)
    labels = [CONDITION_LABELS[name] for name in order]
    repair_s = []
    grounding_s = []
    repair_calls = []
    grounding_calls = []
    repair_tokens = []
    grounding_tokens = []
    failures = []
    for condition in order:
        points = [row for row in cases if row["condition"] == condition]
        repair_s.append(seconds(sum(row["repair_duration_ns"] for row in points)))
        grounding_s.append(seconds(sum(row["grounding_duration_ns"] for row in points)))
        repair_calls.append(sum(row["repair_calls"] for row in points))
        grounding_calls.append(sum(row["grounding_assessor_calls"] for row in points))
        repair_tokens.append(
            sum(row["repair_prompt_eval_count"] + row["repair_eval_count"] for row in points)
        )
        grounding_tokens.append(
            sum(
                row["grounding_prompt_eval_count"] + row["grounding_eval_count"]
                for row in points
            )
        )
        failures.append(sum(row["output_failure"] is not None for row in points))
    x = list(range(len(labels)))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
    ax = axes[0, 0]
    ax.bar([value - width / 2 for value in x], repair_s, width, label="Repair model")
    ax.bar([value + width / 2 for value in x], grounding_s, width, label="Live grounding assessor")
    ax.set_ylabel("Ollama duration (seconds)")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend(fontsize=7)
    ax.set_title("Recorded model duration")
    ax = axes[0, 1]
    ax.bar([value - width / 2 for value in x], repair_tokens, width, label="Repair model")
    ax.bar(
        [value + width / 2 for value in x],
        grounding_tokens,
        width,
        label="Live grounding assessor",
    )
    ax.set_ylabel("Tokens")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend(fontsize=7)
    ax.set_title("Recorded tokens")
    ax = axes[1, 0]
    ax.bar([value - width / 2 for value in x], repair_calls, width, label="Repair generations")
    ax.bar(
        [value + width / 2 for value in x],
        grounding_calls,
        width,
        label="Live grounding assessor calls",
    )
    ax.set_ylabel("Calls")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend(fontsize=7)
    ax.set_title("Recorded calls")
    ax = axes[1, 1]
    ax.bar(labels, failures, color="0.35")
    ax.set_ylabel("Cases")
    ax.set_ylim(0, 10.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_title("Output failures")
    save_figure(fig, output_dir / "recorded_model_effort")
    plt.close(fig)
    return {
        "repair_calls": repair_calls,
        "grounding_calls": grounding_calls,
        "failures": failures,
    }


def build_manifest(payload, spec, analysis_path, dynamics_path, output_dir):
    return {
        "report": "Repair quality and recorded model cost figures and tables",
        "view": spec["view"],
        "analysis_path": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "rq2_path": str(dynamics_path.resolve()),
        "rq2_sha256": sha256_file(dynamics_path),
        "trajectory_sha256": (payload.get("input") or {}).get("trajectory_sha256"),
        "primary_n": payload["primary_f1"]["n"],
        "empty_reference_n": payload["empty_reference"]["n"],
        "outputs": sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file() and path.name != MANIFEST_FILE
        ),
        "models_or_validators_run": False,
    }


def run_report(
    analysis_path=DEFAULT_ANALYSIS,
    dynamics_path=DEFAULT_DYNAMICS,
    spec_path=REPORTING_SPEC,
    output_dir=DEFAULT_OUTPUT_DIR,
    tables_only=False,
):
    spec = load_spec(spec_path)
    payload = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    dynamics = json.loads(Path(dynamics_path).read_text(encoding="utf-8"))
    validate_analysis(payload)
    cases = join_dynamics(payload, dynamics)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_rows = quality_cost_rows(payload, cases)
    empty_rows = empty_reference_rows(payload)
    write_csv(quality_rows, output_dir / "quality_cost_summary.csv")
    write_quality_cost_tex(quality_rows, output_dir / "quality_cost_summary.tex")
    write_csv(empty_rows, output_dir / "empty_reference_summary.csv")
    write_empty_reference_tex(empty_rows, output_dir / "empty_reference_summary.tex")
    (output_dir / NOTES_FILE).write_text(draft_results_notes(payload, cases), encoding="utf-8")
    if not tables_only:
        plot_paired_f1_change(cases, payload, output_dir)
        plot_condition_f1_change(payload, output_dir)
        plot_empty_reference_extras(payload, cases, output_dir)
        plot_quality_side_effect_tradeoff(cases, output_dir)
        plot_recorded_model_effort(cases, output_dir)
    manifest = build_manifest(payload, spec, analysis_path, dynamics_path, output_dir)
    (output_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--dynamics", type=Path, default=DEFAULT_DYNAMICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    manifest = run_report(
        analysis_path=args.analysis,
        dynamics_path=args.dynamics,
        output_dir=args.output_dir,
        tables_only=args.tables_only,
    )
    print(f"wrote: {args.output_dir}")
    for name in manifest["outputs"]:
        print(f"  {name}")
    print("No language model or validator was executed.")


if __name__ == "__main__":
    main()
