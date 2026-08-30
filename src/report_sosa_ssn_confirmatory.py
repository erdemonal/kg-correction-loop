from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "results" / "sosa_ssn_confirmatory_analysis.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "sosa_ssn_confirmatory_report"

CONDITIONS = (
    "cardinality",
    "disjointness",
    "domain_range",
    "functional_property_conflict",
    "grounding",
    "temporal",
)

LABELS = {
    "cardinality": "Cardinality",
    "disjointness": "Disjointness",
    "domain_range": "Domain/range",
    "functional_property_conflict": "Functional",
    "grounding": "Grounding",
    "temporal": "Temporal",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_analysis(payload: dict) -> None:
    if payload.get("version") != 1:
        raise RuntimeError("unsupported confirmatory analysis version")
    integrity = payload.get("integrity", {})
    if integrity.get("cases") != 180 or integrity.get("unique_case_ids") != 180:
        raise RuntimeError("confirmatory analysis must contain 180 unique cases")
    if integrity.get("cases_per_condition") != {name: 30 for name in CONDITIONS}:
        raise RuntimeError("confirmatory condition allocation has drifted")
    for section in ("grounding", "repair"):
        groups = payload.get(section, {}).get("by_condition", {})
        if set(groups) != set(CONDITIONS):
            raise RuntimeError(f"{section} condition set has drifted")
        if any(groups[name].get("n") != 30 for name in CONDITIONS):
            raise RuntimeError(f"{section} condition denominator has drifted")
    coverage = payload.get("validator_coverage_at_round_zero", {})
    if set(coverage.get("by_condition", {})) != set(CONDITIONS):
        raise RuntimeError("validator coverage condition set has drifted")


def interval_fields(value: dict) -> dict:
    return {
        "count": value["count"],
        "n": value["n"],
        "rate": value["rate"],
        "lower_95": value["lower_95"],
        "upper_95": value["upper_95"],
    }


def validator_rows(payload: dict) -> list[dict]:
    groups = payload["validator_coverage_at_round_zero"]["by_condition"]
    rows = []
    for condition in CONDITIONS:
        value = groups[condition]
        rows.append(
            {
                "condition": condition,
                "label": LABELS[condition],
                "n": value["n"],
                "shacl_count": value["raw_shacl"]["count"],
                "shacl_rate": value["raw_shacl"]["rate"],
                "owl_count": value["owl_consistency"]["count"],
                "owl_rate": value["owl_consistency"]["rate"],
                "grounding_count": value["grounding_v3"]["count"],
                "grounding_rate": value["grounding_v3"]["rate"],
            }
        )
    return rows


def repair_rows(payload: dict) -> list[dict]:
    groups = payload["repair"]["by_condition"]
    rows = []
    for condition in CONDITIONS:
        value = groups[condition]
        rows.append(
            {
                "condition": condition,
                "label": LABELS[condition],
                "n": value["n"],
                "ever_count": value["ever_target_resolution"]["count"],
                "ever_rate": value["ever_target_resolution"]["rate"],
                "final_count": value["end_to_end_target_resolution"]["count"],
                "final_rate": value["end_to_end_target_resolution"]["rate"],
                "validated_count": value["validated_state"]["count"],
                "validated_rate": value["validated_state"]["rate"],
                "exact_count": value["end_to_end_reference_recovery"]["count"],
                "exact_rate": value["end_to_end_reference_recovery"]["rate"],
                "output_failure_count": value["output_failure"]["count"],
                "output_failure_rate": value["output_failure"]["rate"],
                "collateral_count": value["any_collateral_edit"]["count"],
                "collateral_rate": value["any_collateral_edit"]["rate"],
                "new_violation_count": value["any_new_violation"]["count"],
                "new_violation_rate": value["any_new_violation"]["rate"],
                "f1_improved": value["paired_f1_changes"].get("improved", 0),
                "f1_unchanged": value["paired_f1_changes"].get("unchanged", 0),
                "f1_worsened": value["paired_f1_changes"].get("worsened", 0),
                "mean_initial_f1": value["mean_initial_f1"],
                "mean_last_validated_f1": value["mean_last_validated_f1"],
                "mean_f1_change": value["mean_paired_f1_change"],
                "mean_repair_rounds": value["mean_repair_rounds"],
            }
        )
    return rows


def overall_rows(payload: dict) -> list[dict]:
    grounding = payload["grounding"]["overall"]
    coverage = payload["validator_coverage_at_round_zero"]["overall"]
    repair = payload["repair"]["overall"]
    metrics = [
        ("grounding_target_match", grounding["target_matches_expected"]),
        ("shacl_target_coverage", coverage["raw_shacl"]),
        ("owl_target_coverage", coverage["owl_consistency"]),
        ("grounding_target_coverage", coverage["grounding_v3"]),
        ("ever_target_resolution", repair["ever_target_resolution"]),
        ("final_target_resolution", repair["end_to_end_target_resolution"]),
        ("validated_state", repair["validated_state"]),
        ("exact_reference_recovery", repair["end_to_end_reference_recovery"]),
        ("output_failure", repair["output_failure"]),
        ("any_collateral_edit", repair["any_collateral_edit"]),
        ("any_new_violation", repair["any_new_violation"]),
    ]
    return [{"metric": name, **interval_fields(value)} for name, value in metrics]


def cost_rows(payload: dict) -> list[dict]:
    grounding = payload["grounding"]["initial_cost"]
    repair = payload["repair"]["cost"]
    return [
        {
            "stage": "initial_grounding",
            "model_calls": grounding["calls"],
            "prompt_tokens": grounding["prompt_tokens"],
            "generated_tokens": grounding["generated_tokens"],
            "recorded_duration_seconds": grounding["duration_seconds"],
            "wall_seconds": grounding["wall_seconds"],
        },
        {
            "stage": "repair_generation",
            "model_calls": repair["repair_calls"],
            "prompt_tokens": repair["repair_prompt_tokens"],
            "generated_tokens": repair["repair_generated_tokens"],
            "recorded_duration_seconds": repair["repair_duration_seconds"],
            "wall_seconds": repair["wall_seconds"],
        },
        {
            "stage": "repair_live_grounding",
            "model_calls": repair["live_grounding_calls"],
            "prompt_tokens": repair["live_grounding_prompt_tokens"],
            "generated_tokens": repair["live_grounding_generated_tokens"],
            "recorded_duration_seconds": repair["live_grounding_duration_seconds"],
            "wall_seconds": "",
        },
    ]


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create the figures") from exc
    return plt, PercentFormatter


def _style_axis(ax, percent_formatter) -> None:
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(percent_formatter(1.0))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")


def plot_validator_coverage(rows: list[dict], output_dir: Path) -> None:
    plt, percent_formatter = _matplotlib()
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    series = (
        ("SHACL", "shacl_rate", "#0072B2", -width),
        ("OWL 2 DL", "owl_rate", "#D55E00", 0),
        ("Grounding", "grounding_rate", "#009E73", width),
    )
    for label, field, color, offset in series:
        ax.bar([value + offset for value in x], [row[field] for row in rows], width, label=label, color=color)
    _style_axis(ax, percent_formatter)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Controlled targets detected")
    ax.set_title("Validator coverage by error condition")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    _save(fig, output_dir, "validator_coverage_by_condition")
    plt.close(fig)


def plot_repair_outcomes(rows: list[dict], output_dir: Path) -> None:
    plt, percent_formatter = _matplotlib()
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.19
    series = (
        ("Resolved at least once", "ever_rate", "#56B4E9", -1.5 * width),
        ("Resolved at final output", "final_rate", "#0072B2", -0.5 * width),
        ("Validated final state", "validated_rate", "#009E73", 0.5 * width),
        ("Exact clean recovery", "exact_rate", "#CC79A7", 1.5 * width),
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for label, field, color, offset in series:
        ax.bar([value + offset for value in x], [row[field] for row in rows], width, label=label, color=color)
    _style_axis(ax, percent_formatter)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Share of cases")
    ax.set_title("Repair outcomes by error condition")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    _save(fig, output_dir, "repair_outcomes_by_condition")
    plt.close(fig)


def plot_repair_risks(rows: list[dict], output_dir: Path) -> None:
    plt, percent_formatter = _matplotlib()
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.24
    series = (
        ("Output failure", "output_failure_rate", "#D55E00", -width),
        ("Collateral edit", "collateral_rate", "#E69F00", 0),
        ("New violation", "new_violation_rate", "#CC79A7", width),
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for label, field, color, offset in series:
        ax.bar([value + offset for value in x], [row[field] for row in rows], width, label=label, color=color)
    _style_axis(ax, percent_formatter)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Share of cases")
    ax.set_title("Repair risks by error condition")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    _save(fig, output_dir, "repair_risks_by_condition")
    plt.close(fig)


def plot_f1_changes(rows: list[dict], output_dir: Path) -> None:
    plt, percent_formatter = _matplotlib()
    labels = [row["label"] for row in rows]
    improved = [row["f1_improved"] / row["n"] for row in rows]
    unchanged = [row["f1_unchanged"] / row["n"] for row in rows]
    worsened = [row["f1_worsened"] / row["n"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(labels, improved, label="Improved", color="#009E73")
    ax.bar(labels, unchanged, bottom=improved, label="Unchanged", color="#BDBDBD")
    bases = [a + b for a, b in zip(improved, unchanged)]
    ax.bar(labels, worsened, bottom=bases, label="Worsened", color="#D55E00")
    _style_axis(ax, percent_formatter)
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    ax.set_ylabel("Share of cases")
    ax.set_title("Paired graph F1 change after repair")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    _save(fig, output_dir, "paired_f1_change_by_condition")
    plt.close(fig)


def write_tables(payload: dict, output_dir: Path) -> tuple[list[dict], list[dict]]:
    validators = validator_rows(payload)
    repairs = repair_rows(payload)
    write_csv(
        output_dir / "overall_summary.csv",
        overall_rows(payload),
        ["metric", "count", "n", "rate", "lower_95", "upper_95"],
    )
    write_csv(
        output_dir / "validator_coverage_by_condition.csv",
        validators,
        list(validators[0]),
    )
    write_csv(
        output_dir / "repair_outcomes_by_condition.csv",
        repairs,
        list(repairs[0]),
    )
    costs = cost_rows(payload)
    write_csv(output_dir / "cost_summary.csv", costs, list(costs[0]))
    return validators, repairs


def build_manifest(analysis_path: Path, output_dir: Path) -> dict:
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json")
    return {
        "analysis_path": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "outputs": {path.name: sha256_file(path) for path in files},
        "models_or_validators_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()

    if not args.analysis.exists():
        raise SystemExit(f"Missing confirmatory analysis: {args.analysis}")
    payload = read_json(args.analysis)
    validate_analysis(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validators, repairs = write_tables(payload, args.output_dir)
    if not args.tables_only:
        plot_validator_coverage(validators, args.output_dir)
        plot_repair_outcomes(repairs, args.output_dir)
        plot_repair_risks(repairs, args.output_dir)
        plot_f1_changes(repairs, args.output_dir)
    manifest = build_manifest(args.analysis, args.output_dir)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("confirmatory report generated")
    print(f"tables: 4")
    print(f"figures: {0 if args.tables_only else 4} PNG + {0 if args.tables_only else 4} PDF")
    print(f"output: {args.output_dir.resolve()}")
    print("No model, grounding assessor, validator, reasoner, or repair was executed.")


if __name__ == "__main__":
    main()
