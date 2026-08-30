from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "results" / "sosa_ssn_feedback_framing_analysis.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "sosa_ssn_feedback_framing_report"

FRAMINGS = ("verdict", "location", "explanation")
LABELS = {
    "verdict": "Verdict",
    "location": "Location",
    "explanation": "Explanation",
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
        raise RuntimeError("unsupported SOSA and SSN feedback framing analysis version")
    integrity = payload.get("integrity", {})
    if integrity.get("observations") != 90 or integrity.get("paired_cases") != 30:
        raise RuntimeError("RQ3 analysis must contain 30 cases and 90 observations")
    if integrity.get("observations_per_framing") != {name: 30 for name in FRAMINGS}:
        raise RuntimeError("RQ3 framing allocation has drifted")
    if integrity.get("complete_paired_matrix") is not True:
        raise RuntimeError("RQ3 paired matrix is incomplete")
    primary = payload.get("primary_outcome", {})
    if primary.get("name") != "controlled_target_removed":
        raise RuntimeError("RQ3 primary outcome changed")
    if set(primary.get("by_framing", {})) != set(FRAMINGS):
        raise RuntimeError("RQ3 primary framing set changed")
    if primary.get("omnibus", {}).get("test") != "Cochran Q":
        raise RuntimeError("RQ3 omnibus test changed")
    if len(primary.get("pairwise", [])) != 3:
        raise RuntimeError("RQ3 pairwise comparison set changed")
    if payload.get("execution", {}).get("models_or_validators_run") is not False:
        raise RuntimeError("analysis claims that models or validators were executed")
    for name in FRAMINGS:
        value = payload.get("secondary_outcomes", {}).get(name, {})
        if value.get("n") != 30:
            raise RuntimeError(f"RQ3 denominator changed for {name}")


def primary_rows(payload: dict) -> list[dict]:
    output = []
    for name in FRAMINGS:
        value = payload["secondary_outcomes"][name]
        output.append({
            "framing": name,
            "label": LABELS[name],
            "n": value["n"],
            "target_removed_count": value["controlled_target_removed"]["count"],
            "target_removed_rate": value["controlled_target_removed"]["rate"],
            "target_removed_lower_95": value["controlled_target_removed"]["lower_95"],
            "target_removed_upper_95": value["controlled_target_removed"]["upper_95"],
            "exact_recovery_count": value["exact_reference_recovery"]["count"],
            "exact_recovery_rate": value["exact_reference_recovery"]["rate"],
            "output_failure_count": value["output_failure"]["count"],
            "output_failure_rate": value["output_failure"]["rate"],
            "usable_output_count": value["usable_outputs"]["count"],
            "usable_output_rate": value["usable_outputs"]["rate"],
        })
    return output


def secondary_rows(payload: dict) -> list[dict]:
    output = []
    for name in FRAMINGS:
        value = payload["secondary_outcomes"][name]
        usable = value["among_usable_outputs"]
        output.append({
            "framing": name,
            "label": LABELS[name],
            "usable_n": value["usable_outputs"]["count"],
            "collateral_count": usable["collateral_edit"]["count"],
            "collateral_rate": usable["collateral_edit"]["rate"],
            "new_shacl_count": usable["new_raw_shacl_findings"]["count"],
            "new_shacl_rate": usable["new_raw_shacl_findings"]["rate"],
            "new_grounding_count": usable["new_grounding_findings"]["count"],
            "new_grounding_rate": usable["new_grounding_findings"]["rate"],
            "mean_edit_from_injected": value["edit_distance_from_injected"]["mean"],
            "median_edit_from_injected": value["edit_distance_from_injected"]["median"],
            "mean_edit_from_clean": value["edit_distance_from_clean_reference"]["mean"],
            "median_edit_from_clean": value["edit_distance_from_clean_reference"]["median"],
        })
    return output


def pairwise_rows(payload: dict) -> list[dict]:
    output = []
    for value in payload["primary_outcome"]["pairwise"]:
        output.append({
            "comparison": f"{value['left']}_vs_{value['right']}",
            "left": value["left"],
            "right": value["right"],
            "n": value["n"],
            "both": value["both"],
            "left_only": value["left_only"],
            "right_only": value["right_only"],
            "neither": value["neither"],
            "risk_difference_left_minus_right": value["risk_difference_left_minus_right"],
            "risk_difference_lower_95": value["risk_difference_lower_95"],
            "risk_difference_upper_95": value["risk_difference_upper_95"],
            "p_value_raw": value["p_value_raw"],
            "p_value_holm": value["p_value_holm"],
            "reject_at_alpha_0_05": value["reject_at_alpha_0_05"],
        })
    return output


def inference_rows(payload: dict) -> list[dict]:
    omnibus = payload["primary_outcome"]["omnibus"]
    return [{
        "test": omnibus["test"],
        "statistic": omnibus["statistic"],
        "degrees_of_freedom": omnibus["degrees_of_freedom"],
        "p_value": omnibus["p_value"],
        "alpha": 0.05,
        "reject_equal_marginal_rates": omnibus["p_value"] < 0.05,
        "multiplicity_correction": payload["primary_outcome"]["multiplicity_correction"],
    }]


def cost_rows(payload: dict) -> list[dict]:
    repair = payload["cost"]["repair_generation"]
    grounding = payload["cost"]["live_grounding"]
    return [
        {
            "stage": "repair_generation",
            "model_calls": repair["calls"],
            "prompt_tokens": repair["prompt_tokens"],
            "generated_tokens": repair["generated_tokens"],
            "recorded_duration_seconds": repair["recorded_duration_seconds"],
            "wall_seconds": payload["cost"]["wall_seconds"],
        },
        {
            "stage": "live_grounding_measurement",
            "model_calls": grounding["unique_calls_within_case"],
            "prompt_tokens": grounding["prompt_tokens"],
            "generated_tokens": grounding["generated_tokens"],
            "recorded_duration_seconds": grounding["recorded_duration_seconds"],
            "wall_seconds": "",
        },
    ]


def write_tables(payload: dict, output_dir: Path) -> dict[str, list[dict]]:
    tables = {
        "primary_outcomes_by_framing.csv": primary_rows(payload),
        "secondary_outcomes_by_framing.csv": secondary_rows(payload),
        "paired_primary_comparisons.csv": pairwise_rows(payload),
        "primary_omnibus_test.csv": inference_rows(payload),
        "cost_summary.csv": cost_rows(payload),
    }
    for name, rows in tables.items():
        write_csv(output_dir / name, rows, list(rows[0]))
    return tables


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to create the RQ3 figures") from exc
    return plt, PercentFormatter


def _style_axis(ax, percent_formatter) -> None:
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(percent_formatter(1.0))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")


def plot_primary(rows: list[dict], output_dir: Path) -> None:
    plt, percent_formatter = _matplotlib()
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.24
    series = (
        ("Target removed", "target_removed_rate", "#0072B2", -width),
        ("Exact recovery", "exact_recovery_rate", "#009E73", 0),
        ("Output failure", "output_failure_rate", "#D55E00", width),
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for label, field, color, offset in series:
        bars = ax.bar(
            [value + offset for value in x],
            [row[field] for row in rows],
            width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, labels=[f"{row[field]:.0%}" for row in rows], padding=3, fontsize=8)
    _style_axis(ax, percent_formatter)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Share of 30 paired cases")
    ax.set_title("One-step repair outcomes by feedback framing")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _save(fig, output_dir, "feedback_framing_primary_outcomes")
    plt.close(fig)


def plot_pairwise(rows: list[dict], output_dir: Path) -> None:
    plt, _percent_formatter = _matplotlib()
    labels = [
        f"{LABELS[row['left']]} vs {LABELS[row['right']]}"
        for row in rows
    ]
    y = list(range(len(rows)))
    height = 0.32
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    left = ax.barh(
        [value - height / 2 for value in y],
        [row["left_only"] for row in rows],
        height,
        color="#0072B2",
        label="Left framing only",
    )
    right = ax.barh(
        [value + height / 2 for value in y],
        [row["right_only"] for row in rows],
        height,
        color="#D55E00",
        label="Right framing only",
    )
    ax.bar_label(left, labels=[str(row["left_only"]) for row in rows], padding=3)
    ax.bar_label(right, labels=[str(row["right_only"]) for row in rows], padding=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Discordant paired cases")
    ax.set_title("Paired transitions in controlled target removal")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _save(fig, output_dir, "feedback_framing_paired_transitions")
    plt.close(fig)


def plot_secondary(rows: list[dict], output_dir: Path) -> None:
    plt, percent_formatter = _matplotlib()
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.24
    series = (
        ("Collateral edit", "collateral_rate", "#E69F00", -width),
        ("New SHACL finding", "new_shacl_rate", "#CC79A7", 0),
        ("New grounding finding", "new_grounding_rate", "#56B4E9", width),
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for label, field, color, offset in series:
        ax.bar(
            [value + offset for value in x],
            [row[field] for row in rows],
            width,
            label=label,
            color=color,
        )
    _style_axis(ax, percent_formatter)
    ax.set_xticks(x, [f"{row['label']}\n(n={row['usable_n']} usable)" for row in rows])
    ax.set_ylabel("Share of usable outputs")
    ax.set_title("Repair side effects by feedback framing")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _save(fig, output_dir, "feedback_framing_secondary_outcomes")
    plt.close(fig)


def build_manifest(analysis_path: Path, output_dir: Path) -> dict:
    outputs = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    return {
        "version": 1,
        "analysis_path": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "models_or_validators_run": False,
        "outputs": outputs,
    }


def run(analysis_path: Path, output_dir: Path) -> dict:
    payload = read_json(analysis_path)
    validate_analysis(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = write_tables(payload, output_dir)
    plot_primary(tables["primary_outcomes_by_framing.csv"], output_dir)
    plot_pairwise(tables["paired_primary_comparisons.csv"], output_dir)
    plot_secondary(tables["secondary_outcomes_by_framing.csv"], output_dir)
    manifest = build_manifest(analysis_path, output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run(args.analysis, args.output_dir)
    print("SOSA and SSN feedback framing report generated")
    print("tables: 5")
    print("figures: 3 PNG + 3 PDF")
    print(f"output: {args.output_dir.resolve()}")
    print("No model, grounding assessor, validator, reasoner, or repair was executed.")


if __name__ == "__main__":
    main()
