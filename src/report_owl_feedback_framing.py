import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "results" / "owl_feedback_framing_analysis.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "owl_feedback_report"
REPORTING_SPEC = ROOT / "experiments" / "owl_feedback_reporting_spec.json"
CONDITIONS = ("verdict", "location", "explanation")
LABELS = {
    "verdict": "Verdict",
    "location": "Verdict + location",
    "explanation": "Verdict + explanation",
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_analysis(payload):
    if payload.get("analysis_unit") != "controlled case" or payload.get("paired_by_case") is not True:
        raise RuntimeError("OWL framing analysis must preserve pairing by controlled case")
    if payload.get("n_paired_cases") != 10 or payload.get("case_condition_observations") != 30:
        raise RuntimeError("OWL framing analysis must contain ten cases and 30 observations")
    if tuple(payload.get("conditions", ())) != CONDITIONS:
        raise RuntimeError("OWL framing condition order changed")
    if not isinstance(payload.get("cases"), list) or len(payload["cases"]) != 10:
        raise RuntimeError("OWL framing case records are incomplete")
    for condition in CONDITIONS:
        summary = payload.get("by_condition", {}).get(condition)
        if not isinstance(summary, dict) or summary.get("counts", {}).get("n") != 10:
            raise RuntimeError(f"Missing or incomplete condition: {condition}")
        counts, rates = summary["counts"], summary["rates"]
        if counts["usable_outputs"] + counts["output_failures"] != counts["n"]:
            raise RuntimeError("Usable outputs and failures must sum to paired cases")
        for field in ("owl_consistent", "collateral_edit", "new_raw_shacl_findings", "new_grounding_findings"):
            if rates[field]["denominator"] != counts["usable_outputs"]:
                raise RuntimeError(f"Denominator for outcomes that depend on a parsed graph is incorrect for {field}")


def fraction(numerator, denominator):
    return f"{numerator}/{denominator}" if denominator else "not observed"


def condition_rows(payload):
    rows = []
    for condition in CONDITIONS:
        item = payload["by_condition"][condition]
        counts = item["counts"]
        usable = counts["usable_outputs"]
        total = counts["n"]
        rows.append(
            {
                "condition": condition,
                "label": LABELS[condition],
                "n_paired_cases": total,
                "usable_outputs": usable,
                "target_removed": fraction(counts["controlled_target_removed"], total),
                "owl_consistent_given_usable": fraction(counts["owl_consistent"], usable),
                "exact_reference_recovery": fraction(counts["reference_recovery"], total),
                "collateral_given_usable": fraction(counts["collateral_edit"], usable),
                "new_shacl_given_usable": fraction(counts["new_raw_shacl_findings"], usable),
                "new_grounding_given_usable": fraction(counts["new_grounding_findings"], usable),
                "residual_owl_after_target_removal": fraction(
                    counts["owl_inconsistent_after_target_removal"],
                    counts["controlled_target_removed"],
                ),
                "output_failures": fraction(counts["output_failures"], total),
                "mean_edit_distance_given_usable": round(item["edits"]["mean_from_injected"], 3),
            }
        )
    return rows


def paired_rows(payload):
    rows = []
    for name, item in payload["paired_target_comparisons"].items():
        rows.append(
            {
                "comparison": name,
                "left": LABELS[item["left"]],
                "right": LABELS[item["right"]],
                "paired_cases": item["n_paired_cases"],
                "both_resolved": item["both_resolved"],
                "left_only": item["left_only"],
                "right_only": item["right_only"],
                "neither_resolved": item["neither_resolved"],
                "same": item["same"],
                "net_target_difference": item["net_target_difference"],
            }
        )
    return rows


def domain_rows(payload):
    rows = []
    for domain in ("movie", "music"):
        for condition in CONDITIONS:
            counts = payload["by_domain"][domain][condition]["counts"]
            usable = counts["usable_outputs"]
            total = counts["n"]
            rows.append(
                {
                    "domain": domain,
                    "condition": condition,
                    "paired_cases": total,
                    "usable_outputs": usable,
                    "target_removed": fraction(counts["controlled_target_removed"], total),
                    "owl_consistent_given_usable": fraction(counts["owl_consistent"], usable),
                    "exact_reference_recovery": fraction(counts["reference_recovery"], total),
                    "collateral_given_usable": fraction(counts["collateral_edit"], usable),
                    "new_shacl_given_usable": fraction(counts["new_raw_shacl_findings"], usable),
                    "new_grounding_given_usable": fraction(counts["new_grounding_findings"], usable),
                    "output_failures": fraction(counts["output_failures"], total),
                }
            )
    return rows


def write_csv(rows, path):
    if not rows:
        raise RuntimeError("Cannot create a publication table from zero rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_condition_latex(rows, path):
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"Feedback & Target & OWL$^\dagger$ & Reference & Collateral$^\dagger$ & New SHACL$^\dagger$ & New grounding$^\dagger$ & Failure " + "\\" * 2,
        r"\hline",
    ]
    for row in rows:
        cells = [
            row["label"], row["target_removed"], row["owl_consistent_given_usable"],
            row["exact_reference_recovery"], row["collateral_given_usable"],
            row["new_shacl_given_usable"], row["new_grounding_given_usable"], row["output_failures"],
        ]
        lines.append(" & ".join(cells) + " " + "\\" * 2)
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\par\small $^\dagger$Denominator includes usable model outputs only. The other outcomes use all ten paired cases.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_latex(rows, path):
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Paired comparison & Both & Left only & Right only & Neither & Net " + "\\" * 2,
        r"\hline",
    ]
    for row in rows:
        label = row["left"] + " vs. " + row["right"]
        values = [label] + [
            str(row[field])
            for field in ("both_resolved", "left_only", "right_only", "neither_resolved", "net_target_difference")
        ]
        lines.append(" & ".join(values) + " " + "\\" * 2)
    lines.extend([r"\hline", r"\end{tabular}", r"\par\small Each comparison reuses the same ten controlled cases."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draft_results_notes(payload):
    counts = {key: payload["by_condition"][key]["counts"] for key in CONDITIONS}
    compare_verdict = payload["paired_target_comparisons"]["explanation_vs_verdict"]
    compare_location = payload["paired_target_comparisons"]["explanation_vs_location"]
    residual_cases = payload["residual_owl_case_ids"]
    failure_types = payload["pooled_case_condition_observations"]["counts"]["output_failure_types"]

    def value(condition, field, denominator="n"):
        item = counts[condition]
        return fraction(item[field], item[denominator])

    return "\n".join(
        [
            "# RQ3: OWL feedback framing",
            "",
            "Ten controlled disjointness cases were evaluated under three paired OWL feedback conditions, with one independent repair generation per condition. SHACL and grounding findings were measured after repair but were not included in the repair prompt.",
            "",
            "## Target removal and output usability",
            "",
            "The controlled target was removed in "
            + value("verdict", "controlled_target_removed")
            + " verdict only cases, "
            + value("location", "controlled_target_removed")
            + " location cases, and "
            + value("explanation", "controlled_target_removed")
            + " explanation cases. Compared with verdict only feedback, the explanation resolved "
            + str(compare_verdict["left_only"])
            + " additional paired case, with "
            + str(compare_verdict["right_only"])
            + " cases favoring verdict only feedback. Compared with location feedback, "
            + str(compare_location["left_only"])
            + " cases favored the explanation and "
            + str(compare_location["right_only"])
            + " favored location feedback. These are descriptive paired differences, not population estimates or claims of statistical dominance.",
            "",
            "Unusable model outputs occurred in "
            + value("verdict", "output_failures")
            + ", "
            + value("location", "output_failures")
            + ", and "
            + value("explanation", "output_failures")
            + " cases, respectively. Recorded parser outcomes were retained without retry or rescoring ("
            + ", ".join(f"{key}: {number}" for key, number in sorted(failure_types.items()))
            + "). An unusable output is not an OWL inconsistent graph.",
            "",
            "## Logical consistency and reference recovery",
            "",
            "Among usable outputs only, OWL consistency was observed in "
            + value("verdict", "owl_consistent", "usable_outputs")
            + ", "
            + value("location", "owl_consistent", "usable_outputs")
            + ", and "
            + value("explanation", "owl_consistent", "usable_outputs")
            + " cases under verdict, location, and explanation feedback, respectively. Exact recovery of the clean controlled reference graph occurred in "
            + value("verdict", "reference_recovery")
            + ", "
            + value("location", "reference_recovery")
            + ", and "
            + value("explanation", "reference_recovery")
            + " cases. Exact reference recovery does not establish recovery of every fact expressed by the source text.",
            "",
            "A removed controlled target did not guarantee OWL consistency. Residual inconsistency after target removal was observed in the same "
            + str(len(residual_cases))
            + " paired Music cases ("
            + ", ".join(residual_cases)
            + "). Their repetition across feedback conditions does not create additional independent cases.",
            "",
            "## Collateral edits and additional validation findings",
            "",
            "Among usable outputs, collateral edits occurred in "
            + value("verdict", "collateral_edit", "usable_outputs")
            + ", "
            + value("location", "collateral_edit", "usable_outputs")
            + ", and "
            + value("explanation", "collateral_edit", "usable_outputs")
            + " cases. New raw SHACL findings occurred in "
            + value("verdict", "new_raw_shacl_findings", "usable_outputs")
            + ", "
            + value("location", "new_raw_shacl_findings", "usable_outputs")
            + ", and "
            + value("explanation", "new_raw_shacl_findings", "usable_outputs")
            + " cases, while new grounding findings occurred in "
            + value("verdict", "new_grounding_findings", "usable_outputs")
            + ", "
            + value("location", "new_grounding_findings", "usable_outputs")
            + ", and "
            + value("explanation", "new_grounding_findings", "usable_outputs")
            + " cases. Grounding findings reflect judgments from the locked assessor rather than human review of each assertion.",
            "",
            "The explanation was derived from the controlled construction. It was not a newly generated reasoner explanation. The 30 recorded rows are paired observations of a case and a condition over ten controlled cases, not 30 independent experimental units.",
            "",
            "> More informative OWL feedback modestly improved target removal on the first repair, but it did not prevent extensive collateral edits, additional grounding findings, or persistent inconsistencies after removal of the controlled target.",
            "",
        ]
    )


def matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to make the figures. Install it with: pip install matplotlib"
        ) from exc
    return plt


def save_figure(figure, stem):
    paths = []
    for suffix in ("png", "pdf"):
        path = stem.with_suffix("." + suffix)
        figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(path)
    return paths


def plot_outcomes(payload, output_dir):
    plt = matplotlib_pyplot()
    figure, axis = plt.subplots(figsize=(9.1, 4.5))
    metrics = (
        ("controlled_target_removed", "Target removed", "#137c8b"),
        ("owl_consistent", "OWL consistent", "#476c9b"),
        ("reference_recovery", "Exact reference", "#8e6c88"),
        ("output_failures", "Output failure", "#d48855"),
    )
    width = 0.18
    for index, (field, label, color) in enumerate(metrics):
        positions = [point + (index - 1.5) * width for point in range(len(CONDITIONS))]
        values = [payload["by_condition"][condition]["counts"][field] for condition in CONDITIONS]
        bars = axis.bar(positions, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.13, str(value), ha="center", fontsize=9)
    axis.set_xticks(range(len(CONDITIONS)), [LABELS[key] for key in CONDITIONS])
    axis.set_ylabel("Cases, out of the same 10 paired cases")
    axis.set_ylim(0, 11)
    axis.set_title("Outcomes of the first repair by OWL feedback framing")
    axis.legend(frameon=False, ncol=2, loc="upper center")
    axis.text(
        0.01, -0.18,
        "OWL consistency is observable only for usable outputs (verdict n=8, location n=9, explanation n=9).",
        transform=axis.transAxes, fontsize=8, color="#46505a",
    )
    axis.spines[["top", "right"]].set_visible(False)
    paths = save_figure(figure, output_dir / "framing_outcomes")
    plt.close(figure)
    return paths


def plot_side_effects(payload, output_dir):
    plt = matplotlib_pyplot()
    figure, axis = plt.subplots(figsize=(8.6, 4.3))
    metrics = (
        ("collateral_edit", "Collateral edits", "#c56c5b"),
        ("new_raw_shacl_findings", "New raw SHACL findings", "#80669d"),
        ("new_grounding_findings", "New grounding findings", "#d7a046"),
    )
    width = 0.23
    for index, (field, label, color) in enumerate(metrics):
        positions = [point + (index - 1) * width for point in range(len(CONDITIONS))]
        rates = [payload["by_condition"][condition]["rates"][field] for condition in CONDITIONS]
        bars = axis.bar(positions, [100 * item["estimate"] for item in rates], width=width, label=label, color=color)
        for bar, item in zip(bars, rates):
            axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                fraction(item["numerator"], item["denominator"]), ha="center", fontsize=8,
            )
    axis.set_xticks(range(len(CONDITIONS)), [LABELS[key] for key in CONDITIONS])
    axis.set_ylim(0, 135)
    axis.set_ylabel("Percentage of usable outputs")
    axis.set_title("Collateral changes and new findings among usable repairs")
    axis.legend(frameon=False, ncol=3, loc="upper center", fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    paths = save_figure(figure, output_dir / "framing_side_effects")
    plt.close(figure)
    return paths


def short_case_label(case_id):
    parts = case_id.split("_")
    return f"{parts[2].title()} {parts[-1]}"


def plot_paired_targets(payload, output_dir):
    plt = matplotlib_pyplot()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colors = ["#2b8a78", "#80669d", "#d9a441", "#c45850"]
    descriptions = [
        "Target removed, OWL consistent",
        "Target removed, still OWL inconsistent",
        "Target retained in usable output",
        "Unusable model output",
    ]
    matrix = []
    for case in payload["cases"]:
        statuses = []
        for condition in CONDITIONS:
            outcome = case["conditions"][condition]
            if not outcome["usable_output"]:
                statuses.append(3)
            elif not outcome["controlled_target_removed"]:
                statuses.append(2)
            elif not outcome["owl_consistent"]:
                statuses.append(1)
            else:
                statuses.append(0)
        matrix.append(statuses)

    figure, axis = plt.subplots(figsize=(8.6, 5.9))
    axis.imshow(matrix, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
    axis.set_xticks(range(len(CONDITIONS)), [LABELS[key] for key in CONDITIONS])
    axis.set_yticks(range(len(payload["cases"])), [short_case_label(case["id"]) for case in payload["cases"]])
    axis.set_title("The same ten controlled cases under all three feedback conditions")
    axis.set_xticks([index + 0.5 for index in range(len(CONDITIONS) - 1)], minor=True)
    axis.set_yticks([index + 0.5 for index in range(len(payload["cases"]) - 1)], minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.axhline(4.5, color="#263238", linewidth=2)
    legend = [Patch(facecolor=color, label=label) for color, label in zip(colors, descriptions)]
    axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False, ncol=2, fontsize=8)
    paths = save_figure(figure, output_dir / "framing_paired_targets")
    plt.close(figure)
    return paths


def run(analysis_path=DEFAULT_ANALYSIS, output_dir=DEFAULT_OUTPUT_DIR, skip_figures=False):
    analysis_path, output_dir = map(Path, (analysis_path, output_dir))
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    validate_analysis(payload)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions, comparisons, domains = condition_rows(payload), paired_rows(payload), domain_rows(payload)
    outputs = [
        output_dir / "condition_summary.csv",
        output_dir / "paired_comparisons.csv",
        output_dir / "domain_summary.csv",
        output_dir / "condition_summary.tex",
        output_dir / "paired_comparisons.tex",
        output_dir / "rq3_results_notes.md",
    ]
    write_csv(conditions, outputs[0])
    write_csv(comparisons, outputs[1])
    write_csv(domains, outputs[2])
    write_condition_latex(conditions, outputs[3])
    write_paired_latex(comparisons, outputs[4])
    outputs[5].write_text(draft_results_notes(payload), encoding="utf-8")

    if not skip_figures:
        outputs.extend(plot_outcomes(payload, output_dir))
        outputs.extend(plot_side_effects(payload, output_dir))
        outputs.extend(plot_paired_targets(payload, output_dir))

    manifest_path = output_dir / "owl_feedback_reporting_manifest.json"
    manifest = {
        "analysis_sha256": sha256_file(analysis_path),
        "analysis_git_head": payload["analysis_provenance"]["analysis_git_head"],
        "run_git_head": payload["input"]["run_git_head"],
        "result_sha256": payload["input"]["result_sha256"],
        "reporting_script_sha256": sha256_file(__file__),
        "reporting_spec_sha256": sha256_file(REPORTING_SPEC),
        "figures_skipped": skip_figures,
        "artifacts": {path.name: sha256_file(path) for path in outputs},
        "models_executed": False,
        "validators_executed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {output_dir}")
    print(f"artifacts: {len(outputs) + 1}")
    print("No language model, grounding assessor, or validator was executed.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    run(args.analysis, args.output_dir, args.skip_figures)


if __name__ == "__main__":
    main()
