import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SYMBOLIC_RESULTS = (
    ROOT / "results" / "controlled_symbolic_validation.jsonl"
)
GROUNDING_ANALYSIS = (
    ROOT / "results" / "controlled_grounding_target_analysis.json"
)
ADJUDICATION = (
    ROOT / "experiments" / "controlled_grounding_adjudication.json"
)
CALIBRATION_SPLIT = (
    ROOT / "experiments" / "grounding_calibration_split.json"
)

OUTPUT_JSON = ROOT / "results" / "validation_coverage_analysis.json"
OUTPUT_CSV = ROOT / "results" / "validation_coverage_cases.csv"


CONDITIONS = (
    "disjointness",
    "domain_range",
    "cardinality",
    "temporal",
    "grounding",
)


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


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_unique(rows, source):
    output = {}

    for row in rows:
        case_id = row.get("id")

        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"{source}: missing case id")

        if case_id in output:
            raise RuntimeError(f"{source}: duplicate id {case_id}")

        output[case_id] = row

    return output


def symbolic_detection(symbolic_row, validator):
    observed = symbolic_row.get("observed", {})
    clean = observed.get("clean", {})
    injected = observed.get("injected", {})

    if validator not in clean or validator not in injected:
        raise RuntimeError(
            f"{symbolic_row.get('id')}: missing symbolic result "
            f"for {validator}"
        )

    clean_value = clean[validator]
    injected_value = injected[validator]

    if not isinstance(clean_value, bool) or not isinstance(
        injected_value, bool
    ):
        raise RuntimeError(
            f"{symbolic_row.get('id')}: invalid symbolic values "
            f"for {validator}"
        )

    return {
        "clean": clean_value,
        "injected": injected_value,
        "detected": clean_value and not injected_value,
    }


def grounding_outcome(expected, observed):
    if expected and observed:
        return "true_positive"

    if not expected and observed:
        return "false_positive"

    if expected and not observed:
        return "false_negative"

    return "true_negative"


def build_split_index(payload):
    output = {}

    for split_name in ("calibration", "heldout"):
        rows = payload.get(split_name)

        if not isinstance(rows, list):
            raise RuntimeError(
                f"grounding split has no {split_name} list"
            )

        for row in rows:
            case_id = row.get("id")

            if not isinstance(case_id, str) or not case_id:
                raise RuntimeError(
                    f"{split_name}: missing case id"
                )

            if case_id in output:
                raise RuntimeError(
                    f"case appears in both grounding splits: {case_id}"
                )

            output[case_id] = {
                "split": split_name,
                "human_grounding_error": row.get(
                    "human_grounding_error"
                ),
            }

    return output


def main_pattern(row):
    names = []

    if row["raw_shacl_detected"]:
        names.append("shacl")

    if row["owl_inconsistency_detected"]:
        names.append("owl")

    if row["grounding_detected"]:
        names.append("grounding")

    return "+".join(names) if names else "none"


def summarize_rows(rows):
    by_condition = {}

    for condition in CONDITIONS:
        selected = [
            row for row in rows
            if row["condition"] == condition
        ]

        outcomes = Counter(
            row["grounding_outcome"] for row in selected
        )

        by_condition[condition] = {
            "n": len(selected),
            "raw_shacl_detected": sum(
                row["raw_shacl_detected"]
                for row in selected
            ),
            "owlrl_shacl_detected": sum(
                row["owlrl_shacl_detected"]
                for row in selected
            ),
            "owl_inconsistency_detected": sum(
                row["owl_inconsistency_detected"]
                for row in selected
            ),
            "grounding_detected": sum(
                row["grounding_detected"]
                for row in selected
            ),
            "grounding_expected_error": sum(
                row["grounding_expected_error"]
                for row in selected
            ),
            "grounding_matches_expected": sum(
                row["grounding_matches_expected"]
                for row in selected
            ),
            "grounding_true_positive": outcomes["true_positive"],
            "grounding_false_positive": outcomes["false_positive"],
            "grounding_true_negative": outcomes["true_negative"],
            "grounding_false_negative": outcomes["false_negative"],
        }

    outcomes = Counter(
        row["grounding_outcome"] for row in rows
    )
    overlap = Counter(
        row["observed_main_pattern"] for row in rows
    )

    return {
        "n": len(rows),
        "raw_shacl_detected": sum(
            row["raw_shacl_detected"] for row in rows
        ),
        "owlrl_shacl_detected": sum(
            row["owlrl_shacl_detected"] for row in rows
        ),
        "owl_inconsistency_detected": sum(
            row["owl_inconsistency_detected"] for row in rows
        ),
        "grounding_detected": sum(
            row["grounding_detected"] for row in rows
        ),
        "grounding_expected_error": sum(
            row["grounding_expected_error"] for row in rows
        ),
        "grounding_matches_expected": sum(
            row["grounding_matches_expected"] for row in rows
        ),
        "grounding_outcomes": {
            "true_positive": outcomes["true_positive"],
            "false_positive": outcomes["false_positive"],
            "true_negative": outcomes["true_negative"],
            "false_negative": outcomes["false_negative"],
        },
        "observed_overlap_main": dict(
            sorted(overlap.items())
        ),
        "by_condition": by_condition,
    }


def build_rows(
    symbolic_rows,
    grounding_payload,
    adjudication_payload,
    split_payload,
):
    symbolic = index_unique(
        symbolic_rows,
        SYMBOLIC_RESULTS,
    )

    grounding_cases = grounding_payload.get("cases")

    if not isinstance(grounding_cases, list):
        raise RuntimeError(
            "grounding target analysis has no cases list"
        )

    grounding = index_unique(
        grounding_cases,
        GROUNDING_ANALYSIS,
    )

    if set(symbolic) != set(grounding):
        raise RuntimeError(
            "symbolic and grounding analyses contain different case ids"
        )

    if len(symbolic) != 50:
        raise RuntimeError(
            f"expected 50 controlled cases, found {len(symbolic)}"
        )

    adjudication_cases = adjudication_payload.get("cases")

    if not isinstance(adjudication_cases, list):
        raise RuntimeError(
            "grounding adjudication has no cases list"
        )

    adjudication = index_unique(
        adjudication_cases,
        ADJUDICATION,
    )
    split_index = build_split_index(split_payload)

    rows = []

    for case_id in symbolic:
        symbolic_row = symbolic[case_id]
        grounding_row = grounding[case_id]

        if symbolic_row.get("domain") != grounding_row.get("domain"):
            raise RuntimeError(
                f"{case_id}: domain differs across result files"
            )

        if (
            symbolic_row.get("condition")
            != grounding_row.get("condition")
        ):
            raise RuntimeError(
                f"{case_id}: condition differs across result files"
            )

        target = grounding_row.get("target")

        if not isinstance(target, dict):
            raise RuntimeError(
                f"{case_id}: missing target grounding analysis"
            )

        expected = target.get("expected_grounding_error")
        observed = target.get("observed_grounding_error")
        matches = target.get("matches_expected")

        if not all(
            isinstance(value, bool)
            for value in (expected, observed, matches)
        ):
            raise RuntimeError(
                f"{case_id}: invalid grounding target values"
            )

        if matches != (expected == observed):
            raise RuntimeError(
                f"{case_id}: inconsistent grounding match flag"
            )

        raw = symbolic_detection(
            symbolic_row,
            "raw_shacl",
        )
        owlrl = symbolic_detection(
            symbolic_row,
            "owlrl_shacl",
        )
        owl = symbolic_detection(
            symbolic_row,
            "owl_consistent",
        )

        split = split_index.get(case_id)
        review = adjudication.get(case_id)

        outcome = grounding_outcome(expected, observed)

        if review is not None:
            expected_review = {
                "false_positive": "false_positive",
                "false_negative": "false_negative",
            }.get(review.get("adjudication"))

            if expected_review is None:
                raise RuntimeError(
                    f"{case_id}: unexpected adjudication value"
                )

            if outcome != expected_review:
                raise RuntimeError(
                    f"{case_id}: adjudication disagrees with "
                    f"frozen target result"
                )
        elif outcome in {"false_positive", "false_negative"}:
            raise RuntimeError(
                f"{case_id}: grounding mismatch has no adjudication"
            )

        row = {
            "id": case_id,
            "domain": symbolic_row["domain"],
            "condition": symbolic_row["condition"],
            "raw_shacl_clean": raw["clean"],
            "raw_shacl_injected": raw["injected"],
            "raw_shacl_detected": raw["detected"],
            "owlrl_shacl_clean": owlrl["clean"],
            "owlrl_shacl_injected": owlrl["injected"],
            "owlrl_shacl_detected": owlrl["detected"],
            "owl_clean_consistent": owl["clean"],
            "owl_injected_consistent": owl["injected"],
            "owl_inconsistency_detected": owl["detected"],
            "grounding_expected_error": expected,
            "grounding_detected": observed,
            "grounding_matches_expected": matches,
            "grounding_outcome": outcome,
            "grounding_adjudication": (
                review.get("adjudication")
                if review is not None
                else ""
            ),
            "pilot_split": (
                split["split"] if split is not None else "none"
            ),
            "pilot_human_grounding_error": (
                split["human_grounding_error"]
                if split is not None
                else None
            ),
        }
        row["observed_main_pattern"] = main_pattern(row)
        rows.append(row)

    return rows


def sensitivity(rows, predicate):
    selected = [row for row in rows if predicate(row)]
    return summarize_rows(selected)


def write_csv(rows):
    fieldnames = [
        "id",
        "domain",
        "condition",
        "raw_shacl_detected",
        "owlrl_shacl_detected",
        "owl_inconsistency_detected",
        "grounding_expected_error",
        "grounding_detected",
        "grounding_matches_expected",
        "grounding_outcome",
        "grounding_adjudication",
        "observed_main_pattern",
        "pilot_split",
        "pilot_human_grounding_error",
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
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
        writer.writerows(rows)


def print_condition_table(summary):
    print("condition coverage")
    print(
        "  condition          n   SHACL  SHACL+RL  OWL  "
        "grounding  grounding match"
    )

    for condition in CONDITIONS:
        row = summary["by_condition"][condition]
        print(
            f"  {condition:<17} "
            f"{row['n']:>2} "
            f"{row['raw_shacl_detected']:>7} "
            f"{row['owlrl_shacl_detected']:>9} "
            f"{row['owl_inconsistency_detected']:>4} "
            f"{row['grounding_detected']:>10} "
            f"{row['grounding_matches_expected']:>15}"
        )


def main():
    required = [
        SYMBOLIC_RESULTS,
        GROUNDING_ANALYSIS,
        ADJUDICATION,
        CALIBRATION_SPLIT,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        names = "\n  ".join(
            str(path.relative_to(ROOT))
            for path in missing
        )
        raise SystemExit(
            "Missing frozen validation inputs:\n  " + names
        )

    rows = build_rows(
        read_jsonl(SYMBOLIC_RESULTS),
        read_json(GROUNDING_ANALYSIS),
        read_json(ADJUDICATION),
        read_json(CALIBRATION_SPLIT),
    )

    overall = summarize_rows(rows)

    calibration_overlap = [
        row["id"] for row in rows
        if row["pilot_split"] == "calibration"
    ]
    heldout_overlap = [
        row["id"] for row in rows
        if row["pilot_split"] == "heldout"
    ]

    payload = {
        "analysis": (
            "Coverage and overlap analysis for the frozen controlled "
            "validation results. SHACL and OWL detection are defined by "
            "a clean passing state followed by an injected failing state. "
            "Grounding uses the frozen primary modification result."
        ),
        "main_validators": {
            "shacl": "raw SHACL condition",
            "owl": "OWL consistency with HermiT",
            "grounding": "frozen v3 grounding assessor",
        },
        "supplementary_condition": (
            "SHACL with pySHACL OWL RL inference enabled"
        ),
        "grounding_note": (
            "Grounding firings are reported as observed model behavior. "
            "The expected source support status and the six reviewed "
            "mismatches are kept separately, so false grounding signals "
            "are not treated as valid coverage."
        ),
        "overall": overall,
        "pilot_overlap": {
            "calibration": {
                "n": len(calibration_overlap),
                "ids": calibration_overlap,
            },
            "heldout": {
                "n": len(heldout_overlap),
                "ids": heldout_overlap,
            },
            "any_pilot_split": {
                "n": (
                    len(calibration_overlap)
                    + len(heldout_overlap)
                ),
                "ids": (
                    calibration_overlap
                    + heldout_overlap
                ),
            },
        },
        "sensitivity": {
            "exclude_calibration_overlap": sensitivity(
                rows,
                lambda row: row["pilot_split"] != "calibration",
            ),
            "exclude_heldout_overlap": sensitivity(
                rows,
                lambda row: row["pilot_split"] != "heldout",
            ),
            "exclude_all_pilot_overlap": sensitivity(
                rows,
                lambda row: row["pilot_split"] == "none",
            ),
        },
        "cases": rows,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(rows)

    print(f"cases: {overall['n']}")
    print_condition_table(overall)
    print()
    print(
        "grounding outcomes: "
        f"TP={overall['grounding_outcomes']['true_positive']} "
        f"FP={overall['grounding_outcomes']['false_positive']} "
        f"TN={overall['grounding_outcomes']['true_negative']} "
        f"FN={overall['grounding_outcomes']['false_negative']}"
    )
    print(
        "grounding matches expected: "
        f"{overall['grounding_matches_expected']}/{overall['n']}"
    )
    print()
    print("observed overlap among main validators")

    for pattern, count in (
        overall["observed_overlap_main"].items()
    ):
        print(f"  {pattern}: {count}")

    print()
    print(
        "pilot overlap: "
        f"calibration={len(calibration_overlap)}, "
        f"heldout={len(heldout_overlap)}, "
        f"total={len(calibration_overlap) + len(heldout_overlap)}"
    )

    print(
        "  calibration: "
        + ", ".join(calibration_overlap)
    )
    print(
        "  heldout: "
        + ", ".join(heldout_overlap)
    )

    no_pilot = payload["sensitivity"][
        "exclude_all_pilot_overlap"
    ]
    print()
    print(
        "sensitivity excluding all pilot overlap: "
        f"{no_pilot['grounding_matches_expected']}/"
        f"{no_pilot['n']} grounding matches expected"
    )
    print(
        "  grounding outcomes: "
        f"TP={no_pilot['grounding_outcomes']['true_positive']} "
        f"FP={no_pilot['grounding_outcomes']['false_positive']} "
        f"TN={no_pilot['grounding_outcomes']['true_negative']} "
        f"FN={no_pilot['grounding_outcomes']['false_negative']}"
    )

    print()
    print(f"wrote: {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote: {OUTPUT_CSV.relative_to(ROOT)}")
    print("No validator or language model was executed.")


if __name__ == "__main__":
    main()
