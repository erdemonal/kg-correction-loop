import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "controlled" / "manifest.jsonl"
RESULTS = ROOT / "results" / "controlled_grounding_validation.jsonl"
OUTPUT = ROOT / "results" / "controlled_grounding_target_analysis.json"

EXPECTED_TARGET = {
    "disjointness": True,
    "domain_range": False,
    "cardinality": False,
    "temporal": True,
    "grounding": True,
}


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


def index_unique(rows, path):
    output = {}

    for row in rows:
        case_id = row.get("id")

        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"{path}: missing case id")

        if case_id in output:
            raise RuntimeError(f"{path}: duplicate id {case_id}")

        output[case_id] = row

    return output


def load_payload(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    triples = payload.get("triples")

    if not isinstance(triples, list):
        raise RuntimeError(f"{path}: missing triples list")

    normalized = []

    for triple in triples:
        if (
            not isinstance(triple, list)
            or len(triple) != 3
            or not all(isinstance(value, str) for value in triple)
        ):
            raise RuntimeError(f"{path}: invalid triple {triple!r}")

        normalized.append(tuple(triple))

    return payload, normalized


def judgment_map(state_result):
    judgments = state_result.get("judgments")

    if not isinstance(judgments, list):
        raise RuntimeError("Result state has no judgments list")

    output = {}

    for judgment in judgments:
        triple = judgment.get("triple")
        verdict = judgment.get("verdict")

        if (
            not isinstance(triple, list)
            or len(triple) != 3
            or verdict not in {"SUPPORTED", "UNSUPPORTED"}
        ):
            raise RuntimeError(f"Invalid judgment: {judgment!r}")

        key = tuple(triple)

        if key in output:
            raise RuntimeError(f"Duplicate judgment triple: {triple!r}")

        output[key] = verdict

    return output


def delta(clean_triples, injected_triples):
    clean = set(clean_triples)
    injected = set(injected_triples)

    return {
        "added": sorted(injected - clean),
        "removed": sorted(clean - injected),
        "unchanged": sorted(clean & injected),
    }


def validate_delta(condition, case_delta):
    added = case_delta["added"]
    removed = case_delta["removed"]

    if condition in {"disjointness", "domain_range", "grounding"}:
        if len(added) != 1 or removed:
            raise RuntimeError(
                f"{condition}: expected one added triple and no removals, "
                f"found added={len(added)} removed={len(removed)}"
            )
        return

    if condition == "cardinality":
        if added or len(removed) != 1:
            raise RuntimeError(
                "cardinality: expected one removed triple and no additions, "
                f"found added={len(added)} removed={len(removed)}"
            )
        return

    if condition == "temporal":
        if len(added) != 2 or len(removed) != 2:
            raise RuntimeError(
                "temporal: expected two added and two removed triples, "
                f"found added={len(added)} removed={len(removed)}"
            )
        return

    raise RuntimeError(f"Unsupported condition: {condition}")


def target_detection(condition, case_delta, clean_map, injected_map):
    if condition == "cardinality":
        removed = case_delta["removed"]
        verdicts = [clean_map[triple] for triple in removed]

        return {
            "expected_grounding_error": False,
            "observed_grounding_error": False,
            "matches_expected": True,
            "target_asserted_after_modification": False,
            "target_triples": [],
            "target_verdicts": [],
            "removed_clean_triples": [list(triple) for triple in removed],
            "removed_clean_verdicts": verdicts,
            "removed_clean_supported": all(
                verdict == "SUPPORTED" for verdict in verdicts
            ),
        }

    added = case_delta["added"]
    verdicts = [injected_map[triple] for triple in added]
    observed = any(verdict == "UNSUPPORTED" for verdict in verdicts)
    expected = EXPECTED_TARGET[condition]

    return {
        "expected_grounding_error": expected,
        "observed_grounding_error": observed,
        "matches_expected": observed == expected,
        "target_asserted_after_modification": True,
        "target_triples": [list(triple) for triple in added],
        "target_verdicts": verdicts,
        "removed_clean_triples": [
            list(triple) for triple in case_delta["removed"]
        ],
        "removed_clean_verdicts": [
            clean_map[triple] for triple in case_delta["removed"]
        ],
        "removed_clean_supported": all(
            clean_map[triple] == "SUPPORTED"
            for triple in case_delta["removed"]
        ),
    }


def background_diagnostics(
    condition,
    case_delta,
    clean_map,
    injected_map,
):
    target_clean = set(case_delta["removed"])
    target_injected = set(case_delta["added"])

    clean_unsupported_all = sorted(
        triple
        for triple, verdict in clean_map.items()
        if verdict == "UNSUPPORTED"
    )
    clean_unsupported_background = sorted(
        triple
        for triple, verdict in clean_map.items()
        if verdict == "UNSUPPORTED" and triple not in target_clean
    )
    injected_unsupported_all = sorted(
        triple
        for triple, verdict in injected_map.items()
        if verdict == "UNSUPPORTED"
    )
    injected_unsupported_background = sorted(
        triple
        for triple, verdict in injected_map.items()
        if verdict == "UNSUPPORTED" and triple not in target_injected
    )

    return {
        "clean_any_unsupported": bool(clean_unsupported_all),
        "clean_unsupported_all": [
            list(triple) for triple in clean_unsupported_all
        ],
        "clean_background_unsupported": [
            list(triple) for triple in clean_unsupported_background
        ],
        "injected_any_unsupported": bool(injected_unsupported_all),
        "injected_unsupported_all": [
            list(triple) for triple in injected_unsupported_all
        ],
        "injected_background_unsupported": [
            list(triple) for triple in injected_unsupported_background
        ],
    }


def summarize(rows):
    by_condition = {}

    for condition in EXPECTED_TARGET:
        selected = [
            row for row in rows if row["condition"] == condition
        ]

        by_condition[condition] = {
            "n": len(selected),
            "expected_detected": sum(
                row["target"]["expected_grounding_error"]
                for row in selected
            ),
            "observed_detected": sum(
                row["target"]["observed_grounding_error"]
                for row in selected
            ),
            "matching_expected": sum(
                row["target"]["matches_expected"]
                for row in selected
            ),
            "clean_cases_with_any_unsupported": sum(
                row["background"]["clean_any_unsupported"]
                for row in selected
            ),
            "clean_cases_with_background_unsupported": sum(
                bool(
                    row["background"][
                        "clean_background_unsupported"
                    ]
                )
                for row in selected
            ),
            "removed_target_clean_false_flags": sum(
                (
                    bool(row["target"]["removed_clean_triples"])
                    and not row["target"]["removed_clean_supported"]
                )
                for row in selected
            ),
        }

    target_matches = sum(
        row["target"]["matches_expected"] for row in rows
    )

    return {
        "cases": len(rows),
        "target_matching_expected": target_matches,
        "target_mismatches": len(rows) - target_matches,
        "clean_cases_with_any_unsupported": sum(
            row["background"]["clean_any_unsupported"]
            for row in rows
        ),
        "clean_cases_with_background_unsupported": sum(
            bool(row["background"]["clean_background_unsupported"])
            for row in rows
        ),
        "by_condition": by_condition,
    }


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")

    if not RESULTS.exists():
        raise SystemExit(f"Missing grounding results: {RESULTS}")

    manifest = index_unique(read_jsonl(MANIFEST), MANIFEST)
    results = index_unique(read_jsonl(RESULTS), RESULTS)

    if set(manifest) != set(results):
        raise RuntimeError(
            "Manifest and grounding results contain different case ids"
        )

    rows = []

    for case_id in manifest:
        manifest_row = manifest[case_id]
        result_row = results[case_id]
        condition = manifest_row["condition"]

        clean_path = ROOT / manifest_row["files"]["grounding_clean"]
        injected_path = (
            ROOT / manifest_row["files"]["grounding_injected"]
        )

        _, clean_triples = load_payload(clean_path)
        _, injected_triples = load_payload(injected_path)

        case_delta = delta(clean_triples, injected_triples)
        validate_delta(condition, case_delta)

        clean_map = judgment_map(result_row["clean"])
        injected_map = judgment_map(result_row["injected"])

        if set(clean_map) != set(clean_triples):
            raise RuntimeError(
                f"{case_id}: clean judgments do not match payload"
            )

        if set(injected_map) != set(injected_triples):
            raise RuntimeError(
                f"{case_id}: injected judgments do not match payload"
            )

        target = target_detection(
            condition,
            case_delta,
            clean_map,
            injected_map,
        )

        background = background_diagnostics(
            condition,
            case_delta,
            clean_map,
            injected_map,
        )

        rows.append(
            {
                "id": case_id,
                "domain": manifest_row["domain"],
                "condition": condition,
                "delta": {
                    "added": [
                        list(triple)
                        for triple in case_delta["added"]
                    ],
                    "removed": [
                        list(triple)
                        for triple in case_delta["removed"]
                    ],
                },
                "target": target,
                "background": background,
            }
        )

    summary = summarize(rows)

    payload = {
        "analysis": (
            "Primary-modification target analysis of the frozen v3 "
            "grounding run. Whole-graph unsupported assertions are "
            "reported separately as background diagnostics."
        ),
        "source_results": str(RESULTS.relative_to(ROOT)),
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "target_detection_rule": {
            "addition": (
                "A grounding detection is attributed to the controlled "
                "modification when an added target assertion is judged "
                "UNSUPPORTED."
            ),
            "temporal_swap": (
                "A grounding detection is attributed when at least one "
                "new swapped date assertion is judged UNSUPPORTED."
            ),
            "cardinality_removal": (
                "Deletion creates no asserted statement for the grounding "
                "judge, so the controlled cardinality omission is not "
                "grounding-detectable. The removed clean assertion's "
                "verdict is retained only as a diagnostic."
            ),
        },
        "expected_target_grounding": EXPECTED_TARGET,
        "summary": summary,
        "cases": rows,
    }

    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        "target matching expected: "
        f"{summary['target_matching_expected']}/"
        f"{summary['cases']}"
    )
    print(
        "target mismatches: "
        f"{summary['target_mismatches']}"
    )
    print(
        "clean cases with any unsupported assertion: "
        f"{summary['clean_cases_with_any_unsupported']}/"
        f"{summary['cases']}"
    )
    print(
        "clean cases with background unsupported assertion: "
        f"{summary['clean_cases_with_background_unsupported']}/"
        f"{summary['cases']}"
    )

    print("by condition")

    for condition in (
        "disjointness",
        "domain_range",
        "cardinality",
        "temporal",
        "grounding",
    ):
        row = summary["by_condition"][condition]
        print(
            f"  {condition}: "
            f"target match {row['matching_expected']}/{row['n']}, "
            f"observed detected {row['observed_detected']}/{row['n']}, "
            "clean any unsupported "
            f"{row['clean_cases_with_any_unsupported']}/{row['n']}, "
            "removed-target clean false flags "
            f"{row['removed_target_clean_false_flags']}/{row['n']}"
        )

    mismatches = [
        row for row in rows
        if not row["target"]["matches_expected"]
    ]

    print(f"target mismatch ids: {len(mismatches)}")

    for row in mismatches:
        print(
            f"  {row['id']} / {row['condition']}: "
            f"expected "
            f"{row['target']['expected_grounding_error']}, "
            f"observed "
            f"{row['target']['observed_grounding_error']}, "
            f"target verdicts={row['target']['target_verdicts']}"
        )

    print(f"wrote: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
