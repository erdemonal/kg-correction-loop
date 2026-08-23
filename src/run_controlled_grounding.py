import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.grounding_judge import (
    JUDGE_VERSION,
    MODEL,
    judge_case,
    load_prompt,
    model_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "controlled" / "manifest.jsonl"
RESULT = ROOT / "results" / "controlled_grounding_validation.jsonl"
METADATA = RESULT.with_suffix(RESULT.suffix + ".meta.json")

EXPECTED_INJECTED = {
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


def load_payload(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}") from exc

    required = {"id", "domain", "source_text", "triples"}

    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError(
            f"{path}: grounding payload must contain exactly "
            f"{sorted(required)}"
        )

    if not isinstance(payload["id"], str) or not payload["id"]:
        raise RuntimeError(f"{path}: invalid id")

    if payload["domain"] not in {"movie", "music"}:
        raise RuntimeError(f"{path}: invalid domain")

    if (
        not isinstance(payload["source_text"], str)
        or not payload["source_text"].strip()
    ):
        raise RuntimeError(f"{path}: invalid source_text")

    triples = payload["triples"]

    if not isinstance(triples, list):
        raise RuntimeError(f"{path}: triples must be a list")

    for triple in triples:
        if (
            not isinstance(triple, list)
            or len(triple) != 3
            or not all(isinstance(value, str) for value in triple)
        ):
            raise RuntimeError(f"{path}: invalid triple {triple!r}")

    return payload


def validate_manifest(rows):
    if len(rows) != 50:
        raise RuntimeError(
            f"Expected 50 controlled cases, found {len(rows)}"
        )

    ids = [row.get("id") for row in rows]

    if len(set(ids)) != 50:
        raise RuntimeError("Controlled manifest contains duplicate ids")

    for row in rows:
        if row.get("domain") not in {"movie", "music"}:
            raise RuntimeError(f"{row.get('id')}: invalid domain")

        condition = row.get("condition")

        if condition not in EXPECTED_INJECTED:
            raise RuntimeError(
                f"{row.get('id')}: unsupported condition {condition!r}"
            )

        files = row.get("files")

        if not isinstance(files, dict):
            raise RuntimeError(f"{row.get('id')}: missing files mapping")

        for key in ("grounding_clean", "grounding_injected"):
            value = files.get(key)

            if not isinstance(value, str) or not value:
                raise RuntimeError(
                    f"{row.get('id')}: missing {key}"
                )


def expected_grounding(condition, state):
    if state == "clean":
        return False

    if state == "injected":
        return EXPECTED_INJECTED[condition]

    raise ValueError(f"Unsupported state: {state}")


def payload_for_manifest(row, state):
    key = (
        "grounding_clean"
        if state == "clean"
        else "grounding_injected"
    )
    path = ROOT / row["files"][key]
    payload = load_payload(path)

    if payload["id"] != row["id"]:
        raise RuntimeError(
            f"{row['id']}: {state} payload id mismatch"
        )

    if payload["domain"] != row["domain"]:
        raise RuntimeError(
            f"{row['id']}: {state} payload domain mismatch"
        )

    if payload["source_text"] != row["source_text"]:
        raise RuntimeError(
            f"{row['id']}: {state} source text does not match manifest"
        )

    return payload


def summarize(rows):
    state_summary = {
        "clean": {
            "n": 0,
            "expected_grounding_error": 0,
            "observed_grounding_error": 0,
            "matching_expected": 0,
        },
        "injected": {
            "n": 0,
            "expected_grounding_error": 0,
            "observed_grounding_error": 0,
            "matching_expected": 0,
        },
    }

    by_condition = {}

    for row in rows:
        condition = row["condition"]
        bucket = by_condition.setdefault(
            condition,
            {
                "n": 0,
                "clean_matches": 0,
                "injected_matches": 0,
                "injected_expected_grounding_error": 0,
                "injected_observed_grounding_error": 0,
            },
        )
        bucket["n"] += 1

        for state in ("clean", "injected"):
            observed = row[state]["grounding_error"]
            expected = row[state]["expected_grounding_error"]
            summary = state_summary[state]

            summary["n"] += 1
            summary["expected_grounding_error"] += int(expected)
            summary["observed_grounding_error"] += int(observed)
            summary["matching_expected"] += int(observed == expected)

        bucket["clean_matches"] += int(
            row["clean"]["grounding_error"]
            == row["clean"]["expected_grounding_error"]
        )
        bucket["injected_matches"] += int(
            row["injected"]["grounding_error"]
            == row["injected"]["expected_grounding_error"]
        )
        bucket["injected_expected_grounding_error"] += int(
            row["injected"]["expected_grounding_error"]
        )
        bucket["injected_observed_grounding_error"] += int(
            row["injected"]["grounding_error"]
        )

    return state_summary, by_condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(
            "Controlled manifest not found. Run "
            "`python -m src.build_controlled_dataset --overwrite` first."
        )

    if (
        (RESULT.exists() or METADATA.exists())
        and not args.overwrite
    ):
        raise SystemExit(
            "Controlled grounding result already exists. "
            "Use --overwrite only if you intentionally want to replace it."
        )

    manifest = read_jsonl(MANIFEST)
    validate_manifest(manifest)

    metadata = model_metadata()

    if JUDGE_VERSION != "v3":
        raise RuntimeError(
            f"Controlled grounding requires frozen judge v3, got "
            f"{JUDGE_VERSION!r}"
        )

    metadata.update(
        {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "judge_version": JUDGE_VERSION,
            "model": MODEL,
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "cases": len(manifest),
            "states_per_case": 2,
            "evaluation_units": len(manifest) * 2,
            "decision_unit": "asserted content triple",
            "case_aggregation": (
                "grounding_error if any asserted content triple "
                "is unsupported"
            ),
            "evidence": "source sentence only",
            "expected_grounding": {
                "clean": False,
                "injected": EXPECTED_INJECTED,
            },
            "prompt_frozen_before_controlled_evaluation": True,
            "heldout_opened_before_controlled_evaluation": True,
            "post_heldout_prompt_or_model_changes": False,
        }
    )

    template = load_prompt()
    results = []

    for index, manifest_row in enumerate(manifest, start=1):
        case_id = manifest_row["id"]
        condition = manifest_row["condition"]

        print(
            f"[{index:02d}/{len(manifest):02d}] "
            f"{case_id} ({condition})"
        )

        case_result = {
            "id": case_id,
            "domain": manifest_row["domain"],
            "condition": condition,
        }

        for state in ("clean", "injected"):
            payload = payload_for_manifest(
                manifest_row,
                state,
            )

            judgment = judge_case(
                payload["source_text"],
                payload["triples"],
                template=template,
            )

            expected = expected_grounding(
                condition,
                state,
            )

            case_result[state] = {
                "expected_grounding_error": expected,
                "grounding_error": judgment["grounding_error"],
                "matches_expected": (
                    judgment["grounding_error"] == expected
                ),
                "triple_count": judgment["triple_count"],
                "unsupported_count": (
                    judgment["unsupported_count"]
                ),
                "judgments": judgment["judgments"],
            }

        results.append(case_result)

    state_summary, by_condition = summarize(results)

    mismatches = []

    for row in results:
        for state in ("clean", "injected"):
            if not row[state]["matches_expected"]:
                mismatches.append(
                    {
                        "id": row["id"],
                        "condition": row["condition"],
                        "state": state,
                        "expected": row[state][
                            "expected_grounding_error"
                        ],
                        "observed": row[state][
                            "grounding_error"
                        ],
                    }
                )

    metadata["summary"] = state_summary
    metadata["by_condition"] = by_condition
    metadata["mismatch_count"] = len(mismatches)
    metadata["mismatches"] = mismatches

    RESULT.parent.mkdir(parents=True, exist_ok=True)

    with RESULT.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("clean")
    print(
        "  matching expected: "
        f"{state_summary['clean']['matching_expected']}"
        f"/{state_summary['clean']['n']}"
    )
    print(
        "  observed grounding errors: "
        f"{state_summary['clean']['observed_grounding_error']}"
    )

    print("injected")
    print(
        "  matching expected: "
        f"{state_summary['injected']['matching_expected']}"
        f"/{state_summary['injected']['n']}"
    )
    print(
        "  observed grounding errors: "
        f"{state_summary['injected']['observed_grounding_error']}"
    )

    print("by condition")

    for condition in (
        "disjointness",
        "domain_range",
        "cardinality",
        "temporal",
        "grounding",
    ):
        row = by_condition[condition]
        print(
            f"  {condition}: "
            f"clean {row['clean_matches']}/{row['n']}, "
            f"injected {row['injected_matches']}/{row['n']}, "
            "injected grounding errors "
            f"{row['injected_observed_grounding_error']}"
            f"/{row['n']}"
        )

    print(f"mismatches: {len(mismatches)}")

    for row in mismatches:
        print(
            f"  {row['id']} / {row['condition']} / "
            f"{row['state']}: expected {row['expected']}, "
            f"observed {row['observed']}"
        )

    print(f"model digest: {metadata['model_digest']}")
    print(f"ollama version: {metadata['ollama_version']}")
    print(f"prompt sha256: {metadata['prompt_sha256']}")
    print(f"wrote: {RESULT.relative_to(ROOT)}")
    print(f"metadata: {METADATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
