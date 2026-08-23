import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments" / "pilot_annotation.jsonl"
OUTPUT = ROOT / "experiments" / "grounding_calibration_split.json"

SEED = 42
CALIBRATION_FRACTION = 0.6


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


def eligible_rows(rows):
    return [
        row
        for row in rows
        if row.get("annotated") is True
        and row.get("parse_issue") is False
    ]


def stable_key(seed, domain, label, case_id):
    text = f"{seed}:{domain}:{int(label)}:{case_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_rows(
    rows,
    seed=SEED,
    calibration_fraction=CALIBRATION_FRACTION,
):
    groups = {}

    for row in eligible_rows(rows):
        domain = row["domain"]
        grounding = "grounding_error" in row.get("labels", [])
        groups.setdefault((domain, grounding), []).append(row)

    calibration = []
    heldout = []

    for (domain, grounding), group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda row: stable_key(
                seed,
                domain,
                grounding,
                row["id"],
            ),
        )

        if len(ordered) == 1:
            calibration_count = 1
        else:
            calibration_count = int(
                len(ordered) * calibration_fraction
            )
            calibration_count = max(
                1,
                min(len(ordered) - 1, calibration_count),
            )

        calibration.extend(ordered[:calibration_count])
        heldout.extend(ordered[calibration_count:])

    def compact(row):
        return {
            "id": row["id"],
            "domain": row["domain"],
            "human_grounding_error": (
                "grounding_error" in row.get("labels", [])
            ),
        }

    calibration = sorted(
        (compact(row) for row in calibration),
        key=lambda row: row["id"],
    )
    heldout = sorted(
        (compact(row) for row in heldout),
        key=lambda row: row["id"],
    )

    return calibration, heldout


def summarize(rows):
    summary = {
        "total": len(rows),
        "movie": 0,
        "music": 0,
        "grounding_error": 0,
        "no_grounding_error": 0,
    }

    for row in rows:
        summary[row["domain"]] += 1

        if row["human_grounding_error"]:
            summary["grounding_error"] += 1
        else:
            summary["no_grounding_error"] += 1

    return summary


def build_payload(rows):
    calibration, heldout = split_rows(rows)

    eligible = eligible_rows(rows)

    return {
        "version": 1,
        "source": "experiments/pilot_annotation.jsonl",
        "seed": SEED,
        "calibration_fraction": CALIBRATION_FRACTION,
        "eligibility": {
            "annotated": True,
            "exclude_parse_issue": True,
        },
        "counts": {
            "pilot_rows": len(rows),
            "eligible_rows": len(eligible),
            "excluded_parse_issue_or_unannotated": (
                len(rows) - len(eligible)
            ),
            "calibration": summarize(calibration),
            "heldout": summarize(heldout),
        },
        "calibration": calibration,
        "heldout": heldout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if OUTPUT.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {OUTPUT}. "
            "Use --overwrite only if you intend to replace it."
        )

    payload = build_payload(read_jsonl(PILOT))

    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"pilot rows: {payload['counts']['pilot_rows']}")
    print(f"eligible: {payload['counts']['eligible_rows']}")

    for split_name in ("calibration", "heldout"):
        counts = payload["counts"][split_name]
        print(split_name)
        print(f"  total: {counts['total']}")
        print(f"  movie: {counts['movie']}")
        print(f"  music: {counts['music']}")
        print(
            "  grounding_error: "
            f"{counts['grounding_error']}"
        )
        print(
            "  no_grounding_error: "
            f"{counts['no_grounding_error']}"
        )

    print(f"wrote: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
