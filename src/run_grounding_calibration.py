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
PILOT = ROOT / "experiments" / "pilot_annotation.jsonl"
SPLIT = ROOT / "experiments" / "grounding_calibration_split.json"
RESULTS_ROOT = ROOT / "results"


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


def index_unique(rows):
    output = {}

    for row in rows:
        case_id = row["id"]

        if case_id in output:
            raise RuntimeError(f"Duplicate pilot id: {case_id}")

        output[case_id] = row

    return output


def metrics(rows):
    tp = fp = tn = fn = 0

    for row in rows:
        human = row["human_grounding_error"]
        predicted = row["predicted_grounding_error"]

        if human and predicted:
            tp += 1
        elif not human and predicted:
            fp += 1
        elif not human and not predicted:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def output_paths(split_name):
    result = (
        RESULTS_ROOT
        / f"grounding_judge_{JUDGE_VERSION}_{split_name}.jsonl"
    )
    metadata = result.with_suffix(
        result.suffix + ".meta.json"
    )
    return result, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("calibration", "heldout"),
        required=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not SPLIT.exists():
        raise SystemExit(
            "Calibration split not found. Run "
            "`python -m src.prepare_grounding_calibration` first."
        )

    result_path, metadata_path = output_paths(args.split)

    if (
        (result_path.exists() or metadata_path.exists())
        and not args.overwrite
    ):
        raise SystemExit(
            "Grounding result already exists. Use --overwrite "
            "only if you intend to replace it."
        )

    pilot = index_unique(read_jsonl(PILOT))
    split_payload = json.loads(
        SPLIT.read_text(encoding="utf-8")
    )
    selected = split_payload[args.split]

    metadata = model_metadata()
    metadata.update(
        {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "split": args.split,
            "split_file": str(SPLIT.relative_to(ROOT)),
            "pilot_file": str(PILOT.relative_to(ROOT)),
            "case_aggregation": (
                "grounding_error if any asserted triple "
                "is unsupported"
            ),
            "evidence": "source sentence only",
        }
    )

    template = load_prompt()
    results = []

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    for index, selected_row in enumerate(selected, start=1):
        case_id = selected_row["id"]
        row = pilot[case_id]

        print(
            f"[{index:02d}/{len(selected):02d}] "
            f"{case_id}"
        )

        judgment = judge_case(
            row["sent"],
            row["triples"],
            template=template,
        )

        results.append(
            {
                "id": case_id,
                "domain": row["domain"],
                "human_grounding_error": (
                    selected_row["human_grounding_error"]
                ),
                "predicted_grounding_error": (
                    judgment["grounding_error"]
                ),
                "triple_count": judgment["triple_count"],
                "unsupported_count": (
                    judgment["unsupported_count"]
                ),
                "judgments": judgment["judgments"],
            }
        )

    score = metrics(results)
    metadata["metrics"] = score
    metadata["cases"] = len(results)
    metadata["model"] = MODEL

    with result_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    mismatches = [
        row
        for row in results
        if row["human_grounding_error"]
        != row["predicted_grounding_error"]
    ]

    print(f"judge version: {JUDGE_VERSION}")
    print(f"cases: {score['n']}")
    print(f"tp: {score['tp']}")
    print(f"fp: {score['fp']}")
    print(f"tn: {score['tn']}")
    print(f"fn: {score['fn']}")
    print(f"precision: {score['precision']:.3f}")
    print(f"recall: {score['recall']:.3f}")
    print(f"f1: {score['f1']:.3f}")
    print(f"accuracy: {score['accuracy']:.3f}")
    print(f"mismatches: {len(mismatches)}")

    for row in mismatches:
        print(
            f"  {row['id']}: "
            f"human={row['human_grounding_error']} "
            f"predicted={row['predicted_grounding_error']}"
        )

    print(f"model digest: {metadata['model_digest']}")
    print(f"ollama version: {metadata['ollama_version']}")
    print(f"prompt sha256: {metadata['prompt_sha256']}")
    print(f"wrote: {result_path.relative_to(ROOT)}")
    print(f"metadata: {metadata_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
