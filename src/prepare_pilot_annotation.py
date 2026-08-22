import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SELECTION = ROOT / "experiments/pilot_selection.json"
OUTPUT = ROOT / "experiments/pilot_annotation.jsonl"

DOMAINS = {
    "movie": {
        "baseline": ROOT / "outputs/baseline/movie_llama31.jsonl",
        "stats": ROOT / "outputs/evaluation/llama31/ont_1_movie_llm_stats.jsonl",
        "prefix": "ont_1_movie_test_",
    },
    "music": {
        "baseline": ROOT / "outputs/baseline/music_llama31.jsonl",
        "stats": ROOT / "outputs/evaluation/llama31/ont_2_music_llm_stats.jsonl",
        "prefix": "ont_2_music_test_",
    },
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
    indexed = {}

    for row in rows:
        case_id = row.get("id")

        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"Missing or invalid id in {path}")

        if case_id in indexed:
            raise RuntimeError(f"Duplicate id in {path}: {case_id}")

        indexed[case_id] = row

    return indexed


def validate_triples(triples, case_id, field):
    if not isinstance(triples, list):
        raise RuntimeError(f"{case_id}: {field} is not a list")

    for triple in triples:
        if (
            not isinstance(triple, list)
            or len(triple) != 3
            or not all(isinstance(value, str) for value in triple)
        ):
            raise RuntimeError(
                f"{case_id}: invalid triple in {field}: {triple!r}"
            )


def prepare_domain(domain, config, selected_ids):
    baseline = index_unique(
        read_jsonl(config["baseline"]),
        config["baseline"],
    )
    stats = index_unique(
        read_jsonl(config["stats"]),
        config["stats"],
    )

    prepared = []

    for case_id in selected_ids:
        if not case_id.startswith(config["prefix"]):
            raise RuntimeError(
                f"{domain}: unexpected case id: {case_id}"
            )

        if case_id not in baseline:
            raise RuntimeError(
                f"{domain}: selected case missing from baseline: {case_id}"
            )

        if case_id not in stats:
            raise RuntimeError(
                f"{domain}: selected case missing from stats: {case_id}"
            )

        generation = baseline[case_id]
        stat = stats[case_id]

        if generation.get("status") != "ok":
            raise RuntimeError(
                f"{case_id}: selected generation is not ok"
            )

        if generation.get("error") is not None:
            raise RuntimeError(
                f"{case_id}: selected generation has an error"
            )

        if generation.get("done_reason") != "stop":
            raise RuntimeError(
                f"{case_id}: selected generation did not stop normally"
            )

        response = generation.get("response")

        if not isinstance(response, str):
            raise RuntimeError(
                f"{case_id}: missing raw model response"
            )

        sent = stat.get("sent")

        if not isinstance(sent, str) or not sent.strip():
            raise RuntimeError(
                f"{case_id}: missing source sentence"
            )

        triples = generation.get("triples")
        parsed_triples_raw = generation.get("parsed_triples_raw")

        validate_triples(triples, case_id, "triples")
        validate_triples(
            parsed_triples_raw,
            case_id,
            "parsed_triples_raw",
        )

        prepared.append(
            {
                "id": case_id,
                "domain": domain,
                "sent": sent,
                "response": response,
                "triples": triples,
                "parsed_triples_raw": parsed_triples_raw,
                "annotated": False,
                "labels": [],
                "uncovered": False,
                "parse_issue": False,
                "notes": "",
            }
        )

    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if OUTPUT.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {OUTPUT}. "
            "Use --overwrite only if you intend to replace it."
        )

    with SELECTION.open(encoding="utf-8") as f:
        selection = json.load(f)

    selected_ids = selection.get("selected_ids")

    if not isinstance(selected_ids, dict):
        raise RuntimeError("Invalid selected_ids in pilot selection")

    expected_domains = set(DOMAINS)

    if set(selected_ids) != expected_domains:
        raise RuntimeError(
            f"Unexpected domains in pilot selection: "
            f"{sorted(selected_ids)}"
        )

    all_selected = []

    for domain in DOMAINS:
        ids = selected_ids[domain]

        if not isinstance(ids, list):
            raise RuntimeError(
                f"{domain}: selected ids are not a list"
            )

        if len(ids) != selection.get("per_domain_n"):
            raise RuntimeError(
                f"{domain}: selected count does not match per_domain_n"
            )

        if len(ids) != len(set(ids)):
            raise RuntimeError(
                f"{domain}: duplicate selected ids"
            )

        all_selected.extend(ids)

    if len(all_selected) != len(set(all_selected)):
        raise RuntimeError("Duplicate selected ids across domains")

    rows = []

    for domain, config in DOMAINS.items():
        rows.extend(
            prepare_domain(
                domain,
                config,
                selected_ids[domain],
            )
        )

    expected_total = (
        selection["per_domain_n"] * len(DOMAINS)
    )

    if len(rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} annotation rows, "
            f"got {len(rows)}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    print(
        f"prepared: movie={len(selected_ids['movie'])} "
        f"music={len(selected_ids['music'])} "
        f"total={len(rows)}"
    )
    print("truncated selected: 0")
    print(f"wrote: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
