import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "experiments" / "controlled_candidate_pool.json"
OUTPUT = ROOT / "experiments" / "controlled_review_batch.jsonl"

SEED = 42
REVIEW_N = 10

DOMAINS = {
    "movie": {
        "baseline": ROOT / "outputs" / "baseline" / "movie_llama31.jsonl",
        "stats": (
            ROOT
            / "outputs"
            / "evaluation"
            / "llama31"
            / "ont_1_movie_llm_stats.jsonl"
        ),
    },
    "music": {
        "baseline": ROOT / "outputs" / "baseline" / "music_llama31.jsonl",
        "stats": (
            ROOT
            / "outputs"
            / "evaluation"
            / "llama31"
            / "ont_2_music_llm_stats.jsonl"
        ),
    },
}

CATEGORIES = (
    "cardinality",
    "disjointness",
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


def case_number(case_id):
    return int(case_id.rsplit("_", 1)[1])


def choose_review_batch(pool, source_rows, review_n=REVIEW_N, seed=SEED):
    selected = []
    used_ids = set()

    verified_ids = set()

    for domain in ("movie", "music"):
        domain_pool = pool["domains"][domain]

        for key in ("verified_domain_range", "verified_temporal"):
            for item in domain_pool[key]:
                verified_ids.add(item["case_id"])

    for domain in ("movie", "music"):
        for category in CATEGORIES:
            candidates = [
                dict(item)
                for item in pool["domains"][domain][category]
                if item["case_id"] not in used_ids
                and item["case_id"] not in verified_ids
            ]

            candidates.sort(key=lambda item: case_number(item["case_id"]))

            rng = random.Random(
                f"{seed}:{domain}:{category}"
            )
            rng.shuffle(candidates)

            if len(candidates) < review_n:
                raise RuntimeError(
                    f"{domain}/{category}: requested {review_n} review "
                    f"cases but only {len(candidates)} are available"
                )

            chosen = candidates[:review_n]

            for order, candidate in enumerate(chosen, start=1):
                case_id = candidate["case_id"]
                source = source_rows[domain].get(case_id)

                if source is None:
                    raise RuntimeError(
                        f"{domain}/{category}: missing source row for {case_id}"
                    )

                selected.append(
                    {
                        "id": case_id,
                        "domain": domain,
                        "error_type": category,
                        "review_order": order,
                        "sent": source["sent"],
                        "triples": source["triples"],
                        "proposal": {
                            key: value
                            for key, value in candidate.items()
                            if key
                            not in {
                                "case_id",
                                "pilot_status",
                                "manual_review_required",
                            }
                        },
                        "pilot_status": candidate.get(
                            "pilot_status",
                            "not_annotated",
                        ),
                        "accepted": None,
                        "notes": "",
                    }
                )

                used_ids.add(case_id)

    return selected


def load_source_rows():
    source_rows = {}

    for domain, config in DOMAINS.items():
        baseline = index_unique(
            read_jsonl(config["baseline"]),
            config["baseline"],
        )
        stats = index_unique(
            read_jsonl(config["stats"]),
            config["stats"],
        )

        joined = {}

        for case_id, generation in baseline.items():
            stat = stats.get(case_id)

            if stat is None:
                continue

            sent = stat.get("sent")
            triples = generation.get("triples")

            if (
                isinstance(sent, str)
                and sent.strip()
                and isinstance(triples, list)
            ):
                joined[case_id] = {
                    "sent": sent,
                    "triples": triples,
                }

        source_rows[domain] = joined

    return source_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if OUTPUT.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {OUTPUT}. "
            "Use --overwrite only if you intend to replace it."
        )

    with POOL.open(encoding="utf-8") as f:
        pool = json.load(f)

    rows = choose_review_batch(
        pool,
        load_source_rows(),
    )

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    counts = {}

    for row in rows:
        key = (row["domain"], row["error_type"])
        counts[key] = counts.get(key, 0) + 1

    for domain in ("movie", "music"):
        print(domain)
        for category in CATEGORIES:
            print(
                f"  {category}: "
                f"{counts[(domain, category)]}"
            )

    print(f"total: {len(rows)}")
    print(f"wrote: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
