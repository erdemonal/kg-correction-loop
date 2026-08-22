import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PER_DOMAIN_N = 20

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

OUTPUT = ROOT / "experiments/pilot_selection.json"


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


def numeric_id(case_id, prefix):
    if not case_id.startswith(prefix):
        raise RuntimeError(
            f"Unexpected id for domain {prefix}: {case_id}"
        )

    match = re.fullmatch(re.escape(prefix) + r"(\d+)", case_id)

    if match is None:
        raise RuntimeError(f"Invalid case id: {case_id}")

    return int(match.group(1))


def validate_generation(row, case_id):
    status = row.get("status")
    error = row.get("error")
    done_reason = row.get("done_reason")

    if status == "ok":
        if error is not None or done_reason != "stop":
            raise RuntimeError(
                f"Inconsistent successful generation: {case_id}"
            )
    elif status == "truncated":
        if done_reason != "length":
            raise RuntimeError(
                f"Inconsistent truncated generation: {case_id}"
            )
    elif status != "api_error":
        raise RuntimeError(
            f"Unknown generation status for {case_id}: {status!r}"
        )


def select_domain(name, config):
    baseline_rows = read_jsonl(config["baseline"])
    stats_rows = read_jsonl(config["stats"])

    baseline = index_unique(baseline_rows, config["baseline"])
    stats = index_unique(stats_rows, config["stats"])

    if set(baseline) != set(stats):
        missing = sorted(set(baseline) - set(stats))
        extra = sorted(set(stats) - set(baseline))

        raise RuntimeError(
            f"{name}: baseline/stats id mismatch; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    eligible = []

    for case_id, row in baseline.items():
        numeric_id(case_id, config["prefix"])
        validate_generation(row, case_id)

        if (
            row["status"] == "ok"
            and row["error"] is None
            and row["done_reason"] == "stop"
        ):
            if case_id not in stats:
                raise RuntimeError(
                    f"Eligible case missing from stats: {case_id}"
                )

            sent = stats[case_id].get("sent")

            if not isinstance(sent, str) or not sent.strip():
                raise RuntimeError(
                    f"Missing source sentence in stats: {case_id}"
                )

            eligible.append(case_id)

    for case_id in stats:
        numeric_id(case_id, config["prefix"])

    eligible = sorted(
        eligible,
        key=lambda case_id: numeric_id(case_id, config["prefix"]),
    )

    if len(eligible) < PER_DOMAIN_N:
        raise RuntimeError(
            f"{name}: requested {PER_DOMAIN_N} cases, "
            f"but only {len(eligible)} are eligible"
        )

    rng = random.Random(SEED)
    selected = rng.sample(eligible, PER_DOMAIN_N)
    selected = sorted(
        selected,
        key=lambda case_id: numeric_id(case_id, config["prefix"]),
    )

    if len(selected) != PER_DOMAIN_N:
        raise RuntimeError(
            f"{name}: expected {PER_DOMAIN_N} selected cases"
        )

    if len(set(selected)) != len(selected):
        raise RuntimeError(f"{name}: duplicate selected ids")

    for case_id in selected:
        row = baseline[case_id]

        if row["status"] != "ok":
            raise RuntimeError(
                f"{name}: non-ok case selected: {case_id}"
            )

        if row["done_reason"] == "length":
            raise RuntimeError(
                f"{name}: truncated case selected: {case_id}"
            )

    return len(eligible), selected


def main():
    eligible_counts = {}
    selected_ids = {}

    for name, config in DOMAINS.items():
        eligible_count, selected = select_domain(name, config)
        eligible_counts[name] = eligible_count
        selected_ids[name] = selected

    combined = selected_ids["movie"] + selected_ids["music"]

    if len(combined) != 2 * PER_DOMAIN_N:
        raise RuntimeError("Expected 40 selected cases")

    if len(set(combined)) != len(combined):
        raise RuntimeError("Duplicate ids across domains")

    result = {
        "seed": SEED,
        "per_domain_n": PER_DOMAIN_N,
        "eligible_counts": eligible_counts,
        "selected_ids": selected_ids,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(
        f"eligible: movie={eligible_counts['movie']} "
        f"music={eligible_counts['music']}"
    )
    print(
        f"selected: movie={len(selected_ids['movie'])} "
        f"music={len(selected_ids['music'])} "
        f"total={len(combined)}"
    )
    print("selected truncated: 0")
    print(f"wrote: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
