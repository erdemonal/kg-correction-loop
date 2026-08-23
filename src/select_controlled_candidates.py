import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.controlled_cases import DOMAIN_TYPES, RANGE_TYPES, RELATIONS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "controlled_candidate_pool.json"
PILOT_ANNOTATION = ROOT / "experiments" / "pilot_annotation.jsonl"

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
        "prefix": "ont_1_movie_test_",
        "cardinality_property": "director",
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
        "prefix": "ont_2_music_test_",
        "cardinality_property": "composer",
    },
}

GENERIC_TERMS = {
    "movie": {
        "film",
        "human",
        "city",
        "country",
        "film production company",
        "written work",
        "film character",
        "award",
        "genre",
    },
    "music": {
        "musical work",
        "human",
        "album",
        "music genre",
        "language",
        "voice",
        "award",
        "musical profession",
        "record producer",
        "composer",
        "music",
        "composed musical work",
        "single",
        "singer",
        "disc jockey",
        "lyricist",
    },
}

VERIFIED_CASES = {
    "movie": {
        "domain_range": [
            {
                "case_id": "ont_1_movie_test_767",
                "role": "primary",
                "property": "narrative_location",
                "expected_type": "Q515",
                "explicit_type": "Q6256",
            }
        ],
        "temporal": [
            {
                "case_id": "ont_1_movie_test_467",
                "role": "primary",
                "first_property": "premiereDate",
                "second_property": "theatricalReleaseDate",
                "first_date": "2007-09-15",
                "second_date": "2008-04-18",
            },
            {
                "case_id": "ont_1_movie_test_235",
                "role": "alternative",
                "first_property": "premiereDate",
                "second_property": "theatricalReleaseDate",
            },
        ],
    },
    "music": {
        "domain_range": [
            {
                "case_id": "ont_2_music_test_230",
                "role": "primary",
                "property": "record_label",
                "expected_type": "Q482994",
                "explicit_type": "Q134556",
            }
        ],
        "temporal": [
            {
                "case_id": "ont_2_music_test_215",
                "role": "primary",
                "first_property": "radioPremiereDate",
                "second_property": "digitalReleaseDate",
                "first_date": "2011-11-11",
                "second_date": "2011-11-14",
            },
            {
                "case_id": "ont_2_music_test_120",
                "role": "alternative",
                "first_property": "recordingDate",
                "second_property": "releaseDate",
            },
        ],
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


def numeric_case_key(case_id):
    try:
        return (0, int(case_id.rsplit("_", 1)[1]))
    except (IndexError, ValueError):
        return (1, case_id)


def normalized(value):
    return " ".join(value.strip().lower().replace("_", " ").split())


def validate_triples(triples):
    if not isinstance(triples, list):
        return False

    return all(
        isinstance(triple, list)
        and len(triple) == 3
        and all(isinstance(value, str) for value in triple)
        for triple in triples
    )


def has_unknown_relation(domain, triples):
    known = RELATIONS[domain]
    return any(triple[1] not in known for triple in triples)


def has_generic_placeholder(domain, triples):
    generic = GENERIC_TERMS[domain]

    for subject, _, obj in triples:
        if normalized(subject) in generic or normalized(obj) in generic:
            return True

    return False


def load_pilot_status(path=PILOT_ANNOTATION):
    if not path.exists():
        return {}

    status = {}

    for row in read_jsonl(path):
        case_id = row.get("id")

        flagged = (
            bool(row.get("labels"))
            or bool(row.get("uncovered"))
            or bool(row.get("parse_issue"))
        )

        status[case_id] = "flagged" if flagged else "clean"

    return status


def human_support(domain, triples):
    support = defaultdict(list)

    for subject, predicate, obj in triples:
        if DOMAIN_TYPES[domain].get(predicate) == "Q5":
            support[subject].append(
                {
                    "relation": predicate,
                    "position": "subject",
                }
            )

        if RANGE_TYPES[domain].get(predicate) == "Q5":
            support[obj].append(
                {
                    "relation": predicate,
                    "position": "object",
                }
            )

    return dict(support)


def subject_support(domain, triples, class_id):
    support = defaultdict(list)

    for subject, predicate, _ in triples:
        if DOMAIN_TYPES[domain].get(predicate) == class_id:
            support[subject].append(predicate)

    return dict(support)


def first_cardinality_candidate(domain, case_id, triples, target_property):
    by_subject = defaultdict(list)

    for subject, predicate, obj in triples:
        if predicate == target_property:
            by_subject[subject].append(obj)

    for subject in sorted(by_subject):
        values = sorted(set(by_subject[subject]))

        if len(values) == 1:
            return {
                "case_id": case_id,
                "subject": subject,
                "property": target_property,
                "value": values[0],
            }

    return None


def movie_disjointness_candidate(case_id, triples):
    humans = human_support("movie", triples)
    films = subject_support("movie", triples, "Q11424")

    existing = {
        (subject, predicate, obj)
        for subject, predicate, obj in triples
    }

    for film in sorted(films):
        for human in sorted(humans):
            if film == human:
                continue

            proposed = (film, "production_company", human)

            if proposed in existing:
                continue

            return {
                "case_id": case_id,
                "injection": list(proposed),
                "human_support": humans[human][0],
            }

    return None


def music_disjointness_candidate(case_id, triples):
    humans = human_support("music", triples)

    if len(humans) < 2:
        return None

    ordered = sorted(humans)

    for subject in ordered:
        for obj in ordered:
            if subject == obj:
                continue

            proposed = (subject, "performer", obj)

            if proposed in {
                (s, p, o)
                for s, p, o in triples
            }:
                continue

            return {
                "case_id": case_id,
                "injection": list(proposed),
                "subject_human_support": humans[subject][0],
                "object_human_support": humans[obj][0],
            }

    return None


def movie_grounding_candidate(case_id, triples):
    films = subject_support("movie", triples, "Q11424")
    humans = human_support("movie", triples)
    existing_directors = {
        (subject, obj)
        for subject, predicate, obj in triples
        if predicate == "director"
    }

    for film in sorted(films):
        for human in sorted(humans):
            if (film, human) in existing_directors:
                continue

            supports = [
                item
                for item in humans[human]
                if item["relation"] != "director"
            ]

            if not supports:
                continue

            return {
                "case_id": case_id,
                "injection": [film, "director", human],
                "human_support": supports[0],
            }

    return None


def music_grounding_candidate(case_id, triples):
    works = subject_support("music", triples, "Q2188189")
    existing_composers = {
        (subject, obj)
        for subject, predicate, obj in triples
        if predicate == "composer"
    }

    by_work = defaultdict(list)

    for subject, predicate, obj in triples:
        if (
            subject in works
            and predicate != "composer"
            and RANGE_TYPES["music"].get(predicate) == "Q5"
        ):
            by_work[subject].append((obj, predicate))

    for work in sorted(by_work):
        for human, support_relation in sorted(by_work[work]):
            if (work, human) in existing_composers:
                continue

            return {
                "case_id": case_id,
                "injection": [work, "composer", human],
                "human_support": {
                    "relation": support_relation,
                    "position": "object",
                },
            }

    return None


def annotate_candidate(candidate, pilot_status):
    if candidate is None:
        return None

    status = pilot_status.get(candidate["case_id"], "not_annotated")
    candidate = dict(candidate)
    candidate["pilot_status"] = status
    candidate["manual_review_required"] = status != "clean"
    return candidate


def prepare_verified_cases(stats, domain):
    prepared = {
        "domain_range": [],
        "temporal": [],
    }

    for category in prepared:
        for item in VERIFIED_CASES[domain][category]:
            case_id = item["case_id"]

            if case_id not in stats:
                raise RuntimeError(
                    f"Verified case missing from stats: {case_id}"
                )

            sent = stats[case_id].get("sent")

            if not isinstance(sent, str) or not sent.strip():
                raise RuntimeError(
                    f"Verified case has no source sentence: {case_id}"
                )

            record = dict(item)
            record["sent"] = sent
            prepared[category].append(record)

    return prepared


def build_domain_candidates(
    domain,
    baseline_rows,
    stats_rows,
    pilot_status,
    *,
    include_verified=True,
):
    config = DOMAINS[domain]
    baseline = index_unique(baseline_rows, config["baseline"])
    stats = index_unique(stats_rows, config["stats"])

    exclusions = Counter()
    eligible = []

    for case_id in sorted(baseline, key=numeric_case_key):
        row = baseline[case_id]

        if not case_id.startswith(config["prefix"]):
            exclusions["unexpected_id"] += 1
            continue

        if row.get("done_reason") != "stop":
            exclusions["truncated"] += 1
            continue

        if row.get("status") != "ok":
            exclusions["generation_status"] += 1
            continue

        if row.get("error") is not None:
            exclusions["generation_error"] += 1
            continue

        triples = row.get("triples")

        if not validate_triples(triples):
            exclusions["invalid_triples"] += 1
            continue

        if has_unknown_relation(domain, triples):
            exclusions["unknown_relation"] += 1
            continue

        if has_generic_placeholder(domain, triples):
            exclusions["generic_placeholder"] += 1
            continue

        if pilot_status.get(case_id) == "flagged":
            exclusions["pilot_flagged"] += 1
            continue

        stat = stats.get(case_id)

        if stat is None:
            exclusions["missing_stats"] += 1
            continue

        sent = stat.get("sent")

        if not isinstance(sent, str) or not sent.strip():
            exclusions["missing_source"] += 1
            continue

        eligible.append(
            {
                "id": case_id,
                "sent": sent,
                "triples": triples,
            }
        )

    cardinality = []
    disjointness = []
    grounding = []

    for row in eligible:
        case_id = row["id"]
        triples = row["triples"]

        candidate = first_cardinality_candidate(
            domain,
            case_id,
            triples,
            config["cardinality_property"],
        )

        if candidate is not None:
            cardinality.append(
                annotate_candidate(candidate, pilot_status)
            )

        if domain == "movie":
            candidate = movie_disjointness_candidate(case_id, triples)
        else:
            candidate = music_disjointness_candidate(case_id, triples)

        if candidate is not None:
            disjointness.append(
                annotate_candidate(candidate, pilot_status)
            )

        if domain == "movie":
            candidate = movie_grounding_candidate(case_id, triples)
        else:
            candidate = music_grounding_candidate(case_id, triples)

        if candidate is not None:
            grounding.append(
                annotate_candidate(candidate, pilot_status)
            )

    if include_verified:
        verified = prepare_verified_cases(stats, domain)
    else:
        verified = {
            "domain_range": [],
            "temporal": [],
        }

    return {
        "eligible_baseline_cases": len(eligible),
        "excluded": dict(sorted(exclusions.items())),
        "cardinality": cardinality,
        "disjointness": disjointness,
        "grounding": grounding,
        "verified_domain_range": verified["domain_range"],
        "verified_temporal": verified["temporal"],
    }


def build_candidate_pool():
    pilot_status = load_pilot_status()

    domains = {}

    for domain, config in DOMAINS.items():
        domains[domain] = build_domain_candidates(
            domain,
            read_jsonl(config["baseline"]),
            read_jsonl(config["stats"]),
            pilot_status,
        )

    return {
        "version": 1,
        "purpose": "structural candidate pool for controlled case selection",
        "selection_applied": False,
        "faults_injected": False,
        "domains": domains,
    }


def print_summary(pool):
    for domain in ("movie", "music"):
        result = pool["domains"][domain]

        print(domain)
        print(
            f"  eligible baseline cases: "
            f"{result['eligible_baseline_cases']}"
        )
        print(f"  cardinality: {len(result['cardinality'])}")
        print(f"  disjointness: {len(result['disjointness'])}")
        print(f"  grounding: {len(result['grounding'])}")
        print(
            f"  verified domain/range: "
            f"{len(result['verified_domain_range'])}"
        )
        print(
            f"  verified temporal: "
            f"{len(result['verified_temporal'])}"
        )

        if result["excluded"]:
            print("  excluded:")
            for reason, count in result["excluded"].items():
                print(f"    {reason}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if OUTPUT.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {OUTPUT}. "
            "Use --overwrite only if you intend to replace it."
        )

    pool = build_candidate_pool()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(pool, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print_summary(pool)
    print(f"wrote: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
