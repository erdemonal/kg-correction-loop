from src.prepare_controlled_review import choose_review_batch


def candidate(case_id):
    return {
        "case_id": case_id,
        "pilot_status": "not_annotated",
        "manual_review_required": True,
        "injection": ["a", "p", "b"],
    }


def make_pool():
    domains = {}

    for domain, prefix in (
        ("movie", "ont_1_movie_test_"),
        ("music", "ont_2_music_test_"),
    ):
        domains[domain] = {
            "cardinality": [
                candidate(f"{prefix}{i}")
                for i in range(1, 21)
            ],
            "disjointness": [
                candidate(f"{prefix}{i}")
                for i in range(21, 41)
            ],
            "grounding": [
                candidate(f"{prefix}{i}")
                for i in range(41, 61)
            ],
            "verified_domain_range": [
                {"case_id": f"{prefix}90"}
            ],
            "verified_temporal": [
                {"case_id": f"{prefix}91"}
            ],
        }

    return {
        "domains": domains,
    }


def make_sources():
    sources = {}

    for domain, prefix in (
        ("movie", "ont_1_movie_test_"),
        ("music", "ont_2_music_test_"),
    ):
        sources[domain] = {
            f"{prefix}{i}": {
                "sent": f"Sentence {i}",
                "triples": [["a", "director", "b"]],
            }
            for i in range(1, 100)
        }

    return sources


def test_review_batch_is_deterministic():
    pool = make_pool()
    sources = make_sources()

    first = choose_review_batch(
        pool,
        sources,
        review_n=5,
        seed=42,
    )
    second = choose_review_batch(
        pool,
        sources,
        review_n=5,
        seed=42,
    )

    assert first == second


def test_review_batch_has_no_reused_case_ids():
    rows = choose_review_batch(
        make_pool(),
        make_sources(),
        review_n=5,
        seed=42,
    )

    ids = [row["id"] for row in rows]

    assert len(ids) == len(set(ids))


def test_review_batch_is_balanced():
    rows = choose_review_batch(
        make_pool(),
        make_sources(),
        review_n=5,
        seed=42,
    )

    counts = {}

    for row in rows:
        key = (row["domain"], row["error_type"])
        counts[key] = counts.get(key, 0) + 1

    assert set(counts.values()) == {5}
    assert len(rows) == 30


def test_verified_cases_are_not_selected_for_structural_review():
    pool = make_pool()
    pool["domains"]["movie"]["cardinality"].append(
        candidate("ont_1_movie_test_90")
    )

    rows = choose_review_batch(
        pool,
        make_sources(),
        review_n=5,
        seed=42,
    )

    assert "ont_1_movie_test_90" not in {
        row["id"]
        for row in rows
    }
