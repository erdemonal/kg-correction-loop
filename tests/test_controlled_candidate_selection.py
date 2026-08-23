from src.select_controlled_candidates import (
    build_domain_candidates,
    has_generic_placeholder,
    has_unknown_relation,
)


def baseline_row(case_id, triples, **overrides):
    row = {
        "id": case_id,
        "status": "ok",
        "error": None,
        "done_reason": "stop",
        "triples": triples,
    }
    row.update(overrides)
    return row


def stat_row(case_id, sent="Example source sentence."):
    return {
        "id": case_id,
        "sent": sent,
    }


def test_movie_clean_case_produces_structural_candidates():
    case_id = "ont_1_movie_test_117"
    baseline = [
        baseline_row(
            case_id,
            [
                ["Film A", "director", "Person A"],
                ["Film A", "screenwriter", "Person B"],
            ],
        )
    ]
    stats = [stat_row(case_id)]

    result = build_domain_candidates(
        "movie",
        baseline,
        stats,
        {},
        include_verified=False,
    )

    assert len(result["cardinality"]) == 1
    assert len(result["disjointness"]) == 1
    assert len(result["grounding"]) == 1

    assert result["cardinality"][0]["property"] == "director"
    assert result["disjointness"][0]["injection"][1] == "production_company"
    assert result["grounding"][0]["injection"] == [
        "Film A",
        "director",
        "Person B",
    ]


def test_music_clean_case_produces_structural_candidates():
    case_id = "ont_2_music_test_27"
    baseline = [
        baseline_row(
            case_id,
            [
                ["Work A", "composer", "Person A"],
                ["Work A", "lyrics_by", "Person B"],
            ],
        )
    ]
    stats = [stat_row(case_id)]

    result = build_domain_candidates(
        "music",
        baseline,
        stats,
        {},
        include_verified=False,
    )

    assert len(result["cardinality"]) == 1
    assert len(result["disjointness"]) == 1
    assert len(result["grounding"]) == 1

    assert result["cardinality"][0]["property"] == "composer"
    assert result["disjointness"][0]["injection"] == [
        "Person A",
        "performer",
        "Person B",
    ]
    assert result["grounding"][0]["injection"] == [
        "Work A",
        "composer",
        "Person B",
    ]


def test_flagged_pilot_case_is_excluded():
    case_id = "ont_1_movie_test_31"
    baseline = [
        baseline_row(
            case_id,
            [["Film A", "director", "Person A"]],
        )
    ]
    stats = [stat_row(case_id)]

    result = build_domain_candidates(
        "movie",
        baseline,
        stats,
        {case_id: "flagged"},
        include_verified=False,
    )

    assert result["eligible_baseline_cases"] == 0
    assert result["excluded"]["pilot_flagged"] == 1


def test_truncated_case_is_excluded():
    case_id = "ont_2_music_test_10"
    baseline = [
        baseline_row(
            case_id,
            [["Work A", "composer", "Person A"]],
            done_reason="length",
        )
    ]
    stats = [stat_row(case_id)]

    result = build_domain_candidates(
        "music",
        baseline,
        stats,
        {},
        include_verified=False,
    )

    assert result["eligible_baseline_cases"] == 0
    assert result["excluded"]["truncated"] == 1


def test_unknown_relation_is_detected():
    triples = [["Film A", "animation_director", "Person A"]]

    assert has_unknown_relation("movie", triples) is True


def test_generic_signature_placeholder_is_detected():
    triples = [["musical work", "composer", "human"]]

    assert has_generic_placeholder("music", triples) is True


def test_clean_pilot_candidate_does_not_require_manual_review():
    case_id = "ont_1_movie_test_117"
    baseline = [
        baseline_row(
            case_id,
            [
                ["Film A", "director", "Person A"],
                ["Film A", "screenwriter", "Person B"],
            ],
        )
    ]
    stats = [stat_row(case_id)]

    result = build_domain_candidates(
        "movie",
        baseline,
        stats,
        {case_id: "clean"},
        include_verified=False,
    )

    assert result["cardinality"][0]["pilot_status"] == "clean"
    assert result["cardinality"][0]["manual_review_required"] is False


def test_unannotated_candidate_requires_manual_review():
    case_id = "ont_2_music_test_500"
    baseline = [
        baseline_row(
            case_id,
            [
                ["Work A", "composer", "Person A"],
                ["Work A", "lyrics_by", "Person B"],
            ],
        )
    ]
    stats = [stat_row(case_id)]

    result = build_domain_candidates(
        "music",
        baseline,
        stats,
        {},
        include_verified=False,
    )

    assert result["grounding"][0]["pilot_status"] == "not_annotated"
    assert result["grounding"][0]["manual_review_required"] is True
