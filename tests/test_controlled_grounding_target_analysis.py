import pytest

from src.analyze_controlled_grounding import (
    EXPECTED_TARGET,
    delta,
    target_detection,
    validate_delta,
)


def test_expected_target_pattern_is_locked():
    assert EXPECTED_TARGET == {
        "disjointness": True,
        "domain_range": False,
        "cardinality": False,
        "temporal": True,
        "grounding": True,
    }


def test_addition_target_uses_only_added_triple_verdict():
    case_delta = {
        "added": [("A", "director", "B")],
        "removed": [],
        "unchanged": [("A", "screenwriter", "C")],
    }
    clean_map = {
        ("A", "screenwriter", "C"): "UNSUPPORTED",
    }
    injected_map = {
        ("A", "screenwriter", "C"): "UNSUPPORTED",
        ("A", "director", "B"): "SUPPORTED",
    }

    result = target_detection(
        "domain_range",
        case_delta,
        clean_map,
        injected_map,
    )

    assert result["observed_grounding_error"] is False
    assert result["matches_expected"] is True


def test_background_false_flag_does_not_create_target_detection():
    case_delta = {
        "added": [("A", "director", "B")],
        "removed": [],
        "unchanged": [("A", "screenwriter", "C")],
    }
    clean_map = {
        ("A", "screenwriter", "C"): "UNSUPPORTED",
    }
    injected_map = {
        ("A", "screenwriter", "C"): "UNSUPPORTED",
        ("A", "director", "B"): "UNSUPPORTED",
    }

    result = target_detection(
        "grounding",
        case_delta,
        clean_map,
        injected_map,
    )

    assert result["observed_grounding_error"] is True
    assert result["matches_expected"] is True


def test_cardinality_removal_is_not_grounding_detectable():
    removed = ("A", "director", "B")
    case_delta = {
        "added": [],
        "removed": [removed],
        "unchanged": [],
    }
    clean_map = {removed: "UNSUPPORTED"}
    injected_map = {}

    result = target_detection(
        "cardinality",
        case_delta,
        clean_map,
        injected_map,
    )

    assert result["observed_grounding_error"] is False
    assert result["matches_expected"] is True
    assert result["removed_clean_supported"] is False


def test_temporal_swap_detected_if_any_new_assertion_is_unsupported():
    case_delta = {
        "added": [
            ("A", "premiereDate", "2020-02-01"),
            ("A", "releaseDate", "2020-01-01"),
        ],
        "removed": [
            ("A", "premiereDate", "2020-01-01"),
            ("A", "releaseDate", "2020-02-01"),
        ],
        "unchanged": [],
    }
    clean_map = {
        case_delta["removed"][0]: "SUPPORTED",
        case_delta["removed"][1]: "SUPPORTED",
    }
    injected_map = {
        case_delta["added"][0]: "UNSUPPORTED",
        case_delta["added"][1]: "SUPPORTED",
    }

    result = target_detection(
        "temporal",
        case_delta,
        clean_map,
        injected_map,
    )

    assert result["observed_grounding_error"] is True
    assert result["matches_expected"] is True


def test_delta_shapes_are_enforced():
    validate_delta(
        "grounding",
        {
            "added": [("A", "p", "B")],
            "removed": [],
            "unchanged": [],
        },
    )

    with pytest.raises(RuntimeError):
        validate_delta(
            "grounding",
            {
                "added": [],
                "removed": [],
                "unchanged": [],
            },
        )
