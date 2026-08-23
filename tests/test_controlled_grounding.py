import json

import pytest

from src.run_controlled_grounding import (
    EXPECTED_INJECTED,
    expected_grounding,
    load_payload,
    validate_manifest,
)


def test_expected_grounding_pattern_is_locked():
    assert EXPECTED_INJECTED == {
        "disjointness": True,
        "domain_range": False,
        "cardinality": False,
        "temporal": True,
        "grounding": True,
    }

    for condition in EXPECTED_INJECTED:
        assert expected_grounding(condition, "clean") is False

    assert expected_grounding(
        "disjointness",
        "injected",
    ) is True
    assert expected_grounding(
        "domain_range",
        "injected",
    ) is False
    assert expected_grounding(
        "cardinality",
        "injected",
    ) is False
    assert expected_grounding(
        "temporal",
        "injected",
    ) is True
    assert expected_grounding(
        "grounding",
        "injected",
    ) is True


def test_unknown_state_is_rejected():
    with pytest.raises(ValueError):
        expected_grounding("grounding", "other")


def test_load_payload_accepts_content_only_schema(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "domain": "movie",
                "source_text": "A was directed by B.",
                "triples": [["A", "director", "B"]],
            }
        ),
        encoding="utf-8",
    )

    payload = load_payload(path)

    assert payload["triples"] == [["A", "director", "B"]]
    assert "background_types" not in payload


def test_load_payload_rejects_auxiliary_fields(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "domain": "movie",
                "source_text": "A was directed by B.",
                "triples": [["A", "director", "B"]],
                "background_types": [["B", "Q5"]],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        load_payload(path)


def test_manifest_requires_50_unique_cases():
    row = {
        "id": "case",
        "domain": "movie",
        "condition": "grounding",
        "files": {
            "grounding_clean": "clean.json",
            "grounding_injected": "injected.json",
        },
    }

    with pytest.raises(RuntimeError):
        validate_manifest([row] * 50)
