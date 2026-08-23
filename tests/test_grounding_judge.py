import json

import pytest

from src.grounding_judge import (
    aggregate_case,
    parse_judgment,
    render_prompt,
)


def test_render_prompt_contains_only_supplied_case_content():
    template = (
        "Source: {source_text}\n"
        "Triple: {predicate}({subject}, {object})"
    )

    rendered = render_prompt(
        "A was directed by B.",
        ["A", "director", "B"],
        template=template,
    )

    assert rendered == (
        "Source: A was directed by B.\n"
        "Triple: director(A, B)"
    )


def test_parse_judgment_accepts_locked_schema():
    payload = parse_judgment(
        json.dumps(
            {
                "verdict": "SUPPORTED",
                "reason": "The sentence states the relation.",
            }
        )
    )

    assert payload["verdict"] == "SUPPORTED"


def test_parse_judgment_rejects_extra_fields():
    with pytest.raises(RuntimeError):
        parse_judgment(
            json.dumps(
                {
                    "verdict": "SUPPORTED",
                    "reason": "ok",
                    "confidence": 1.0,
                }
            )
        )


def test_case_aggregation_flags_any_unsupported_triple():
    judgments = [
        {"verdict": "SUPPORTED"},
        {"verdict": "UNSUPPORTED"},
    ]

    assert aggregate_case(judgments) is True
    assert aggregate_case(
        [{"verdict": "SUPPORTED"}]
    ) is False
    assert aggregate_case([]) is False
