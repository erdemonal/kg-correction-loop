import json

from src.run_controlled_repair import (
    grounding_identity,
    parse_repair_response,
    render_repair_prompt,
)


def test_strict_repair_parser_accepts_complete_graph():
    result = parse_repair_response(
        "director(Film, Person)\n"
        "cast_member(Film, Actor)",
        {"director", "cast_member"},
    )

    assert result["ok"] is True
    assert result["triples"] == [
        ["Film", "director", "Person"],
        ["Film", "cast_member", "Actor"],
    ]


def test_strict_repair_parser_rejects_explanation():
    result = parse_repair_response(
        "Here is the repaired graph:\n"
        "director(Film, Person)",
        {"director"},
    )

    assert result["ok"] is False
    assert result["failure"] == "unparseable_output"


def test_strict_repair_parser_rejects_unknown_relation():
    result = parse_repair_response(
        "unknown_relation(Film, Person)",
        {"director"},
    )

    assert result["ok"] is False
    assert result["failure"] == (
        "relation_outside_allowed_set"
    )


def test_strict_repair_parser_rejects_empty_output():
    result = parse_repair_response(
        "\n\n",
        {"director"},
    )

    assert result["ok"] is False
    assert result["failure"] == "empty_output"


def test_strict_repair_parser_removes_exact_duplicates():
    result = parse_repair_response(
        "director(Film, Person)\n"
        "director(Film, Person)",
        {"director"},
    )

    assert result["ok"] is True
    assert result["triples"] == [
        ["Film", "director", "Person"],
    ]


def test_rendered_prompt_contains_only_supplied_fields():
    template = (
        "S={source_text}\n"
        "R={allowed_relations}\n"
        "G={current_graph}\n"
        "F={feedback}"
    )
    prompt = render_repair_prompt(
        template,
        "Sentence",
        ("director",),
        (("Film", "director", "Person"),),
        (
            {
                "validator": "raw_shacl",
                "violation_id": "v1",
                "error_type": "cardinality",
                "focus": "Film",
                "path": "director",
                "message": "Missing value",
            },
        ),
    )

    assert "Sentence" in prompt
    assert "director(Film, Person)" in prompt
    assert '"violation_id": "v1"' in prompt
    assert "clean reference" not in prompt.lower()


def test_grounding_identity_is_stable_per_assertion():
    first = grounding_identity(
        ("A", "p", "B")
    )
    second = grounding_identity(
        ("A", "p", "B")
    )
    other = grounding_identity(
        ("A", "p", "C")
    )

    assert first == second
    assert first != other


def test_strict_repair_parser_rejects_trailing_prose():
    result = parse_repair_response(
        "director(Film, Person) because this is correct",
        {"director"},
    )

    assert result["ok"] is False
    assert result["failure"] == "unparseable_output"
