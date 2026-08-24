from src.analyze_validation_coverage import (
    grounding_outcome,
    main_pattern,
    summarize_rows,
)


def test_grounding_outcomes():
    assert grounding_outcome(True, True) == "true_positive"
    assert grounding_outcome(False, True) == "false_positive"
    assert grounding_outcome(True, False) == "false_negative"
    assert grounding_outcome(False, False) == "true_negative"


def make_row(
    condition,
    *,
    shacl=False,
    owlrl=False,
    owl=False,
    expected=False,
    grounding=False,
    split="none",
):
    return {
        "id": f"{condition}-{shacl}-{owl}-{grounding}",
        "domain": "movie",
        "condition": condition,
        "raw_shacl_detected": shacl,
        "owlrl_shacl_detected": owlrl,
        "owl_inconsistency_detected": owl,
        "grounding_expected_error": expected,
        "grounding_detected": grounding,
        "grounding_matches_expected": expected == grounding,
        "grounding_outcome": grounding_outcome(
            expected,
            grounding,
        ),
        "pilot_split": split,
    }


def test_main_pattern_uses_three_main_validators():
    row = make_row(
        "disjointness",
        shacl=True,
        owlrl=True,
        owl=True,
        expected=True,
        grounding=True,
    )

    assert main_pattern(row) == "shacl+owl+grounding"


def test_summary_keeps_grounding_false_signals_separate():
    rows = [
        make_row(
            "domain_range",
            shacl=True,
            expected=False,
            grounding=True,
        ),
        make_row(
            "grounding",
            expected=True,
            grounding=False,
        ),
    ]

    for row in rows:
        row["observed_main_pattern"] = main_pattern(row)

    summary = summarize_rows(rows)

    assert summary["n"] == 2
    assert summary["raw_shacl_detected"] == 1
    assert summary["grounding_detected"] == 1
    assert summary["grounding_matches_expected"] == 0
    assert summary["grounding_outcomes"] == {
        "true_positive": 0,
        "false_positive": 1,
        "true_negative": 0,
        "false_negative": 1,
    }


def test_summary_counts_observed_overlap():
    rows = [
        make_row(
            "disjointness",
            shacl=True,
            owl=True,
            expected=True,
            grounding=True,
        ),
        make_row(
            "cardinality",
            shacl=True,
            expected=False,
            grounding=False,
        ),
        make_row(
            "grounding",
            expected=True,
            grounding=True,
        ),
    ]

    for row in rows:
        row["observed_main_pattern"] = main_pattern(row)

    summary = summarize_rows(rows)

    assert summary["observed_overlap_main"] == {
        "grounding": 1,
        "shacl": 1,
        "shacl+owl+grounding": 1,
    }
