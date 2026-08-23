import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADJUDICATION = (
    ROOT / "experiments" / "controlled_grounding_adjudication.json"
)


def load():
    return json.loads(ADJUDICATION.read_text(encoding="utf-8"))


def test_six_target_mismatches_are_frozen_and_unique():
    payload = load()
    cases = payload["cases"]

    assert len(cases) == 6
    assert len({row["id"] for row in cases}) == 6
    assert payload["policy"]["judge_or_prompt_changed"] is False
    assert payload["policy"]["controlled_cases_changed"] is False
    assert payload["policy"]["judgments_rerun"] is False


def test_adjudication_counts_are_five_fp_one_fn():
    cases = load()["cases"]

    assert sum(
        row["adjudication"] == "false_positive"
        for row in cases
    ) == 5
    assert sum(
        row["adjudication"] == "false_negative"
        for row in cases
    ) == 1


def test_all_false_positives_are_movie_domain_range():
    cases = load()["cases"]

    false_positives = [
        row for row in cases
        if row["adjudication"] == "false_positive"
    ]

    assert all(
        row["domain"] == "movie"
        and row["condition"] == "domain_range"
        and row["expected_grounding_error"] is False
        and row["observed_grounding_error"] is True
        for row in false_positives
    )


def test_false_negative_is_music_252():
    cases = load()["cases"]

    false_negatives = [
        row for row in cases
        if row["adjudication"] == "false_negative"
    ]

    assert [row["id"] for row in false_negatives] == [
        "ont_2_music_test_252"
    ]
    assert false_negatives[0]["expected_grounding_error"] is True
    assert false_negatives[0]["observed_grounding_error"] is False
