from src.grounding_judge import (
    EXPECTED_DIGEST,
    JUDGE_VERSION,
    PROMPT_PATH,
)


def test_grounding_judge_v3_is_locked():
    assert JUDGE_VERSION == "v3"
    assert PROMPT_PATH.name == "grounding_judge_prompt_v3.txt"
    assert EXPECTED_DIGEST == (
        "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"
    )


def test_v3_prompt_contains_final_performer_clarification():
    text = PROMPT_PATH.read_text(encoding="utf-8")

    assert '"X is a song/single by Y" directly supports performer(X, Y)' in text
    assert 'Do not reject this relation merely because the verb "perform" is absent.' in text
    assert '"released by Y"' in text
    assert '"published by Y"' in text
    assert '"distributed by Y"' in text
