import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def test_repair_spec_records_runner_semantics():
    payload = json.loads(
        (EXPERIMENTS / "repair_spec.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["version"] == 4
    assert payload["output"]["invalid_output_retry"] is False
    assert payload["feedback"]["fields"] == [
        "validator",
        "violation_id",
        "error_type",
        "focus",
        "path",
        "message",
    ]
    assert payload["grounding_cache"][
        "seed_from_frozen_clean_and_injected_results"
    ] is True
    assert payload["runner"]["preflight_only_mode"] is True
    assert payload["grounding_cache"]["shared_assertion_precedence"] == "injected_state"
    assert payload["grounding_cache"]["clean_background_exclusion_source"] == "clean_state"
    assert payload["grounding_cache"]["rerun_to_resolve_frozen_disagreement"] is False


def test_protocol_records_output_failure_and_no_feedback():
    text = (
        EXPERIMENTS / "repair_protocol.md"
    ).read_text(encoding="utf-8")

    assert "## Runner behavior" in text
    assert "The model is not asked to regenerate" in text
    assert "stops with `no_feedback`" in text
