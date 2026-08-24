import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def test_repair_protocol_records_fixed_model_and_rounds():
    payload = json.loads(
        (EXPERIMENTS / "repair_spec.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["repair_model"]["name"] == (
        "llama3.1:8b-instruct-q4_K_M"
    )
    assert payload["repair_model"]["must_match_extraction_digest"] is True
    assert payload["repair_model"]["options"] == {
        "temperature": 0,
        "seed": 42,
        "num_ctx": 4096,
        "num_predict": 2048,
    }
    assert payload["rounds"]["max_repair_rounds"] == 5


def test_repair_model_does_not_receive_symbolic_scaffolding():
    payload = json.loads(
        (EXPERIMENTS / "repair_spec.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["input"]["original_text2kgbench_prompt"] is True
    assert payload["input"]["current_content_graph"] is True
    assert payload["input"]["structured_feedback"] is True
    assert payload["input"]["include_auxiliary_rdf_types"] is False
    assert payload["input"]["include_owl_restrictions"] is False
    assert payload["input"]["include_shacl_shapes"] is False


def test_main_feedback_uses_three_main_validators():
    payload = json.loads(
        (EXPERIMENTS / "repair_spec.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["main_validators"] == [
        "raw_shacl",
        "owl_consistency",
        "grounding_v3",
    ]
    assert payload["supplementary_not_used_for_feedback"] == [
        "shacl_with_pyshacl_owl_rl_inference"
    ]


def test_background_grounding_noise_is_not_sent_as_feedback():
    payload = json.loads(
        (EXPERIMENTS / "repair_spec.json").read_text(
            encoding="utf-8"
        )
    )

    feedback = payload["feedback"]

    assert feedback["initial"] == (
        "controlled primary modification only"
    )
    assert feedback[
        "exclude_clean_baseline_grounding_noise"
    ] is True
    assert feedback[
        "use_frozen_grounding_output_without_human_correction"
    ] is True


def test_reference_recovery_is_distinct_from_validator_success():
    payload = json.loads(
        (EXPERIMENTS / "repair_spec.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["reference_recovery"]["definition"] == (
        "final content triple set equals clean reference content triple set"
    )
    assert "not complete recovery" in (
        payload["reference_recovery"]["claim_boundary"]
    )


def test_protocol_does_not_use_internal_work_package_names():
    text = (EXPERIMENTS / "repair_protocol.md").read_text(
        encoding="utf-8"
    )

    forbidden = ["WP1", "WP2", "WP3", "wp1", "wp2", "wp3"]

    for token in forbidden:
        assert token not in text
