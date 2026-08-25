import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def payload():
    return json.loads(
        (
            EXPERIMENTS / "owl_feedback_framing_spec.json"
        ).read_text(encoding="utf-8")
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rq3_is_a_paired_one_step_30_generation_design():
    data = payload()
    design = data["design"]

    assert data["rq2_locked_commit"] == (
        "ef6f2874ca3dd5d551bd30649d64cc61c2f46705"
    )
    assert data["cases"]["error_type"] == "disjointness"
    assert data["cases"]["count"] == 10
    assert data["cases"]["movie_count"] == 5
    assert data["cases"]["music_count"] == 5
    assert design["paired_by_case"] is True
    assert design["conditions_per_case"] == 3
    assert design["repair_steps_per_condition"] == 1
    assert design["repair_generations"] == 30
    assert design["start_each_condition_from_injected_graph"] is True
    assert data["pre_run_audit"]["required"] is True
    assert data["runner"]["custom_result_path_allowed"] is False


def test_rq3_reuses_the_frozen_rq2_prompt_and_model():
    data = payload()
    prompt_path = ROOT / data["prompt"]["path"]

    assert data["prompt"]["reused_without_modification"] is True
    assert sha256(prompt_path) == data["prompt"]["sha256"]
    assert data["repair_model"]["name"] == (
        "llama3.1:8b-instruct-q4_K_M"
    )
    assert data["repair_model"]["options"] == {
        "temperature": 0,
        "seed": 42,
        "num_ctx": 4096,
        "num_predict": 2048,
    }


def test_only_owl_feedback_is_visible_to_the_repair_model():
    feedback = payload()["feedback"]
    measurement = payload()["post_repair_measurement"]

    assert feedback["model_receives_only_owl_feedback"] is True
    assert feedback["shacl_feedback_included"] is False
    assert feedback["grounding_feedback_included"] is False
    assert measurement["visible_to_repair_model"] is False
    assert measurement["ground_novel_assertions_after_all_three_repairs"] is True
    assert measurement[
        "judge_each_unique_novel_assertion_once_per_case"
    ] is True


def test_feedback_conditions_add_information_without_repair_instruction():
    feedback = payload()["feedback"]

    assert feedback["verdict"]["focus"] is None
    assert feedback["verdict"]["message"] == (
        "The graph is logically inconsistent."
    )
    assert feedback["location"]["message"] == (
        feedback["verdict"]["message"]
    )
    assert feedback["explanation"]["expected_repair_included"] is False
    assert feedback["explanation"]["repair_instruction_included"] is False

    for message in feedback["explanation"]["message_by_domain"].values():
        lowered = message.lower()
        assert "disjoint" in lowered
        assert not any(
            word in lowered
            for word in ("remove", "delete", "replace", "add the")
        )


def test_rq3_pins_frozen_rq2_dependencies_by_hash():
    for relative, expected in payload()["locked_dependencies"].items():
        assert sha256(ROOT / relative) == expected


def test_protocol_records_claim_and_measurement_boundaries():
    text = (
        EXPERIMENTS / "owl_feedback_framing_protocol.md"
    ).read_text(encoding="utf-8")

    assert (
        "RQ2 trajectories, prompts, outcomes, analysis, and "
        "reporting remain locked"
    ) in text
    assert "exactly one repair generation" in text
    assert "measurements are never included" in text
    assert "controlled construction" in text
    assert "newly requested HermiT explanation" in text
    assert "does not claim that one framing statistically" in text
    assert "outputs are never overwritten" in text
