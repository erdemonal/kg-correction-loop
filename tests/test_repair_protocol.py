import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

def payload():
    return json.loads((EXPERIMENTS / "repair_spec.json").read_text(encoding="utf-8"))

def test_repair_protocol_records_fixed_model_and_rounds():
    data = payload()
    assert data["repair_model"]["name"] == "llama3.1:8b-instruct-q4_K_M"
    assert data["repair_model"]["must_match_extraction_digest"] is True
    assert data["repair_model"]["options"] == {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 2048}
    assert data["rounds"]["max_repair_rounds"] == 5

def test_repair_prompt_is_frozen_and_hashed():
    data = payload()
    prompt = (EXPERIMENTS / "repair_prompt.txt").read_bytes()
    assert data["prompt"]["frozen_before_run"] is True
    assert data["prompt"]["benchmark_demonstration_included"] is False
    assert hashlib.sha256(prompt).hexdigest() == data["prompt"]["sha256"]
    assert data["prompt"]["sha256"] == "ac091682d5617765ca963a1541da8ee9ed459f3f78ea417bac4270e1297d36e8"

def test_repair_model_does_not_receive_symbolic_scaffolding():
    data = payload()["input"]
    assert data["source_sentence"] is True
    assert data["allowed_relations"] is True
    assert data["current_content_graph"] is True
    assert data["structured_feedback"] is True
    assert data["include_clean_reference_graph"] is False
    assert data["include_human_adjudication"] is False
    assert data["include_auxiliary_rdf_types"] is False
    assert data["include_owl_restrictions"] is False
    assert data["include_shacl_shapes"] is False

def test_main_feedback_uses_three_main_validators():
    data = payload()
    assert data["main_validators"] == ["raw_shacl", "owl_consistency", "grounding_v3"]
    assert data["supplementary_not_used_for_feedback"] == ["shacl_with_pyshacl_owl_rl_inference"]

def test_shacl_identity_includes_path_value_and_shape():
    assert payload()["shacl_identity"] == ["sourceConstraintComponent", "focusNode", "resultPath", "value", "sourceShape"]

def test_main_owl_feedback_has_no_explanation():
    owl = payload()["owl_feedback"]
    assert owl["main_message"] == "The graph is logically inconsistent."
    assert owl["controlled_focus_when_available"] is True
    assert owl["include_explanation"] is False
    assert owl["include_expected_repair"] is False

def test_validation_reconstruction_keeps_scaffolding_out_of_content():
    data = payload()["validation_reconstruction"]
    assert data["preserve_clean_background_types"] is True
    assert data["derive_new_background_types_from_repair"] is False
    assert data["restore_case_shapes"] is True
    assert data["restore_owl_context"] is True
    assert data["restore_cardinality_restriction"] is True
    assert data["preserve_temporal_hermit_date_handling"] is True
    assert data["grounding_content_only"] is True

def test_required_denominators_are_separate():
    outcomes = payload()["outcomes"]
    assert "end_to_end_target_resolution" in outcomes
    assert "target_resolution_given_feedback" in outcomes

def test_background_grounding_noise_is_not_sent_as_feedback():
    feedback = payload()["feedback"]
    assert feedback["initial"] == "controlled primary modification only"
    assert feedback["exclude_clean_baseline_grounding_noise"] is True
    assert feedback["use_frozen_grounding_output_without_human_correction"] is True


def test_reference_recovery_remains_separate_from_validator_success():
    data = payload()["reference_recovery"]
    assert data["definition"] == "final content triple set equals clean reference content triple set"
    assert "not complete recovery" in data["claim_boundary"]
