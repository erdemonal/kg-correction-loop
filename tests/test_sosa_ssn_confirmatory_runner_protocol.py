import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def spec():
    return json.loads(
        (EXPERIMENTS / "sosa_ssn_confirmatory_runner_spec.json").read_text(
            encoding="utf-8"
        )
    )


def test_protocol_keeps_preliminary_and_confirmatory_results_separate():
    payload = spec()
    assert payload["sample"]["preliminary_results_pooled"] is False
    assert payload["sample"]["cases"] == 180
    assert payload["sample"]["cases_per_condition"] == 30


def test_reporting_scope_discloses_source_concentration_and_limits_claims():
    payload = json.loads(
        (EXPERIMENTS / "sosa_ssn_confirmatory_reporting_scope.json").read_text(
            encoding="utf-8"
        )
    )
    composition = payload["locked_sample_composition"]
    assert composition["usgs_daily"] == 168
    assert composition["w3c_examples"] == 12
    assert composition["epa_airdata"] == 0
    assert composition["usgs_distinct_monitoring_locations"] == 168
    assert composition["w3c_examples_are_a_balanced_statistical_stratum"] is False
    assert (
        payload["project_context"][
            "establishes_generalization_across_all_sosa_and_ssn_deployments"
        ]
        is False
    )
    prohibited = " ".join(payload["prohibited_claims"])
    assert "representative of all SOSA and SSN deployments" in prohibited
    assert "cross-domain generalization" in prohibited
    assert "EPA data are represented" in prohibited


def test_grounding_contract_reuses_one_frozen_component_without_human_ground_truth():
    payload = spec()["grounding"]
    assert payload["judge_union_once_per_case"] is True
    assert payload["shared_clean_injected_assertion_rejudged"] is False
    assert payload["symbolic_scaffold_visible"] is False
    assert "Do not relabel" in payload["assessor_mismatch_policy"]


def test_repair_contract_separates_target_validation_and_reference_outcomes():
    payload = spec()["repair"]
    assert payload["clean_reference_visible_to_model"] is False
    assert payload["scaffold_visible_to_model"] is False
    assert payload["invalid_output_retry"] is False
    assert payload["max_rounds"] == 5
    assert payload["target_resolution"]["separate_from_validated_state"] is True
    assert payload["target_resolution"]["separate_from_clean_reference_recovery"] is True


def test_runner_contract_has_case_level_resume_and_no_partial_rows():
    payload = spec()["runner"]
    assert payload["case_level_resume"] is True
    assert payload["duplicate_case_rows_forbidden"] is True
    assert payload["partial_case_rows_written"] is False
    assert payload["full_run_requires_accepted_audit_gate"] is True


def test_before_audit_scope_is_offline_only():
    scope = spec()["execution_scope_before_audit"]
    assert scope == {
        "model_generation": False,
        "grounding_assessment": False,
        "repair": False,
        "offline_preflight_only": True,
    }


def test_protocol_records_offline_ontology_and_reasoner_compatibility():
    text = (
        EXPERIMENTS / "sosa_ssn_confirmatory_runner_protocol.md"
    ).read_text(encoding="utf-8")
    assert "twelve vendored modules" in text
    assert "`owl:imports` routing removed" in text
    assert "`xsd:date` compatibility" in text
    assert "not treated" in text
    assert "as human ground truth" in text
    assert "Target resolution, validated state, and exact clean-reference recovery" in text


def test_audit_request_requires_exact_commit_and_explicit_verdict():
    text = (EXPERIMENTS / "sosa_ssn_pre_run_audit_request.md").read_text(
        encoding="utf-8"
    )
    assert "exact commit" in text
    assert "A — accepted for confirmatory execution" in text
    assert "B — revision required before execution" in text
    assert "C — design invalid" in text
    assert "Do not audit an uncommitted" in text
    assert "168 USGS and 12" in text
    assert "prohibited generalization claims" in text
