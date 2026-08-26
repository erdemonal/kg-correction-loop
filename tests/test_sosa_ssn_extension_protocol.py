import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
PINNED_COMMIT = "37fa55298187464b41c3712620dcbf5bd438b1b2"


def load_json(name):
    return json.loads((EXPERIMENTS / name).read_text(encoding="utf-8"))


def test_extension_uses_pinned_2023_edition_not_2017():
    spec = load_json("sosa_ssn_extension_spec.json")

    assert spec["ontology"]["edition"] == "SOSA/SSN 2023 Edition"
    assert spec["ontology"]["pinned_commit"] == PINNED_COMMIT
    assert "not a W3C Recommendation" in spec["ontology"]["status"]
    assert "not an experimental ontology" in spec["ontology"]["historical_2017_role"]


def test_six_conditions_and_semantic_layers_are_fixed():
    spec = load_json("sosa_ssn_extension_spec.json")

    assert spec["semantic_layers"] == [
        "pinned_sosa_ssn_axioms",
        "project_shacl_application_profile",
        "controlled_injection",
    ]
    assert spec["conditions"] == [
        "disjointness",
        "functional_property_conflict",
        "domain_range",
        "cardinality",
        "temporal",
        "grounding",
    ]


def test_functional_and_disjointness_conditions_are_base_axioms():
    inventory = load_json("sosa_ssn_axiom_inventory.json")
    axioms = {row["id"]: row for row in inventory["verified_base_axioms"]}

    assert inventory["source_commit"] == PINNED_COMMIT
    assert axioms["functional_has_result"]["object"] == "owl:FunctionalProperty"
    assert axioms["functional_has_simple_result"]["object"] == "owl:FunctionalProperty"
    assert axioms["disjoint_collection_classes"]["relation"] == (
        "pairwise owl:disjointWith"
    )
    assert len(axioms["disjoint_collection_classes"]["classes"]) == 4


def test_core_module_hashes_are_complete_sha256_values():
    inventory = load_json("sosa_ssn_axiom_inventory.json")
    hashes = inventory["core_module_sha256"]

    assert len(hashes) == 12
    assert all(path.startswith("ssn/rdf/ontology/core/") for path in hashes)
    assert all(len(digest) == 64 for digest in hashes.values())
    assert all(set(digest) <= set("0123456789abcdef") for digest in hashes.values())


def test_no_annotation_and_no_execution_in_design_phase():
    spec = load_json("sosa_ssn_extension_spec.json")

    assert spec["source_strategy"]["human_annotation"] is False
    assert spec["source_strategy"]["annotation_interface"] is False
    assert spec["sampling"]["final_sample_size_locked"] is False
    assert spec["sampling"]["preliminary_cases_excluded"] is True
    assert not any(spec["execution"].values())


def test_audit_gate_is_after_complete_pre_run_state_and_before_generation():
    spec = load_json("sosa_ssn_extension_spec.json")
    gate = spec["audit_gate"]

    assert gate["single_commit_based_pre_run_audit"] is True
    assert "source pool manifest" in gate["audit_after"]
    assert "prompt and runner tests" in gate["audit_after"]
    assert "any confirmatory model generation" in gate["audit_before"]


def test_protocol_prohibits_invalid_claims_and_preserves_preliminary_study():
    text = (EXPERIMENTS / "sosa_ssn_extension_protocol.md").read_text(
        encoding="utf-8"
    )

    required = [
        "does not use the 2017 Recommendation",
        "work in progress",
        "It must not be called a W3C Recommendation.",
        "No rule may migrate silently from one layer to another.",
        "must not impose a universal rule",
        "does not restore the abandoned 300 case annotation study",
        "The source unit is the experimental unit.",
        "Preliminary and confirmatory estimates are reported separately.",
        "Obtain one audit of that commit.",
        "Do not run an extraction or repair model.",
    ]

    for phrase in required:
        assert phrase in text
