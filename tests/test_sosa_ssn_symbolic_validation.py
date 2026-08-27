import json
from collections import Counter
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, SH

from src.build_sosa_ssn_confirmatory_cases import read_json, read_jsonl, sha256_file
from src.validate_sosa_ssn_symbolic import (
    DEFAULT_SPEC,
    ROOT,
    assert_required_axioms,
    hermit_compatible_graph,
    load_ontology,
    repository_path,
)


def inputs():
    spec = read_json(DEFAULT_SPEC)
    results = read_jsonl(repository_path(spec["outputs"]["results"]))
    manifest = read_json(repository_path(spec["outputs"]["manifest"]))
    return spec, results, manifest


def test_symbolic_validation_inputs_are_hash_locked():
    spec = read_json(DEFAULT_SPEC)
    for name in (
        "cases",
        "case_manifest",
        "case_spec",
        "application_profile",
        "axiom_inventory",
    ):
        assert sha256_file(ROOT / spec["inputs"][name]) == spec["inputs"][f"{name}_sha256"]


def test_all_twelve_ontology_modules_are_hash_locked():
    spec = read_json(DEFAULT_SPEC)
    root = ROOT / spec["ontology"]["vendored_core_root"]
    assert {path.name for path in root.glob("*.ttl")} == set(spec["ontology"]["module_sha256"])
    for name, digest in spec["ontology"]["module_sha256"].items():
        assert sha256_file(root / name) == digest


def test_parsed_ontology_contains_required_real_owl_axioms():
    graph = load_ontology(read_json(DEFAULT_SPEC))
    assert_required_axioms(graph)
    sosa = "http://www.w3.org/ns/sosa/"
    assert (URIRef(sosa + "hasResult"), RDF.type, OWL.FunctionalProperty) in graph
    assert (URIRef(sosa + "hasSimpleResult"), RDF.type, OWL.FunctionalProperty) in graph
    assert (
        URIRef(sosa + "ObservationCollection"),
        OWL.disjointWith,
        URIRef(sosa + "SampleCollection"),
    ) in graph


def test_execution_scope_runs_only_symbolic_preflight_components():
    scope = read_json(DEFAULT_SPEC)["execution_scope"]
    assert scope["runs_shacl"] is True
    assert scope["runs_reasoner"] is True
    assert scope["runs_extractor"] is False
    assert scope["runs_repair_model"] is False
    assert scope["runs_grounding_assessor"] is False
    assert scope["modifies_preliminary_results"] is False
    assert scope["confirmatory_model_outcomes"] is False


def test_hermit_input_strips_import_routing_after_vendored_merge():
    graph = Graph()
    subject = URIRef("https://example.org/ontology")
    imported = URIRef("https://www.w3.org/ns/sosa/common")
    retained = (URIRef("https://example.org/A"), RDF.type, OWL.Class)
    graph.add((subject, OWL.imports, imported))
    graph.add(retained)

    compatible, removed = hermit_compatible_graph(graph)

    assert (subject, OWL.imports, imported) not in compatible
    assert retained in compatible
    assert removed == {"owl_imports": 1, "xsd_date": 0}


def test_preflight_results_cover_180_cases_with_fixed_denominators():
    _spec, results, manifest = inputs()
    assert len(results) == 180
    assert len({row["case_id"] for row in results}) == 180
    assert Counter(row["condition"] for row in results) == {
        "cardinality": 30,
        "disjointness": 30,
        "domain_range": 30,
        "functional_property_conflict": 30,
        "grounding": 30,
        "temporal": 30,
    }
    assert manifest["results"]["case_count"] == 180
    assert manifest["results"]["mismatch_case_count"] == 0


def test_every_clean_graph_passes_both_symbolic_oracles():
    _spec, results, _manifest = inputs()
    assert all(
        row["observed"]["clean"] == {"raw_shacl": True, "owl_consistent": True}
        for row in results
    )
    assert all(row["shacl_evidence"]["clean"]["result_count"] == 0 for row in results)


def test_injected_verdicts_exactly_match_preregistered_pattern():
    spec, results, _manifest = inputs()
    for row in results:
        assert row["observed"]["injected"] == spec["expected_symbolic"]["injected"][row["condition"]]
        assert row["mismatches"] == []


def test_injected_shacl_reports_contain_condition_specific_components():
    spec, results, _manifest = inputs()
    for row in results:
        allowed = set(spec["allowed_injected_shacl_components"][row["condition"]])
        evidence = row["shacl_evidence"]["injected"]
        if not allowed:
            assert evidence["conforms"] is True
            assert evidence["result_count"] == 0
            assert evidence["constraint_components"] == []
        else:
            assert evidence["conforms"] is False
            assert evidence["result_count"] >= 1
            assert allowed & set(evidence["constraint_components"])


def test_owl_negative_cases_were_reasoned_individually():
    _spec, results, manifest = inputs()
    negative = [
        row
        for row in results
        if row["condition"] in {"disjointness", "functional_property_conflict"}
    ]
    assert len(negative) == 60
    assert all(
        row["owl_evidence"]["injected"]["strategy"] == "individual_reasoner_run"
        for row in negative
    )
    assert all(row["observed"]["injected"]["owl_consistent"] is False for row in negative)
    assert manifest["owl_strategy"]["injected_negative_individual_case_count"] == 60


def test_owl_positive_union_strategy_is_recorded_and_logically_scoped():
    _spec, results, manifest = inputs()
    assert manifest["owl_strategy"]["clean_case_count"] == 180
    assert manifest["owl_strategy"]["injected_positive_case_count"] == 120
    for row in results:
        assert row["owl_evidence"]["clean"]["strategy"] == "consistent_union_entails_consistent_subgraph"
        if row["observed"]["injected"]["owl_consistent"]:
            assert row["owl_evidence"]["injected"]["strategy"] == "consistent_union_entails_consistent_subgraph"


def test_results_and_spec_hashes_are_recorded_in_manifest():
    spec, _results, manifest = inputs()
    assert manifest["spec_sha256"] == sha256_file(DEFAULT_SPEC)
    assert manifest["results"]["sha256"] == sha256_file(repository_path(spec["outputs"]["results"]))
    assert manifest["ontology"]["pinned_commit"] == "37fa55298187464b41c3712620dcbf5bd438b1b2"
    assert manifest["ontology"]["parsed_triple_count"] > 0


def test_protocol_states_draft_status_monotonicity_and_date_sanitization():
    text = (ROOT / "experiments" / "sosa_ssn_symbolic_validation_protocol.md").read_text(encoding="utf-8")
    assert "not as a W3C Recommendation" in text
    assert "consistency of a union entails consistency of every subgraph" in text
    assert "nondeterministic live W3C" in text
    assert "xsd:date" in text
    assert "does not run an extractor, repair model, or grounding assessor" in text
