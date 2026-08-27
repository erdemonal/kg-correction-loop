import json
from collections import Counter
from pathlib import Path

from rdflib import Graph

from src.build_sosa_ssn_confirmatory_cases import (
    DEFAULT_SPEC,
    apply_modification,
    case_data_graph,
    case_shapes_graph,
    profile_graph,
    read_json,
    read_jsonl,
    sha256_file,
    triple_key,
    validate_cases,
)


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "experiments" / "sosa_ssn_confirmatory_cases.jsonl"
MANIFEST_PATH = ROOT / "experiments" / "sosa_ssn_case_manifest.json"


def inputs():
    spec = read_json(DEFAULT_SPEC)
    cases = read_jsonl(CASES_PATH)
    return spec, cases


def test_case_inputs_and_application_profile_are_hash_locked():
    spec = read_json(DEFAULT_SPEC)

    for name in ("selection", "source_units", "sampling_manifest"):
        assert sha256_file(ROOT / spec["inputs"][name]) == (
            spec["inputs"][f"{name}_sha256"]
        )
    assert sha256_file(ROOT / spec["profile"]["path"]) == spec["profile"]["sha256"]
    assert sha256_file(ROOT / spec["w3c_registry"]["path"]) == spec["w3c_registry"]["sha256"]
    assert spec["ontology"]["pinned_commit"] == "37fa55298187464b41c3712620dcbf5bd438b1b2"
    assert spec["profile"]["layer"] == "project SHACL application profile, not W3C SOSA and SSN axioms"


def test_profile_is_valid_turtle_and_contains_the_fixed_constraint_families():
    spec = read_json(DEFAULT_SPEC)
    graph = profile_graph(spec)
    text = (ROOT / spec["profile"]["path"]).read_text(encoding="utf-8")

    assert isinstance(graph, Graph)
    assert len(graph) > 0
    for token in (
        "ObservationCollectionDisjointnessShape",
        "SimpleResultFunctionalShape",
        "ResultFunctionalShape",
        "USGSDailyObservationShape",
        "CollectionMemberIntervalShape",
    ):
        assert token in text
    assert "PREFIX sosa:" in text
    assert "PREFIX kcl:" in text


def test_cases_are_180_unique_units_with_thirty_per_condition():
    spec, cases = inputs()
    validate_cases(cases, spec)

    assert len(cases) == 180
    assert len({case["case_id"] for case in cases}) == 180
    assert len({case["source_unit_id"] for case in cases}) == 180
    assert Counter(case["condition"] for case in cases) == {
        condition: 30 for condition in spec["condition_constructors"]
    }
    assert Counter(case["source_family"] for case in cases) == {
        "usgs_daily": 168,
        "w3c_examples": 12,
    }


def test_every_injected_graph_is_exactly_the_recorded_transformation():
    _spec, cases = inputs()

    for case in cases:
        rebuilt = apply_modification(
            case["clean_content_triples"], case["primary_modification"]
        )
        assert rebuilt == case["injected_content_triples"]
        clean = {triple_key(row) for row in case["clean_content_triples"]}
        added = {triple_key(row) for row in case["primary_modification"]["added"]}
        removed = {triple_key(row) for row in case["primary_modification"]["removed"]}
        assert not added & clean
        assert removed <= clean


def test_scaffold_is_excluded_from_content_reference_and_grounding_scopes():
    _spec, cases = inputs()

    assert sum(bool(case["scaffold_triples"]) for case in cases) == 171
    for case in cases:
        content = {triple_key(row) for row in case["clean_content_triples"]}
        scaffold = {triple_key(row) for row in case["scaffold_triples"]}
        assert not content & scaffold
        assert case["grounding_scope"] == "content_triples_only"
        assert case["clean_reference_scope"] == "clean_content_triples_only"


def test_usgs_cases_have_record_derived_graphs_and_profile_scaffold():
    _spec, cases = inputs()
    usgs = [case for case in cases if case["source_family"] == "usgs_daily"]

    assert len(usgs) == 168
    for case in usgs:
        predicates = Counter(row["predicate"] for row in case["clean_content_triples"])
        assert predicates["hasMember"] == 1
        assert predicates["hasSimpleResult"] == 1
        assert predicates["collectionStart"] == 1
        assert predicates["collectionEnd"] == 1
        assert case["scaffold_triples"] == [
            next(
                row
                for row in case["scaffold_triples"]
                if row["predicate"] == "type" and row["object"] == "USGSDailyObservation"
            )
        ]
        assert "In the record derived SOSA representation" in case["source_text"]


def test_disjointness_and_functional_injections_encode_owl_faults():
    _spec, cases = inputs()
    disjoint = [case for case in cases if case["condition"] == "disjointness"]
    functional = [
        case for case in cases if case["condition"] == "functional_property_conflict"
    ]

    for case in disjoint:
        assert case["primary_modification"]["removed"] == []
        assert any(
            row["predicate"] == "type" and row["object"] == "SampleCollection"
            for row in case["primary_modification"]["added"]
        )
        assert case["expected_symbolic"]["injected"] == {
            "raw_shacl": False,
            "owl_consistent": False,
        }

    for case in functional:
        added = case["primary_modification"]["added"]
        assert any(row["predicate"] in {"hasSimpleResult", "hasResult"} for row in added)
        if len(added) > 1:
            assert case["source_unit_id"] == "w3c_tree_height"
            assert any(row["predicate"] == "differentFrom" for row in added)
        assert case["expected_symbolic"]["injected"]["owl_consistent"] is False


def test_domain_cardinality_temporal_and_grounding_faults_are_separate():
    _spec, cases = inputs()
    by_condition = {
        condition: [case for case in cases if case["condition"] == condition]
        for condition in {"domain_range", "cardinality", "temporal", "grounding"}
    }

    for case in by_condition["domain_range"]:
        change = case["primary_modification"]
        assert change["operation"] == "replace"
        assert len(change["added"]) == len(change["removed"]) == 1
        assert change["added"][0]["predicate"] == change["removed"][0]["predicate"]
        assert case["expected_symbolic"]["injected"] == {
            "raw_shacl": False,
            "owl_consistent": True,
        }

    for case in by_condition["cardinality"]:
        change = case["primary_modification"]
        assert change["operation"] == "remove"
        assert change["added"] == []
        assert change["removed"][0]["predicate"] == "observedProperty"

    for case in by_condition["temporal"]:
        change = case["primary_modification"]
        assert change["operation"] == "replace"
        assert change["added"][0]["predicate"] == "phenomenonTime"
        assert change["removed"][0]["predicate"] == "phenomenonTime"
        assert change["added"][0]["object"] != change["removed"][0]["object"]

    for case in by_condition["grounding"]:
        change = case["primary_modification"]
        assert change["operation"] == "replace"
        assert change["added"][0]["predicate"] in {"hasSimpleResult", "resultValue"}
        assert change["added"][0]["predicate"] == change["removed"][0]["predicate"]
        assert case["expected_symbolic"]["injected"] == {
            "raw_shacl": True,
            "owl_consistent": True,
        }


def test_case_graph_helpers_keep_scaffold_out_of_content_but_in_symbolic_data():
    _spec, cases = inputs()
    case = next(case for case in cases if case["source_family"] == "usgs_daily")

    clean_graph = case_data_graph(case, "clean")
    injected_graph = case_data_graph(case, "injected")
    constraint_graph = case_shapes_graph(case)

    assert len(clean_graph) == len(case["clean_content_triples"]) + len(case["scaffold_triples"])
    assert len(injected_graph) == len(case["injected_content_triples"]) + len(case["scaffold_triples"])
    assert isinstance(constraint_graph, Graph)


def test_manifest_records_deterministic_construction_without_execution():
    spec, cases = inputs()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["case_spec_sha256"] == sha256_file(DEFAULT_SPEC)
    assert manifest["cases"]["sha256"] == sha256_file(CASES_PATH)
    assert manifest["cases"]["count"] == len(cases) == 180
    assert manifest["cases"]["by_source_family"] == {
        "usgs_daily": 168,
        "w3c_examples": 12,
    }
    assert not any(manifest["execution"].values())


def test_execution_guard_prohibits_every_experimental_component():
    spec = read_json(DEFAULT_SPEC)

    assert not any(spec["execution"].values())
    protocol = (ROOT / "experiments" / "sosa_ssn_case_protocol.md").read_text(
        encoding="utf-8"
    )
    assert "not reported results" in protocol
    assert "not as normative W3C axioms" in protocol
    assert "not a language model annotation" in protocol
