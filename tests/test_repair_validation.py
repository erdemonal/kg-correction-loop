from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, SH

from src.repair_validation import shacl_violation_identity


def make_result(path):
    graph = Graph()
    result = BNode()
    graph.add((result, RDF.type, SH.ValidationResult))
    graph.add((result, SH.sourceConstraintComponent, SH.MinCountConstraintComponent))
    graph.add((result, SH.focusNode, URIRef("https://example.org/focus")))
    graph.add((result, SH.resultPath, URIRef(path)))
    graph.add((result, SH.sourceShape, URIRef("https://example.org/shape")))
    return graph, result


def test_shacl_identity_changes_when_result_path_changes():
    first_graph, first_result = make_result("https://example.org/director")
    second_graph, second_result = make_result("https://example.org/composer")
    first = shacl_violation_identity(first_graph, first_result)
    second = shacl_violation_identity(second_graph, second_result)
    assert first["violation_id"] != second["violation_id"]


def test_shacl_identity_keeps_missing_value_explicit():
    graph, result = make_result("https://example.org/director")
    identity = shacl_violation_identity(graph, result)["identity"]
    assert "value" in identity
    assert identity["value"] is None


def test_shacl_identity_changes_when_value_changes():
    first_graph, first_result = make_result("https://example.org/path")
    second_graph, second_result = make_result("https://example.org/path")
    first_graph.add((first_result, SH.value, Literal("A")))
    second_graph.add((second_result, SH.value, Literal("B")))
    first = shacl_violation_identity(first_graph, first_result)
    second = shacl_violation_identity(second_graph, second_result)
    assert first["violation_id"] != second["violation_id"]
