from rdflib import Graph
from pyshacl import validate


def run_validation(data_ttl: str, shapes_ttl: str) -> bool:
    data_graph = Graph().parse(data=data_ttl, format="turtle")
    shapes_graph = Graph().parse(data=shapes_ttl, format="turtle")

    conforms, _, _ = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="none",
    )

    return conforms


def test_min_count_detects_missing_property():
    data = """
    @prefix ex: <http://example.org/> .

    ex:alice a ex:Person .
    """

    shapes = """
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .

    ex:PersonShape
        a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [
            sh:path ex:birthDate ;
            sh:minCount 1 ;
        ] .
    """

    assert run_validation(data, shapes) is False


def test_max_count_detects_multiple_values():
    data = """
    @prefix ex: <http://example.org/> .

    ex:alice a ex:Person ;
        ex:birthDate "2000-01-01" ;
        ex:birthDate "2001-01-01" .
    """

    shapes = """
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .

    ex:PersonShape
        a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [
            sh:path ex:birthDate ;
            sh:maxCount 1 ;
        ] .
    """

    assert run_validation(data, shapes) is False


def test_datatype_detects_wrong_literal_type():
    data = """
    @prefix ex: <http://example.org/> .

    ex:alice a ex:Person ;
        ex:birthDate "not-a-date" .
    """

    shapes = """
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:PersonShape
        a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [
            sh:path ex:birthDate ;
            sh:datatype xsd:date ;
        ] .
    """

    assert run_validation(data, shapes) is False


def test_class_constraint_detects_wrong_object_type():
    data = """
    @prefix ex: <http://example.org/> .

    ex:film1 a ex:Film ;
        ex:director ex:company1 .

    ex:company1 a ex:Organization .
    """

    shapes = """
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .

    ex:FilmShape
        a sh:NodeShape ;
        sh:targetClass ex:Film ;
        sh:property [
            sh:path ex:director ;
            sh:class ex:Person ;
        ] .
    """

    assert run_validation(data, shapes) is False


def test_class_constraint_detects_untyped_object():
    data = """
    @prefix ex: <http://example.org/> .

    ex:film1 a ex:Film ;
        ex:director ex:unknown .
    """

    shapes = """
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .

    ex:FilmShape
        a sh:NodeShape ;
        sh:targetClass ex:Film ;
        sh:property [
            sh:path ex:director ;
            sh:class ex:Person ;
        ] .
    """

    assert run_validation(data, shapes) is False


def test_explicit_domain_shape_detects_wrong_subject_type():
    data = """
    @prefix ex: <http://example.org/> .

    ex:alice a ex:Person ;
        ex:director ex:bob .

    ex:bob a ex:Person .
    """

    shapes = """
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .

    ex:DirectorDomainShape
        a sh:NodeShape ;
        sh:targetSubjectsOf ex:director ;
        sh:class ex:Film .
    """

    assert run_validation(data, shapes) is False


def test_numeric_range_detects_out_of_range_value():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:song1 a ex:Song ;
        ex:releaseYear "1800"^^xsd:integer .
    """

    shapes = """
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:SongShape
        a sh:NodeShape ;
        sh:targetClass ex:Song ;
        sh:property [
            sh:path ex:releaseYear ;
            sh:datatype xsd:integer ;
            sh:minInclusive 1900 ;
        ] .
    """

    assert run_validation(data, shapes) is False


def test_valid_graph_conforms():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:alice a ex:Person ;
        ex:birthDate "2000-01-01"^^xsd:date .

    ex:film1 a ex:Film ;
        ex:director ex:alice .
    """

    shapes = """
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:PersonShape
        a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [
            sh:path ex:birthDate ;
            sh:minCount 1 ;
            sh:maxCount 1 ;
            sh:datatype xsd:date ;
        ] .

    ex:FilmShape
        a sh:NodeShape ;
        sh:targetClass ex:Film ;
        sh:property [
            sh:path ex:director ;
            sh:minCount 1 ;
            sh:class ex:Person ;
        ] .
    """

    assert run_validation(data, shapes) is True