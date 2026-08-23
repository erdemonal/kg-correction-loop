from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDFS, XSD

from src.validate_controlled_symbolic import (
    git_blob_sha1,
    hermit_compatible_graph,
)


def test_git_blob_sha1_matches_git_blob_format():
    data = b"hello\n"

    assert git_blob_sha1(data) == (
        "ce013625030ba8dba906f756967f9e9ca394464a"
    )



def test_hermit_compatible_graph_removes_xsd_date_only():
    graph = Graph()
    subject = URIRef("http://example.org/subject")
    property_uri = URIRef("http://example.org/date")
    other_property = URIRef("http://example.org/name")

    graph.add((property_uri, RDFS.range, XSD.date))
    graph.add(
        (
            subject,
            property_uri,
            Literal("2007-09-15", datatype=XSD.date),
        )
    )
    graph.add(
        (
            subject,
            other_property,
            Literal("example"),
        )
    )

    sanitized = hermit_compatible_graph(graph)

    assert (property_uri, RDFS.range, XSD.date) not in sanitized
    assert (
        subject,
        property_uri,
        Literal("2007-09-15", datatype=XSD.date),
    ) not in sanitized
    assert (
        subject,
        other_property,
        Literal("example"),
    ) in sanitized
