import os
import tempfile

from owlready2 import ObjectProperty, Thing, World, sync_reasoner
from pyshacl import validate
from rdflib import Graph


ONTO_IRI = "http://example.org/asymmetry.owl"
NS = f"{ONTO_IRI}#"

DOMAIN_SHAPE = f"""
@prefix ex: <{NS}> .
@prefix sh: <http://www.w3.org/ns/shacl#> .

ex:DirectorDomainShape
    a sh:NodeShape ;
    sh:targetSubjectsOf ex:directorOf ;
    sh:class ex:Person .
"""


def test_domain_mismatch_flagged_by_shacl_but_inferred_by_owl():
    world = World()
    onto = world.get_ontology(ONTO_IRI)

    with onto:
        class Film(Thing):
            pass

        class Person(Thing):
            pass

        class directorOf(ObjectProperty):
            domain = [Person]

        film1 = Film("film1")
        film2 = Film("film2")
        directorOf[film1] = [film2]

    fd, path = tempfile.mkstemp(suffix=".nt")
    os.close(fd)

    onto.save(file=path, format="ntriples")
    raw_graph = Graph().parse(path, format="nt")
    os.remove(path)

    conforms, _, _ = validate(
        raw_graph,
        shacl_graph=Graph().parse(data=DOMAIN_SHAPE, format="turtle"),
        inference="none",
    )

    assert conforms is False

    sync_reasoner([onto])

    assert Person in film1.is_a