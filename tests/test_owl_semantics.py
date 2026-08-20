import pytest

from owlready2 import (
    AllDifferent,
    AllDisjoint,
    FunctionalProperty,
    ObjectProperty,
    OwlReadyInconsistentOntologyError,
    Thing,
    World,
    sync_reasoner,
)


def new_ontology(name):
    world = World()
    ontology = world.get_ontology(f"http://example.org/{name}.owl")
    return world, ontology


def test_domain_axiom_infers_subject_type():
    _, onto = new_ontology("domain")

    with onto:
        class Film(Thing):
            pass

        class Person(Thing):
            pass

        class directedBy(ObjectProperty):
            domain = [Film]
            range = [Person]

        subject = Thing("subject")
        director = Person("director")

        directedBy[subject] = [director]

    sync_reasoner([onto])

    assert Film in subject.is_a


def test_functional_property_with_two_fillers_is_consistent():
    _, onto = new_ontology("functional_consistent")

    with onto:
        class hasValue(ObjectProperty, FunctionalProperty):
            pass

        subject = Thing("subject")
        first = Thing("first")
        second = Thing("second")

        hasValue[subject] = [first, second]

    sync_reasoner([onto])


def test_functional_property_with_different_fillers_is_inconsistent():
    _, onto = new_ontology("functional_inconsistent")

    with onto:
        class hasValue(ObjectProperty, FunctionalProperty):
            pass

        subject = Thing("subject")
        first = Thing("first")
        second = Thing("second")

        hasValue[subject] = [first, second]

        AllDifferent([first, second])

    with pytest.raises(OwlReadyInconsistentOntologyError):
        sync_reasoner([onto])


def test_disjoint_class_membership_is_inconsistent():
    _, onto = new_ontology("disjoint")

    with onto:
        class Person(Thing):
            pass

        class Organization(Thing):
            pass

        AllDisjoint([Person, Organization])

        entity = Person("entity")
        entity.is_a.append(Organization)

    with pytest.raises(OwlReadyInconsistentOntologyError):
        sync_reasoner([onto])


def test_missing_minimum_cardinality_filler_is_consistent():
    _, onto = new_ontology("minimum_cardinality")

    with onto:
        class hasChild(ObjectProperty):
            pass

        class Parent(Thing):
            is_a = [hasChild.min(1, Thing)]

        Parent("parent")

    sync_reasoner([onto])