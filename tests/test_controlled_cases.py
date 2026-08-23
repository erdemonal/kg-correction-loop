from pyshacl import validate
from rdflib.namespace import OWL, RDF

from src.controlled_cases import (
    KCL,
    ControlledCase,
    Statement,
    add_background_type,
    add_min_cardinality_restriction,
    case_from_baseline_row,
    case_to_graph,
    class_uri,
    entity_uri,
    inject_addition,
    inject_cardinality_omission,
    inject_temporal_swap,
    min_count_shape,
    relation_uri,
)


def shacl_conforms(data_graph, shapes_graph):
    conforms, _, _ = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="none",
    )
    return conforms


def movie_117():
    return case_from_baseline_row(
        {
            "id": "ont_1_movie_test_117",
            "domain": "movie",
            "sent": (
                "Toy Story Toons: Hawaiian Vacation is a 2011 Pixar "
                "computer animated short film directed by Gary Rydstrom."
            ),
            "triples": [
                [
                    "Toy Story Toons: Hawaiian Vacation",
                    "director",
                    "Gary Rydstrom",
                ]
            ],
        }
    )


def music_27():
    return case_from_baseline_row(
        {
            "id": "ont_2_music_test_27",
            "domain": "music",
            "sent": (
                "West Side Story is a musical with a book by Arthur Laurents, "
                "music by Leonard Bernstein and lyrics by Stephen Sondheim."
            ),
            "triples": [
                ["West Side Story", "composer", "Leonard Bernstein"],
                ["West Side Story", "lyrics_by", "Stephen Sondheim"],
            ],
        }
    )


def test_movie_types_are_derived_from_the_pinned_property_semantics():
    case = movie_117().with_derived_types()

    assert any(
        item.entity == "Toy Story Toons: Hawaiian Vacation"
        and item.class_id == "Q11424"
        and item.provenance == "domain:director"
        for item in case.background_types
    )
    assert any(
        item.entity == "Gary Rydstrom"
        and item.class_id == "Q5"
        and item.provenance == "range:director"
        for item in case.background_types
    )


def test_music_types_are_derived_from_the_pinned_property_semantics():
    case = music_27().with_derived_types()

    assert any(
        item.entity == "West Side Story"
        and item.class_id == "Q2188189"
        for item in case.background_types
    )
    assert any(
        item.entity == "Leonard Bernstein"
        and item.class_id == "Q5"
        for item in case.background_types
    )
    assert any(
        item.entity == "Stephen Sondheim"
        and item.class_id == "Q5"
        for item in case.background_types
    )


def test_grounding_payload_excludes_background_types():
    case = movie_117().with_derived_types()
    payload = case.grounding_payload()

    assert payload["triples"] == [
        [
            "Toy Story Toons: Hawaiian Vacation",
            "director",
            "Gary Rydstrom",
        ]
    ]
    assert "background_types" not in payload


def test_cardinality_shape_passes_before_and_fails_after_movie_omission():
    clean = movie_117().with_derived_types()
    subject = "Toy Story Toons: Hawaiian Vacation"
    focus = entity_uri(clean.case_id, subject)
    director = relation_uri("movie", "director")
    shapes = min_count_shape(focus, director)

    assert shacl_conforms(case_to_graph(clean), shapes) is True

    injected = inject_cardinality_omission(
        clean,
        subject,
        "director",
    )

    assert shacl_conforms(case_to_graph(injected), shapes) is False
    assert injected.primary_modification.error_type == "cardinality_breach"
    assert injected.primary_modification.operation == "remove"


def test_cardinality_shape_passes_before_and_fails_after_music_omission():
    clean = music_27().with_derived_types()
    subject = "West Side Story"
    focus = entity_uri(clean.case_id, subject)
    composer = relation_uri("music", "composer")
    shapes = min_count_shape(focus, composer)

    assert shacl_conforms(case_to_graph(clean), shapes) is True

    injected = inject_cardinality_omission(
        clean,
        subject,
        "composer",
    )

    assert shacl_conforms(case_to_graph(injected), shapes) is False


def test_owl_minimum_cardinality_context_is_added_as_auxiliary_structure():
    case = movie_117().with_derived_types()
    graph = case_to_graph(case)
    focus = entity_uri(case.case_id, "Toy Story Toons: Hawaiian Vacation")
    director = relation_uri("movie", "director")

    restriction = add_min_cardinality_restriction(
        graph,
        focus,
        director,
    )

    assert (focus, RDF.type, restriction) in graph
    assert (restriction, RDF.type, OWL.Restriction) in graph
    assert (restriction, OWL.onProperty, director) in graph


def test_manual_background_type_records_provenance():
    case = add_background_type(
        movie_117(),
        "South Africa",
        "Q6256",
        "manual:source_text",
    )

    assert case.background_types[0].entity == "South Africa"
    assert case.background_types[0].class_id == "Q6256"
    assert case.background_types[0].provenance == "manual:source_text"


def test_addition_records_one_primary_modification():
    clean = movie_117().with_derived_types()

    injected = inject_addition(
        clean,
        Statement(
            "Toy Story Toons: Hawaiian Vacation",
            "production_company",
            "Gary Rydstrom",
        ),
        "disjointness_violation",
    )

    assert injected.primary_modification.error_type == "disjointness_violation"
    assert injected.primary_modification.operation == "add"


def test_second_primary_modification_is_rejected():
    clean = movie_117().with_derived_types()
    injected = inject_addition(
        clean,
        Statement(
            "Toy Story Toons: Hawaiian Vacation",
            "production_company",
            "Gary Rydstrom",
        ),
        "disjointness_violation",
    )

    try:
        inject_addition(
            injected,
            Statement(
                "Toy Story Toons: Hawaiian Vacation",
                "director",
                "Someone Else",
            ),
            "grounding_error",
        )
    except ValueError as exc:
        assert "primary modification has already been applied" in str(exc)
    else:
        raise AssertionError("A second primary modification was accepted")


def test_temporal_swap_exchanges_values_and_records_one_modification():
    clean = ControlledCase(
        case_id="ont_1_movie_test_467",
        domain="movie",
        source_text=(
            "Emotional Arithmetic opened at the Toronto International Film "
            "Festival on September 15, 2007, and was released in Canada on "
            "April 18, 2008."
        ),
        content=(
            Statement(
                "Emotional Arithmetic",
                "premiereDate",
                "2007-09-15",
                "date",
            ),
            Statement(
                "Emotional Arithmetic",
                "theatricalReleaseDate",
                "2008-04-18",
                "date",
            ),
        ),
    )

    injected = inject_temporal_swap(
        clean,
        "Emotional Arithmetic",
        "premiereDate",
        "theatricalReleaseDate",
    )

    assert injected.content == (
        Statement(
            "Emotional Arithmetic",
            "premiereDate",
            "2008-04-18",
            "date",
        ),
        Statement(
            "Emotional Arithmetic",
            "theatricalReleaseDate",
            "2007-09-15",
            "date",
        ),
    )
    assert injected.primary_modification.error_type == "temporal_impossibility"
    assert injected.primary_modification.operation == "swap"


def test_unknown_relation_is_rejected_when_loading_a_baseline_case():
    row = {
        "id": "bad",
        "domain": "movie",
        "sent": "Example.",
        "triples": [["film", "animation_director", "person"]],
    }

    try:
        case_from_baseline_row(row)
    except ValueError as exc:
        assert "not in the pinned movie ontology" in str(exc)
    else:
        raise AssertionError("Unknown relation was accepted")


def test_entity_uris_are_stable_and_case_scoped():
    first = entity_uri("case-a", "Gary Rydstrom")
    second = entity_uri("case-a", "Gary Rydstrom")
    other_case = entity_uri("case-b", "Gary Rydstrom")

    assert first == second
    assert first != other_case


def test_background_types_are_present_in_the_rdf_graph():
    case = movie_117().with_derived_types()
    graph = case_to_graph(case)

    person = entity_uri(case.case_id, "Gary Rydstrom")
    human = class_uri("movie", "Q5")

    assert (person, RDF.type, human) in graph
