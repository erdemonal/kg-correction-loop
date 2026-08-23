from rdflib.namespace import RDF

from src.build_controlled_dataset import (
    build_all,
    build_case_pair,
    build_symbolic_artifacts,
    load_canonical_rows,
    load_selection,
)
from src.controlled_cases import class_uri, entity_uri


def selected_case(selection, case_id):
    for row in selection["cases"]:
        if row["id"] == case_id:
            return row

    raise AssertionError(f"Missing selected case: {case_id}")


def test_final_selection_builds_50_unique_pairs():
    selection = load_selection()
    bundles = build_all(selection=selection)

    ids = [
        bundle["selected"]["id"]
        for bundle in bundles
    ]

    assert len(ids) == 50
    assert len(set(ids)) == 50

    for bundle in bundles:
        assert (
            bundle["injected"].primary_modification
            is not None
        )


def test_movie_domain_range_keeps_country_without_adding_city():
    selection = load_selection()
    canonical = load_canonical_rows()
    selected = selected_case(
        selection,
        "ont_1_movie_test_767",
    )

    clean, injected = build_case_pair(
        selected,
        canonical,
    )
    _, injected_graph, _, _ = build_symbolic_artifacts(
        selected,
        clean,
        injected,
    )

    country = entity_uri(
        selected["id"],
        "South Africa",
    )

    assert (
        country,
        RDF.type,
        class_uri("movie", "Q6256"),
    ) in injected_graph

    assert (
        country,
        RDF.type,
        class_uri("movie", "Q515"),
    ) not in injected_graph


def test_music_domain_range_keeps_single_without_adding_album():
    selection = load_selection()
    canonical = load_canonical_rows()
    selected = selected_case(
        selection,
        "ont_2_music_test_230",
    )

    clean, injected = build_case_pair(
        selected,
        canonical,
    )
    _, injected_graph, _, _ = build_symbolic_artifacts(
        selected,
        clean,
        injected,
    )

    work = entity_uri(
        selected["id"],
        "Let's Wait Awhile",
    )

    assert (
        work,
        RDF.type,
        class_uri("music", "Q134556"),
    ) in injected_graph

    assert (
        work,
        RDF.type,
        class_uri("music", "Q482994"),
    ) not in injected_graph


def test_temporal_case_contains_two_source_dates_and_swaps_them():
    selection = load_selection()
    canonical = load_canonical_rows()
    selected = selected_case(
        selection,
        "ont_1_movie_test_467",
    )

    clean, injected = build_case_pair(
        selected,
        canonical,
    )

    clean_values = {
        (item.predicate, item.object)
        for item in clean.content
    }
    injected_values = {
        (item.predicate, item.object)
        for item in injected.content
    }

    assert clean_values == {
        ("premiereDate", "2007-09-15"),
        ("theatricalReleaseDate", "2008-04-18"),
    }
    assert injected_values == {
        ("premiereDate", "2008-04-18"),
        ("theatricalReleaseDate", "2007-09-15"),
    }


def test_grounding_payload_does_not_expose_auxiliary_types():
    selection = load_selection()
    canonical = load_canonical_rows()
    selected = selected_case(
        selection,
        "ont_1_movie_test_767",
    )

    _, injected = build_case_pair(
        selected,
        canonical,
    )
    payload = injected.grounding_payload()

    assert "background_types" not in payload
    assert payload["triples"] == [
        [
            "Cry Freedom",
            "narrative_location",
            "South Africa",
        ]
    ]
