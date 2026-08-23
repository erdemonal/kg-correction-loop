from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF


ROOT = Path(__file__).resolve().parents[1]
MOVIE_ENRICHMENT = ROOT / "validation" / "ontologies" / "movie_enrichment.ttl"
MUSIC_ENRICHMENT = ROOT / "validation" / "ontologies" / "music_enrichment.ttl"
MOVIE_SHAPES = ROOT / "validation" / "shapes" / "movie.ttl"
MUSIC_SHAPES = ROOT / "validation" / "shapes" / "music.ttl"

KCL = Namespace("https://github.com/erdemonal/kg-correction-loop#")
MOVIEC = Namespace("https://cenguix.github.io/Text2KGBench/ont_1_movie/concepts#")
MUSICC = Namespace("https://cenguix.github.io/Text2KGBench/ont_2_music/concepts#")


def graph_from_file(path: Path) -> Graph:
    return Graph().parse(path, format="turtle")


def graph_from_turtle(text: str) -> Graph:
    return Graph().parse(data=text, format="turtle")


def conforms(data: str, shapes_path: Path, *, inference="none", ontology=None) -> bool:
    result, _, _ = validate(
        data_graph=graph_from_turtle(data),
        shacl_graph=graph_from_file(shapes_path),
        ont_graph=ontology,
        inference=inference,
    )
    return result


def test_enrichment_files_parse_and_declare_expected_terms():
    movie = graph_from_file(MOVIE_ENRICHMENT)
    music = graph_from_file(MUSIC_ENRICHMENT)

    assert (MOVIEC.Q5, OWL.disjointWith, MOVIEC.Q1762059) in movie
    assert (MUSICC.Q5, OWL.disjointWith, MUSICC.Q2188189) in music

    assert (KCL.premiereDate, RDF.type, OWL.DatatypeProperty) in movie
    assert (KCL.theatricalReleaseDate, RDF.type, OWL.DatatypeProperty) in movie
    assert (KCL.radioPremiereDate, RDF.type, OWL.DatatypeProperty) in music
    assert (KCL.digitalReleaseDate, RDF.type, OWL.DatatypeProperty) in music
    assert (KCL.recordingDate, RDF.type, OWL.DatatypeProperty) in music
    assert (KCL.releaseDate, RDF.type, OWL.DatatypeProperty) in music


def test_movie_disjointness_shape_detects_human_production_company():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix moviec: <https://cenguix.github.io/Text2KGBench/ont_1_movie/concepts#> .
    @prefix movier: <https://cenguix.github.io/Text2KGBench/ont_1_movie/relations#> .

    ex:film movier:P272 ex:person .
    ex:person a moviec:Q5 .
    """

    assert conforms(data, MOVIE_SHAPES) is False


def test_music_disjointness_shape_detects_human_performer_subject():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix musicc: <https://cenguix.github.io/Text2KGBench/ont_2_music/concepts#> .
    @prefix musicr: <https://cenguix.github.io/Text2KGBench/ont_2_music/relations#> .

    ex:person a musicc:Q5 ;
        musicr:P175 ex:performer .
    """

    assert conforms(data, MUSIC_SHAPES) is False


def test_movie_domain_range_shape_detects_country_as_narrative_location():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix moviec: <https://cenguix.github.io/Text2KGBench/ont_1_movie/concepts#> .
    @prefix movier: <https://cenguix.github.io/Text2KGBench/ont_1_movie/relations#> .

    ex:film movier:P840 ex:place .
    ex:place a moviec:Q6256 .
    """

    assert conforms(data, MOVIE_SHAPES) is False


def test_music_domain_shape_detects_single_with_record_label():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix musicc: <https://cenguix.github.io/Text2KGBench/ont_2_music/concepts#> .
    @prefix musicr: <https://cenguix.github.io/Text2KGBench/ont_2_music/relations#> .

    ex:single a musicc:Q134556 ;
        musicr:P264 ex:label .
    """

    assert conforms(data, MUSIC_SHAPES) is False


def test_movie_domain_range_shape_passes_after_owlrl_range_materialization():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix moviec: <https://cenguix.github.io/Text2KGBench/ont_1_movie/concepts#> .
    @prefix movier: <https://cenguix.github.io/Text2KGBench/ont_1_movie/relations#> .

    ex:film movier:P840 ex:place .
    ex:place a moviec:Q6256 .
    """

    ontology = graph_from_turtle("""
    @prefix moviec: <https://cenguix.github.io/Text2KGBench/ont_1_movie/concepts#> .
    @prefix movier: <https://cenguix.github.io/Text2KGBench/ont_1_movie/relations#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    movier:P840 rdfs:range moviec:Q515 .
    """)

    assert conforms(data, MOVIE_SHAPES, inference="none", ontology=ontology) is False
    assert conforms(data, MOVIE_SHAPES, inference="owlrl", ontology=ontology) is True


def test_music_domain_shape_passes_after_owlrl_domain_materialization():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix musicc: <https://cenguix.github.io/Text2KGBench/ont_2_music/concepts#> .
    @prefix musicr: <https://cenguix.github.io/Text2KGBench/ont_2_music/relations#> .

    ex:single a musicc:Q134556 ;
        musicr:P264 ex:label .
    """

    ontology = graph_from_turtle("""
    @prefix musicc: <https://cenguix.github.io/Text2KGBench/ont_2_music/concepts#> .
    @prefix musicr: <https://cenguix.github.io/Text2KGBench/ont_2_music/relations#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    musicr:P264 rdfs:domain musicc:Q482994 .
    """)

    assert conforms(data, MUSIC_SHAPES, inference="none", ontology=ontology) is False
    assert conforms(data, MUSIC_SHAPES, inference="owlrl", ontology=ontology) is True


def test_movie_temporal_shape_detects_reversed_dates():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix kcl: <https://github.com/erdemonal/kg-correction-loop#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:film
        kcl:premiereDate "2008-04-18"^^xsd:date ;
        kcl:theatricalReleaseDate "2007-09-15"^^xsd:date .
    """

    assert conforms(data, MOVIE_SHAPES) is False


def test_music_temporal_shape_detects_reversed_dates():
    data = """
    @prefix ex: <http://example.org/> .
    @prefix kcl: <https://github.com/erdemonal/kg-correction-loop#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:song
        kcl:radioPremiereDate "2011-11-14"^^xsd:date ;
        kcl:digitalReleaseDate "2011-11-11"^^xsd:date .
    """

    assert conforms(data, MUSIC_SHAPES) is False


def test_temporal_shapes_accept_correct_order():
    movie_data = """
    @prefix ex: <http://example.org/> .
    @prefix kcl: <https://github.com/erdemonal/kg-correction-loop#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:film
        kcl:premiereDate "2007-09-15"^^xsd:date ;
        kcl:theatricalReleaseDate "2008-04-18"^^xsd:date .
    """

    music_data = """
    @prefix ex: <http://example.org/> .
    @prefix kcl: <https://github.com/erdemonal/kg-correction-loop#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:song
        kcl:radioPremiereDate "2011-11-11"^^xsd:date ;
        kcl:digitalReleaseDate "2011-11-14"^^xsd:date .
    """

    assert conforms(movie_data, MOVIE_SHAPES) is True
    assert conforms(music_data, MUSIC_SHAPES) is True
