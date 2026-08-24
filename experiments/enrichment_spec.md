# Ontology enrichment specification

## Source ontologies

The controlled study uses the Movie and Music ontologies from Text2KGBench commit `50a3d255371b8817cdff70fd88459ac82b339cfe`.

Movie ontology:

`data/wikidata_tekgen/ontologies/owl/ont_1_movie.ttl`

Music ontology:

`data/wikidata_tekgen/ontologies/owl/ont_2_music.ttl`

The original Text2KGBench ontologies are not modified. Additional OWL axioms and SHACL shapes used in the experiments are stored separately.

The extraction prompts and JSON ontology descriptions from Text2KGBench also remain unchanged. The enrichment is used only during validation and controlled case construction.

For readability, classes and properties are referred to by their Text2KGBench identifiers and labels in this document.

## Controlled cases

Each controlled case starts from a graph that passes the relevant validation checks. One deliberate modification is then applied to create the test condition.

A graph that passes the validation checks is not necessarily a complete representation of every fact in the source sentence. It only needs to satisfy the conditions required for the controlled experiment.

One modification may be detected by more than one validator. This is expected and is recorded as part of the comparison.

Statements produced by the extraction model are kept separate from assertions added for the experiment. Additional assertions may include RDF types and OWL restrictions required by the symbolic validators.

These additional assertions are recorded with their provenance. They are not presented to the grounding assessor as statements produced by the extraction model.

Temporal statements added from explicit information in the source text are treated as content statements and are included in grounding assessment.

## Disjointness

### Movie

The enrichment declares:

`Human (Q5)` disjoint with `Film production company (Q1762059)`.

The existing `production company (P272)` property has `Film production company` as its range.

A test case uses an entity known to be a `Human` as the object of `production company`. OWL then infers that the same entity is also a `Film production company`, which conflicts with the disjointness axiom.

The SHACL constraint checks that a `Human` is not used as the object of `production company`.

### Music

The enrichment declares:

`Human (Q5)` disjoint with `Musical work (Q2188189)`.

The existing `performer (P175)` property has `Musical work` as its domain.

A test case uses an entity known to be a `Human` as the subject of `performer`. OWL then infers that the same entity is also a `Musical work`, which conflicts with the disjointness axiom.

The SHACL constraint checks that a `Human` is not used as the subject of `performer`.

These modifications may also introduce a grounding error if the added relation is not supported by the source text. Such overlap is recorded rather than removed.

No disjointness axiom is added between `City` and `Country` in Movie or between `Single` and `Album` in Music. Adding these axioms would change the intended behavior of the domain and range cases.

## Domain and range

The pinned ontologies already contain the OWL domain and range axioms needed for these cases. No new OWL domain or range axioms are added.

SHACL is used to express the same intended type requirements as explicit graph constraints.

### Movie

The property `narrative location (P840)` has `City (Q515)` as its range.

The SHACL shape checks every object of `narrative location` and requires it to be a `City`.

The controlled modification uses an object explicitly typed as `Country (Q6256)`.

Because `City` and `Country` are not declared disjoint, OWL remains consistent. The OWL range axiom simply allows the reasoner to infer that the object is also a `City`.

Pilot case `ont_1_movie_test_767` is a natural example of this pattern, but it is not automatically used as a controlled case.

### Music

The property `record label (P264)` has `Album (Q482994)` as its domain.

The SHACL shape checks every subject of `record label` and requires it to be an `Album`.

The controlled modification uses a subject explicitly typed as `Single (Q134556)`.

`Single` is a subclass of `Musical work`, but it is not disjoint with `Album`. As a result, OWL remains consistent and may infer that the subject is also an `Album`.

Pilot case `ont_2_music_test_230` is a natural example of this pattern, but it is not automatically used as a controlled case.

SHACL is evaluated on the original graph and in a supplementary condition with pySHACL OWL RL inference enabled. The ontology is supplied during validation.

OWL RL inference may add a type implied by an OWL domain or range axiom. As a result, a SHACL violation in the original graph may disappear when this inference option is enabled. The two conditions are kept separate.

## Cardinality

Cardinality constraints are applied only to the selected controlled case. They are not asserted as global requirements for all films or musical works.

### Movie

The property used for this condition is `director (P57)`.

The selected graph must initially contain exactly one director statement supported by the source text.

The SHACL shape targets only the selected film and requires at least one value for `director`.

The OWL version assigns the selected individual a minimum cardinality restriction of one for `director`.

The modification removes the only director statement.

SHACL reports a violation because the required value is missing.

OWL remains consistent because the open world assumption allows the required director to exist without being explicitly stated in the graph.

The grounding assessor cannot detect the controlled deletion itself because it checks support for assertions that are present. It does not check extraction completeness. Any grounding findings on other baseline assertions are kept as background results.

### Music

The property used for this condition is `composer (P86)`.

The selected graph must initially contain exactly one composer statement supported by the source text.

The SHACL and OWL constraints follow the same structure as the Movie case. SHACL requires at least one composer value for the selected individual, and OWL assigns a minimum cardinality restriction of one.

The modification removes the only composer statement.

The expected symbolic behavior is the same as in the Movie case. SHACL reports a violation and OWL remains consistent. The grounding assessor cannot detect the omission itself. Any other grounding findings are kept as background results.

## Temporal ordering

The pinned Movie and Music ontologies do not contain properties for the pairs of temporal events needed in this condition.

The controlled study introduces a small set of temporal properties for statements that are explicitly supported by the source text.

The temporal constraints compare two dates with SHACL SPARQL. No arbitrary historical cutoff is used.

### Movie

The primary candidate is `ont_1_movie_test_467`.

The source states that the film opened at the Toronto International Film Festival on 15 September 2007 and was released in Canada on 18 April 2008.

The controlled representation records a premiere date and a theatrical release date, with the requirement that the premiere must not occur after the theatrical release.

The modification exchanges the two dates.

`ont_1_movie_test_235` is retained as an alternative example because it also states a premiere before a later theatrical release.

### Music

The primary candidate is `ont_2_music_test_215`.

The source states that the song premiered on US radio on 11 November 2011 and became available for digital download on 14 November 2011.

The controlled representation records a radio premiere date and a digital release date, with the requirement that the radio premiere must not occur after the digital release.

The modification exchanges the two dates.

`ont_2_music_test_120` is retained as an alternative example because it states a recording date before a later release.

For these cases, SHACL SPARQL reports the temporal violation.

The OWL model contains no axiom that orders the two date values. In the current HermiT environment, `xsd:date` assertions are removed only from the copy sent to HermiT because the reasoner does not support that datatype in this setup. As a result, the OWL part of the study does not examine the date assertions or their order.

The grounding assessor also reports an error because exchanging the dates makes the temporal statements inconsistent with the source text.

This overlap is an expected result of the experiment.

## Grounding

Grounding cases contain a relation that is valid according to the symbolic schema but is not supported by the source text.

### Movie

The preferred property is `director (P57)`.

The added object is typed as `Human`, so the range requirement of the property is satisfied.

The source text does not support that person as a director of the film.

In this case, the graph passes the symbolic checks, but the grounding assessor reports the unsupported statement.

### Music

The preferred property is `composer (P86)`.

The added object is typed as `Human`, so the range requirement of the property is satisfied.

The source text does not support that person as a composer of the musical work.

In this case, the graph passes the symbolic checks, but the grounding assessor reports the unsupported statement.

## Functional property conflicts

`functional_property_conflict` remains part of error taxonomy version 1 and remains covered by the semantic sanity tests.

It is not instantiated in the controlled Movie and Music cases.

The dataset audit did not identify a property in either ontology that could reasonably be treated as functional.

In particular, `publication date (P577)` has multiple legitimate values in the benchmark and is declared as an `owl:ObjectProperty` in both pinned ontologies.

Adding a new functional property only to create this error condition would introduce an artificial modeling assumption that is not supported by the selected ontologies.

The study keeps the error type in the taxonomy but reports it as not instantiated for the Movie and Music domains.