# Ontology enrichment specification

## Source ontologies

Pinned Text2KGBench commit and Movie/Music ontology paths.
The original ontologies remain unchanged.

## Controlled cases

Definition of a clean graph.
One deliberate primary modification per case.
Auxiliary assertions are recorded separately from extracted triples.

## Disjointness

Movie class pair.
Music class pair.
OWL and SHACL constraints.
Expected validator behavior.

## Domain and range

Movie property.
Music property.
SHACL constraints for the existing domain and range definitions.
Expected behavior for raw SHACL, OWL, SHACL after materialization, and grounding assessment.

## Cardinality

Movie director property.
Music composer property.
Constraint scoped to the selected case.
Single filler requirement.
Expected validator behavior.

## Temporal ordering

Supported temporal patterns.
Candidate cases.
Modification by reversing the two dates.
Expected overlap between SHACL and grounding assessment.

## Grounding

Unsupported triple that does not otherwise violate the SHACL or OWL constraints.
Expected SHACL and OWL acceptance with a grounding violation.

## Functional property conflicts

Retained in the taxonomy and semantic sanity checks.
Not instantiated in the controlled Movie/Music cases because neither
selected ontology provides a defensible functional property without
introducing an additional artificial modeling assumption.