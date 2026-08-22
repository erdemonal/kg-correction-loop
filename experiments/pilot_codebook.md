# Pilot annotation guide

The pilot checks whether the current error taxonomy adequately describes errors observed in natural model outputs. It is not used to estimate error frequency. Labels are recorded at the case level. A taxonomy label appears at most once in a case, even if the same type of error occurs more than once.

Annotations were produced by a single annotator following this guide.

The source sentence is the only evidence used to assess grounding. The `triples` field is the extracted graph being annotated. The raw `response` is used to determine whether a problem was introduced during parsing. The `parsed_triples_raw` field is used only as a parsing diagnostic. Evaluator-filtered triples are not used.

For schema and logic labels, use the Movie and Music OWL ontologies from the pinned Text2KGBench version. Do not use axioms added later during ontology enrichment.

## Error classes

### `disjointness_violation`

Use this label only when the graph assigns an entity to classes that are explicitly declared disjoint in the ontology. Do not infer disjointness from common sense.

### `functional_property_conflict`

Use this label only when a property is explicitly functional and the graph gives it multiple fillers that are explicitly distinct. Multiple values alone are not sufficient.

### `domain_range_violation`

Use this label when an ontology relation is used with a subject or object that is clearly incompatible with its intended domain or range.

This is a schema error in the pilot. Do not describe an RDFS or OWL domain or range mismatch by itself as an OWL inconsistency, since domain and range axioms can infer types.

### `cardinality_breach`

Use this label only when the graph violates an explicit cardinality or required-property constraint. A fact that was mentioned in the sentence but omitted from the graph is not automatically a cardinality violation.

### `temporal_impossibility`

Use this label only when the graph violates an explicit constraint on temporal order. Do not assign it only because a date appears unusual or unlikely.

### `grounding_error`

Use this label when at least one extracted triple is not supported by the source sentence. This includes invented entities, relations, or facts, and relations whose meaning is not supported by the sentence.

Do not use outside knowledge to justify a triple that is not supported by the sentence. A grounding error may also receive a schema label when both problems are independently present.

## Errors outside the taxonomy

Set `uncovered` to `true` when there is a clear extraction or graph error that is not represented by the current taxonomy. Describe the error briefly in `notes`. Do not force an error into the closest existing class.

For example, use `uncovered` for an ontology vocabulary error such as a relation that does not exist in the ontology, unless a later revision of the taxonomy introduces a specific class for it.

Treat an omission as uncovered only when the source explicitly gives multiple fillers for the same ontology relation and the model extracts at least one of those fillers but omits another. Do not mark missing facts that would require a relation absent from the ontology or a different semantic mapping.

## Parsing problems

Set `parse_issue` to `true` when the raw model response cannot be faithfully converted into triples by the parser.

The flag records a conversion problem and does not assign sole responsibility to the parser. Ambiguous entity formatting or explanatory model output can also cause the conversion to produce unintended triples.

Do not assign a semantic error label when the apparent semantic error was created only by parsing. Other unaffected triples in the same case may still receive semantic labels.

## Completing a case

Set `annotated` to `true` only after the full case has been reviewed.

A case with no observed error has an empty `labels` list, `uncovered` set to `false`, and `parse_issue` set to `false`.
