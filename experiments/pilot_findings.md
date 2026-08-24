# Pilot findings

The pilot contains 40 natural extraction cases, with 20 cases from the Movie domain and 20 from the Music domain. The sample was used to assess the error taxonomy and annotation guide, not to estimate population error rates.

A grounding error was observed in 23 cases and a domain/range violation in 7 cases. Five cases contained both labels. Fifteen cases received no taxonomy label. Six cases contained an error outside the current taxonomy, and five contained a parsing or conversion issue. Ten cases contained no taxonomy label, uncovered error, or parsing issue.

No disjointness violation, functional property conflict, cardinality breach, or temporal impossibility was observed in the pilot. This does not establish their natural frequency. These observations support the use of controlled error injection to study error types that are not represented in this small natural sample.

The six uncovered cases form two recurring patterns. Four contain ontology vocabulary misuse, such as a predicate that is absent from the ontology or a class name used as a relation. Two contain an omitted filler where the source explicitly gives several fillers for the same ontology relation and the model extracts only some of them.

These patterns are retained as pilot observations rather than added to the controlled error taxonomy. Ontology vocabulary membership can be checked directly and does not provide a meaningful distinction among the three validation paradigms. Source omission is primarily a completeness or recall problem and does not by itself make the extracted graph inconsistent or violate an explicit graph constraint.

Two types of conversion issue were observed. In two cases, commas inside entity surface forms made the relation argument syntax ambiguous. In three cases, the model repeated ontology relation signatures in explanatory text and the generic parser interpreted those signatures as extracted triples. The latter cases reflect an interaction between model output format and parsing rather than a parser error alone.

The controlled error taxonomy remains unchanged after the pilot. The uncovered cases and conversion issues remain recorded in the pilot annotations and should be considered when preparing clean graphs for controlled injection.