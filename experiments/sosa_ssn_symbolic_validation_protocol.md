# SOSA and SSN symbolic oracle preflight protocol

## Purpose and stopping rule

This stage verifies the preregistered symbolic behavior of the 180 locked
confirmatory cases before any model generation or repair run. It executes raw
SHACL validation and OWL consistency reasoning only. The resulting records are
preflight evidence that the controlled faults and validator oracles behave as
specified. They are not confirmatory language model outcomes.

The stage passes only if all 180 clean graphs conform to the project SHACL
application profile and are OWL consistent, and every injected graph exactly
matches its condition specific expected pattern. A mismatch is a design error
and stops the pipeline. It must not be relabeled, excluded, or repaired after
seeing model outcomes.

## Locked inputs

The cases, case manifest, case specification, application profile, axiom
inventory, and all twelve vendored SOSA and SSN core modules are SHA-256
locked. The ontology is the SOSA/SSN 2023 Edition draft snapshot at W3C
repository commit `37fa55298187464b41c3712620dcbf5bd438b1b2`. It is described
as a pinned First Public Working Draft snapshot, not as a W3C Recommendation.

## SHACL oracle

Each clean and injected graph is validated independently with pySHACL,
inference set to `none`, against the project application profile plus any
case local constraint. The application profile is not presented as a set of
normative W3C axioms.

The injected violation must include the preregistered SHACL constraint
component for its condition:

| Condition | Required injected component |
|---|---|
| Disjointness | `sh:NotConstraintComponent` |
| Functional property conflict | `sh:MaxCountConstraintComponent` |
| Domain and range profile | `sh:ClassConstraintComponent` or `sh:OrConstraintComponent`, following the targeted endpoint shape |
| Cardinality | `sh:MinCountConstraintComponent` |
| Temporal collection interval | `sh:SPARQLConstraintComponent` |
| Grounding | no SHACL violation |

For each non grounding case, the injected report must contain at least one of
the components allowed for that condition. The `isSampleOf` subject profile is
expressed with `sh:or` over `sosa:Sample` and `sosa:MaterialSample`, so its
domain fault correctly yields `sh:OrConstraintComponent`. Other endpoint
profile faults yield `sh:ClassConstraintComponent`.

All clean reports and both grounding variants must conform.

## OWL oracle

OWL consistency is checked with HermiT through Owlready2 over the merged twelve
module snapshot and the case data graph. Required functional property and
collection disjointness axioms are asserted directly from the parsed vendored
graph before reasoning.

After the twelve hash locked modules are merged, `owl:imports` routing triples
are removed from HermiT input. Their targets are not additional experimental
inputs. They point to modules already present in the locked merge. Removing the
routing triples prevents Owlready2 from making nondeterministic live W3C
requests while preserving the parsed vendored axioms themselves.

For positive cases, the validator checks two unions: all clean graphs together
and all injected graphs expected to remain consistent together. OWL semantics
is monotonic, so consistency of a union entails consistency of every subgraph
included in that union. This supplies a stronger and much faster positive
check than 300 redundant reasoner launches.

Every injected case expected to be inconsistent is checked separately. There
are 60 such runs: 30 disjointness cases and 30 functional property conflict
cases. This prevents one inconsistent graph from masking another.

HermiT does not accept `xsd:date` in its OWL 2 datatype map. The preflight
therefore also removes triples whose object is the `xsd:date` datatype or an
`xsd:date` literal from HermiT input only, matching the existing project
compatibility policy. The tested OWL contradictions depend on disjoint classes
and functional properties, not on removed date triples. Raw SHACL still sees
the complete unsanitized graph.

## Outputs and excluded actions

The validator writes one normalized record per case plus a manifest containing
input hashes, denominator checks, environment versions, validation counts, and
the result file hash. Blank node identifiers and full SHACL report graphs are
not persisted because they are engine local and nondeterministic. Conformance,
violation counts, and normalized constraint component IRIs are retained.

This stage does not run an extractor, repair model, or grounding assessor. It
does not alter the preliminary Movie and Music results, select cases, change
sample size, or produce confirmatory model outcomes. After the complete test
suite is green, the next stage defines the grounding contract and confirmatory
runner.
