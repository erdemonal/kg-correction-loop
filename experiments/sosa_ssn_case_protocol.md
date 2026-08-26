# SOSA and SSN confirmatory case construction protocol

## Purpose

This stage converts the 180 source units fixed in the confirmatory sampling
record into paired clean and single fault graph records. It does not run the
extractor, a repair model, the grounding assessor, SHACL validation, or OWL
reasoning. Expected symbolic outcomes are design time assertions that must be
tested in the next validation stage. They are not reported results.

The extension uses the vendored SOSA/SSN 2023 Edition draft snapshot at commit
`37fa55298187464b41c3712620dcbf5bd438b1b2`. Project specific constraints are
kept in `validation/shapes/sosa_ssn_confirmatory.ttl` and are described as an
application profile, not as normative W3C axioms.

## Fixed sample

The input is the committed selection of 180 source units: 30 independent source
units per condition, comprising 168 USGS daily value records and all 12
selected W3C examples. A source unit is assigned to exactly one condition.
Selection did not use model outputs, validator outputs, human labels, or
preliminary Movie and Music outcomes.

## Three graph layers

Each case keeps three concepts separate:

1. `clean_content_triples` are the record derived claims shown to the model and
   used for exact clean reference comparison and grounding measurement.
2. `scaffold_triples` provide explicit project profile targeting or RDF support.
   They are supplied to symbolic validators but excluded from model visible
   content, grounding judgments, and clean reference comparison.
3. `primary_modification` records the grouped controlled fault as exact added
   and removed triple sets. Applying that transformation to the clean content
   must reproduce `injected_content_triples` exactly.

The clean graph is a minimal controlled projection of each source record. It is
not represented as a complete graph of every fact in the source.

## USGS adapter

Each USGS daily value becomes one `sosa:Observation` inside one record derived
`sosa:ObservationCollection`. The content graph records the monitoring
location, observed property, decimal result, unit, date, result time, and the
closed one day collection interval. A project class,
`kcl:USGSDailyObservation`, is added only as scaffold so that the SHACL profile
can require exactly one feature, property, result, phenomenon time, and result
time without treating this project rule as a W3C rule.

The model visible source text consists of the captured USGS sentence followed
by a deterministic adapter sentence that names the observation, collection,
feature, and property represented in the graph. The suffix is generated from
the same locked record. It is not an LLM annotation.

## W3C example registry

The twelve W3C examples use an explicit minimal graph registry. This registry
is an auditable projection of the pinned examples, not an independently
annotated natural language benchmark. Direct `FeatureOfInterest` types needed
only for raw SHACL checking are stored as scaffold and excluded from grounding
and clean reference evaluation.

## Controlled transformations

| Condition | Single grouped transformation | Expected injected behavior |
|---|---|---|
| Disjointness | Add `SampleCollection` type to an `ObservationCollection`. | SHACL violation and OWL inconsistency |
| Functional property conflict | Add a second distinct `hasSimpleResult`, or a second `hasResult` object plus explicit `owl:differentFrom`. | SHACL violation and OWL inconsistency |
| Domain and range | Replace a valid relation endpoint with an explicitly incompatible entity. | SHACL violation. OWL remains consistent under open world semantics |
| Cardinality | Remove a required `observedProperty` assertion. | SHACL violation. OWL remains consistent |
| Temporal | Move a collection member's phenomenon time outside the recorded collection interval. | SHACL violation. OWL remains consistent |
| Grounding | Replace a supported result value with a different structurally valid value absent from the source record. | No planned SHACL or OWL violation. Grounding failure only |

For the object valued functional case, the added result node and its
`owl:differentFrom` assertion are one fault transformation. The explicit
difference assertion is necessary because OWL does not assume that two names
denote different individuals.

The temporal profile does not impose a universal ordering between
`phenomenonTime` and `resultTime`. It checks only the controlled, source backed
rule that a member observation's phenomenon time lies within its collection's
recorded interval.

## Reproducibility and stopping point

All inputs and profile artifacts are SHA-256 locked. Case construction is
deterministic and fails on changed inputs, duplicate cases, source reuse,
condition count drift, invalid relations, or a transformation that does not
exactly reproduce the injected graph.

After this stage passes the full test suite, the next package will perform
symbolic validation of the clean and injected graphs and record observed SHACL
and OWL outcomes. Only after those outcomes match the preregistered
expectations will prompts, the model runner, and analysis be finalized for the
single audit of the prepared commit.
