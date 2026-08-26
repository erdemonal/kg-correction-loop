# SOSA and SSN confirmatory extension: semantic design protocol

## Status and purpose

This document fixes the semantic design of the SOSA and SSN extension before any
new extraction, validation, grounding, or repair generation is run. It does not
authorize an experimental run. The source pool, eligible case counts, final
sample size, prompts, runner, and run manifest must be fixed in later commits
and accepted in a single audit of that prepared state.

The completed Movie and Music experiments remain a locked preliminary study.
Their cases, prompts, trajectories, results, and reporting logic are not
modified. They are not pooled with this extension.

## Ontology edition and snapshot

The extension uses the SOSA/SSN 2023 Edition as its experimental ontology. It
does not use the 2017 Recommendation. The 2023 Edition is a W3C and OGC First
Public Working Draft. This study uses a pinned draft snapshot. The edition is
work in progress. It must not be called a W3C Recommendation.

The ontology source is pinned to commit
`37fa55298187464b41c3712620dcbf5bd438b1b2` of the official
`w3c/sdw-sosa-ssn` repository. Every ontology module used by the experiment is
identified by its SHA-256 digest in
`experiments/sosa_ssn_axiom_inventory.json`. Later preparation code must copy
the required modules into the experiment workspace and verify the same hashes
before constructing a case. A moving namespace response or a later editor draft
must not replace the pinned snapshot.

The 2017 Recommendation is retained only as historical and compatibility
background. It is not an experimental ontology. It is not mixed with the 2023
snapshot.

## Three semantic layers

The extension keeps three layers separate in code, provenance, and reporting.

1. Pinned SOSA and SSN axioms. These are assertions and axioms present in the
   pinned W3C repository snapshot. They must not be described as project
   enrichments.
2. Project application profile. These are explicit SHACL or SHACL SPARQL
   rules added for the controlled study. Every rule must identify its source
   rationale. Every rule must be described as a project profile rather than an
   axiom of SOSA or SSN.
3. Controlled injections. These are one fault transformations applied to a
   clean graph. Injection metadata is automatic gold for the controlled fault
   only. It is not human ground truth for naturally occurring extraction errors.

No rule may migrate silently from one layer to another.

## Controlled error conditions

The confirmatory extension supports six conditions. Each case contains one
primary controlled fault and a matched clean graph.

### Disjointness

The pinned SSN graph declares `sosa:ActuationCollection`,
`sosa:ObservationCollection`, `sosa:SampleCollection`, and
`sosa:SamplingCollection` pairwise disjoint. A controlled case assigns one
collection individual to two of these classes. This is a base ontology OWL
inconsistency. It is not a disjointness axiom created by the project.

### Functional property conflict

The pinned SOSA graph declares `sosa:hasResult` and
`sosa:hasSimpleResult` functional. A controlled case adds a second genuinely
distinct result to one execution. Object valued conflicts require explicit
distinctness when OWL could otherwise identify the fillers. Literal conflicts
must use values that are distinct in the applicable datatype value space.

This condition is named `functional_property_conflict`. It is separate from a
missing value cardinality breach.

### Domain and range

The project application profile gives selected SOSA properties explicit SHACL
subject and object class constraints. A controlled case replaces one endpoint
with an entity of an incompatible profile class. SOSA `schema:domainIncludes`
and `schema:rangeIncludes` annotations must not be reported as RDFS domain
or range axioms. SSN `owl:allValuesFrom` restrictions may support OWL
inferences. Absence of an inconsistency is not interpreted as OWL
validation success.

### Cardinality

The project application profile identifies the minimum fields required for the
record derived graph pattern used in this study. A controlled case removes one
required value. SHACL evaluates the missing value under closed world
validation. OWL open world semantics do not license a claim that absence is
an inconsistency.

### Temporal

The temporal condition uses the SOSA and SSN collection rule. When a collection
states a temporal range or interval, each member's corresponding value must
match or fall within that range. The project application profile implements
this requirement as SHACL SPARQL over explicitly represented time values.

The extension must not impose a universal rule that `phenomenonTime` precedes
`resultTime`. SOSA explicitly permits forecasts whose phenomenon time is later
than their result time, as well as historical observations with a much earlier
phenomenon time.

### Grounding

A grounding injection adds or substitutes an assertion that is absent from the
source record rendered for the case. The controlled transformation supplies
the expected status for that assertion. Grounding assessor judgments remain
model judgments. They are never described as human gold labels.

## Source corpus design without human annotation

The extension does not restore the abandoned 300 case annotation study. It
does not create an annotation interface.

Source units will be derived from fixed public records and official W3C
examples. A source adapter must preserve the original record, its public source
identifier, retrieval metadata, license or reuse statement, and SHA-256 digest.
A deterministic renderer produces the source text. A deterministic graph
builder produces the clean SOSA and SSN graph from the same record. The
clean graph is therefore record derived controlled data. It is not a manually
annotated natural language gold standard.

The candidate pool should include multiple scenario families rather than many
copies of one template. The initial source families to test are:

- official examples in the pinned `w3c/sdw-sosa-ssn` snapshot
- fixed historical USGS water observation records
- fixed historical US EPA air quality records

The final source list is not locked by this document. A later source pool
manifest must report eligible counts, rejected records and reasons, duplicate
checks, scenario family counts, and all content hashes before sample size is
chosen.

## Sampling and experimental unit rules

No final sample size is fixed yet. Sample size is selected only after the
eligible pool is enumerated without model outcomes. The design target is about
30 independent controlled cases per error condition. The protocol must not
duplicate one source unit merely to reach that target.

The source unit is the experimental unit. Multiple triples or validation
findings from one source unit are not independent observations. If one clean
source unit supports more than one candidate injection, at most one injection
is selected for the primary confirmatory sample unless a grouped analysis is
specified before the run.

Preliminary Movie and Music cases are excluded from confirmatory selection.
Preliminary and confirmatory estimates are reported separately.

## Planned outcomes

The confirmatory study retains the existing distinctions among:

- controlled target detection and validator firing
- end to end target resolution, ever resolved, and last validated resolution
- exact clean reference recovery and controlled target removal
- valid graph regression and unusable model output
- collateral edits and new validation findings
- clean reference set comparison and source grounding
- recorded model duration, wall clock time, and monetary cost

Clean reference comparison is not human source faithfulness. Resampling
intervals describe sensitivity to the controlled case set. They are not
population confidence intervals.

## Required sequence before execution

1. Vendor and hash check the pinned SOSA and SSN modules.
2. Implement and test the explicit project SHACL application profile.
3. Build the source adapters and immutable raw source manifest.
4. Enumerate the eligible candidate pool without running the extractor,
   grounding assessor, validators on generated graphs, or repair model.
5. Fix the sample size and selection rule from the eligible pool.
6. Build the confirmatory cases, prompts, runner, analysis plan, and tests.
7. Commit and push the complete prepared state.
8. Obtain one audit of that commit. The audit checks semantic provenance,
   leakage, case independence, selection, denominators, and run guards.
9. Run only after an accepted verdict for that prepared state.

## Prohibited actions in this phase

- Do not run an extraction or repair model.
- Do not run or tune the grounding assessor.
- Do not alter the locked Movie and Music artifacts.
- Do not restore the 300 case annotation workflow.
- Do not call the 2023 Edition a W3C Recommendation.
- Do not fetch a moving ontology namespace at experiment time.
- Do not select cases or sample size using experimental outcomes.
