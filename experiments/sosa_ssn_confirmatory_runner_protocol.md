# SOSA and SSN confirmatory grounding and repair protocol

## Scope

This package prepares the locked grounding and iterative repair runners for the
180 case SOSA and SSN confirmatory study. It does not execute either model
before a commit level audit of the prepared state is accepted. The Movie and
Music results remain a separate preliminary study and are not pooled with these
cases.

The confirmatory sample contains 30 independent cases for each of six
conditions: disjointness, functional property conflict, domain and range,
cardinality, temporal interval, and grounding. Each source unit occurs once.

## Fixed models and prompts

Repair uses `llama3.1:8b-instruct-q4_K_M`, digest
`46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`,
with the same options and exact repair prompt used in the preliminary study.

Source grounding uses the locked version 3 assessor,
`qwen2.5:7b-instruct-q4_K_M`, digest
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`,
with the unchanged version 3 prompt and options. No SOSA specific prompt rules
are added after seeing the new cases. This preserves component identity across
domains. The assessor is measured as a validator component and is not treated
as human ground truth.

## Grounding run

The decision unit is one asserted content triple. Only `source_text` is used as
evidence. Scaffold triples, SHACL shapes, ontology axioms, the clean reference,
and controlled injection metadata are never shown to the assessor.

For each case, the union of clean and injected content assertions is judged
once. An assertion shared by both states reuses the same locked judgment. This
avoids redundant calls and prevents separate clean and injected calls from
creating two verdicts for an identical sentence and assertion pair. Clean and
injected case summaries are derived from that common cache.

The controlled grounding target is the set of triples added by the grouped
injection. Cardinality removes an assertion, so it has no asserted target for
grounding and its expected target detection is false. The other five
conditions add or replace at least one source unsupported assertion and have an
expected target grounding error. These expectations come from the controlled
transformation, not from the assessor's output. Any assessor mismatch remains a
confirmatory result. It does not trigger selective reruns, prompt edits, case
exclusions, or relabeling.

## Initial repair state and prompt

Round 0 is the injected content graph. The repair request contains only:

1. the locked source text
2. the allowed relation names
3. the current complete content graph
4. structured feedback from raw SHACL, OWL consistency, and locked grounding

The clean reference and scaffold are excluded. The model must return the
complete graph as one `relation(subject, object)` statement per line without
explanation. Empty, malformed, truncated, or out of vocabulary output is an
output failure and is not regenerated.

Object kinds are restored deterministically for symbolic validation. Values
already present in the clean or injected reference retain their recorded kind.
New values use predicate specific rules: entity valued relations remain
entities. Time relations require ISO date or datetime values. Boolean and
decimal simple results are recognized lexically. Result units are strings. An
invalid kind or class is an output failure, not a silently corrected answer.

## Feedback and validation

Raw SHACL uses the locked project application profile plus the case local
shape. OWL consistency uses HermiT over the twelve vendored modules, with live
`owl:imports` routing removed and the recorded `xsd:date` compatibility copy.

At round 0, SHACL and OWL feedback describe the controlled fault without
revealing the expected repair. Grounding feedback is emitted only for target
triples the locked assessor actually marks unsupported. A false negative can
therefore produce no grounding feedback and remains in the end to end
denominator.

After a repair, all current symbolic violations are actionable. New grounding
assertions are judged once and cached. Assertions marked unsupported in the
clean baseline remain recorded but are excluded from feedback, preventing
baseline assessor noise from dominating the controlled correction loop.

## Outcomes and stopping

Target resolution, validated state, and exact clean-reference recovery are
separate outcomes. For symbolic conditions, the round 0 SHACL violation
identities belonging to the condition's preregistered constraint component
family are locked as the controlled target. Target resolution means none of
those identities remains. A changed violation identity is recorded separately
as a new violation and can keep the graph unvalidated without rewriting the
original target outcome. For grounding, target resolution means the controlled
added replacement triple or triples are absent. Validated state means no
actionable feedback remains. Exact recovery means equality with the controlled
clean content graph.

The maximum is five repair rounds. A case stops as validated, stalled,
oscillating, at maximum rounds, with no initial feedback, or with an output
failure. The graph and violation identity state jointly define stalled and
oscillation. Transport failures abort execution and are not scored as model
output failures.

## Resume, cost, and audit gate

Both runners write only complete case rows and resume at case boundaries.
Existing rows are checked against hash, spec, and commit. Duplicate case IDs
are forbidden, and the fixed case order cannot change. Model token counts,
duration fields, rendered repair prompts, prompt hashes, raw responses,
grounding judgments, and validator outcomes are retained for later cost and
dynamics analysis.

Experimental execution is blocked while
`experiments/sosa_ssn_confirmatory_audit_gate.json` is pending. After an auditor
accepts one exact commit, only the gate and audit register may change before
execution. The runners verify that boundary. Offline preflight and unit tests
never contact Ollama and remain allowed while the gate is pending.
