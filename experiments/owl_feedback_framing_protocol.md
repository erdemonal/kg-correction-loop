# RQ3 OWL feedback framing protocol

## Purpose

RQ3 measures how the amount of information in OWL inconsistency feedback
affects the first repair produced from the same controlled graph.

This experiment is separate from the completed RQ2 correction loop.
RQ2 trajectories, prompts, outcomes, analysis, and reporting remain locked.
RQ3 does not reinterpret or replace any RQ2 result.

## Cases and paired design

The experiment uses exactly the 10 controlled disjointness cases in
`experiments/controlled_selection.json`: five Movie cases and five Music
cases. Each case is evaluated once under each of three feedback conditions.
The analysis unit is the controlled case. Comparisons across conditions
are paired by case.

Every condition starts from the same injected content graph at round 0. No
repair output is passed to another condition. The three requests for a case
differ only in the structured OWL feedback item.

The 30 requests use a fixed rotating condition order so that the same
condition is not always the first request for every case. That order is
part of the locked specification and is recorded in the run metadata.

## Repair model and prompt

The repair model and generation settings are the same as in RQ2:

- model: `llama3.1:8b-instruct-q4_K_M`
- temperature: 0
- seed: 42
- num_ctx: 4096
- num_predict: 2048

The exact RQ2 repair prompt in `experiments/repair_prompt.txt` is reused
without modification. Its SHA256 is pinned in the RQ3 specification. Each
request contains the source sentence, allowed relations, injected content
graph, and one structured OWL feedback item. It does not contain the clean
reference graph, human adjudication, auxiliary RDF types, OWL axioms, SHACL
shapes, or expected repair.

The output must be the complete repaired content graph, with exactly one
`relation(subject, object)` statement per line. Explanatory prose, empty
outputs, truncated outputs, and relations outside the allowed set are output
failures. There is no retry after an invalid output.

## Feedback conditions

All conditions use the same feedback schema, validator identity, stable
violation identity, and null path and error type fields. The condition label
is not shown to the model.

### Verdict

The focus is null and the message is:

`The graph is logically inconsistent.`

### Verdict plus location

The message is unchanged. The focus contains the controlled entity involved
in the injected inconsistency.

### Verdict plus explanation

The focus is the same as in the location condition. The message says that the
focus entity is classified under two disjoint classes. For Movie it names
person and production company. For Music it names person and musical work.

The explanation comes from the controlled construction, not from a newly requested HermiT explanation. It does not identify a triple to remove, add,
or replace, and it does not reveal the clean reference graph.

## Single repair step

Each pairing of a case with a condition receives exactly one repair generation. There is no iterative feedback loop, stopping rule, or second
repair request. The design isolates the first repair step. RQ2 already
describes repair across later rounds: 41 of its 43 targets that were ever
resolved were first resolved in round 1.

There are exactly 10 cases times 3 conditions, for 30 repair generations.

## Hidden measurement after repair

SHACL, OWL, and grounding measurements are never included in the repair
prompt and cannot influence generation. Before generation, the runner verifies
once per case that the injected graph has the expected OWL inconsistency and
that all injected assertions have stored grounding judgments. The repaired
graph is measured only after that single output has been produced.

Raw SHACL and OWL validation reconstruct the same context used by RQ2:
clean background types, case shapes, pinned source ontology, controlled
enrichment, and the injected case OWL context. Only content statements are
replaced by the repaired graph.

Grounding measurement uses the v3 assessor from the locked RQ1 setup. Stored
clean and injected judgments seed one measurement cache per controlled case.
After all three repair outputs for that case exist, the union of novel
assertions is judged in sorted order, each unique assertion at most once.
The resulting judgment is then reused across conditions. This cache is used
only for measurement and cannot affect a repair request. The run always
contains 30 generations from the repair model. It may also contain extra
calls to the v3 grounding assessor for novel assertions.

## Outcomes

The following are recorded for each pairing of a case with a condition:

- controlled target removed: the injected primary triple is absent
- OWL consistency after repair
- exact recovery of the clean controlled reference graph
- collateral statements removed or added relative to the clean graph
- raw SHACL findings and new SHACL identities relative to the injected graph
- grounding findings and new grounding identities relative to the injected
  graph
- an unresolved OWL inconsistency after target removal
- output failure category
- statements added to and removed from the injected graph
- symmetric graph edit distance from the injected and clean graphs

Controlled target removal and OWL consistency remain separate. A model may
make the graph consistent by deleting a statement that the source supports
while leaving the injected target triple in place.

HermiT provides only a global consistency verdict in this experiment. If the
target triple is removed but the repaired graph remains inconsistent, the
result is reported as an inconsistency after target removal. It is not called a
new localized OWL violation, because this experiment has no reasoner
explanation that would identify that inconsistency.

For an unusable model output, target removal and reference recovery for that
request are false. Validation and edit measures that depend on a parsed graph
are null.

## Analysis boundary

Counts and paired transitions at the case level are primary. With 10 cases,
the experiment is descriptive. It does not claim that one framing statistically
dominates another, and it does not estimate population performance. Any paired
test or interval included later is exploratory and must keep the controlled
case as the unit.

## Runner and provenance

The runner is `src/run_owl_feedback_framing.py`. Before generation it verifies
the locked prompt, controlled selection, RQ2 implementation dependencies,
model digests, grounding specification, and required local result files.

`--preflight-only` performs these checks without generation or validation.
The completed runner writes 30 JSONL rows and a metadata JSON file. Existing
outputs are never overwritten. Rendered prompts, prompt hashes, raw model
responses, parser results, validation measurements, execution order, input
hashes, and model identities are retained.

Transport failures abort the run and are not scored as model failures. A
completed run is never repeated or selectively regenerated in response to its
outcomes.
