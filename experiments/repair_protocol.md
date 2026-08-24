# Repair experiment protocol

## Purpose

The repair experiment studies what happens after validation feedback is returned to the model that produced the knowledge graph.

The controlled validation setup and RQ1 results are fixed before this experiment begins. The repair experiment does not change the error taxonomy, controlled cases, validator behavior, grounding prompt, or validation results in response to repair outcomes.

## Repair model

The repair model is the same model used for the extraction baseline:

`llama3.1:8b-instruct-q4_K_M`

The model digest recorded for the extraction baseline must match the model used for repair.

Generation settings are also kept the same:

- temperature: 0
- seed: 42
- num_ctx: 4096
- num_predict: 2048

Using the same model keeps model identity fixed between extraction and repair. The experiment measures correction by the extraction model itself rather than performance gained by replacing it with a different corrector.

## Repair prompt

The exact repair prompt is stored in `experiments/repair_prompt.txt`.

Its SHA256 value is recorded in `experiments/repair_spec.json`. The prompt is fixed before any repair case is run and is not changed after the main repair run begins.

Each repair request contains:

1. the source sentence for the controlled case
2. the allowed relation names
3. the current content graph
4. structured validation feedback

The benchmark demonstration example is not included in repair requests. This rule is applied to all controlled cases. One selected Movie case uses the same sentence as the fixed Movie demonstration, so removing the demonstration avoids giving that case extra answer information.

The repair model does not receive the clean reference graph, expected repair, human adjudication, auxiliary RDF type assertions, OWL restrictions, SHACL shapes, or other symbolic scaffolding used only for validation.

### Allowed relations

For Movie and Music cases, the allowed relation set contains the relations from the pinned Text2KGBench ontology for that domain.

For temporal controlled cases, the relation set also contains the controlled temporal relation names already present in the case. These relations belong to the controlled content representation even though they are not part of the pinned Text2KGBench relation set.

A repair output that uses another relation is recorded as an output failure. The evaluation code does not silently replace or rename relations.

## Repair output

The model must return the complete repaired content graph.

The required format is one statement per line:

`relation(subject, object)`

The model is instructed to return only graph statements and no explanation.

The parser follows the same relation, subject, and object format used for the extraction baseline. Exact duplicate statements are removed while preserving order.

Statements that cannot be parsed or that use a relation outside the task vocabulary are recorded as repair output errors. They are not silently corrected.

## Validation used in the repair loop

The main repair loop uses three validation sources:

- raw SHACL
- OWL consistency with HermiT
- the frozen v3 grounding assessor

The supplementary SHACL condition with pySHACL OWL RL inference is not used to generate repair feedback. It remains a separate analysis condition.

The grounding assessor and its prompt remain frozen.

### Validation graph reconstruction

For every repaired graph, the validation code rebuilds the same validation context used for the controlled experiment. It restores the frozen background type assertions from the clean controlled case, the case SHACL shapes, the case OWL context, the pinned source ontology, and the controlled enrichment. Only the repaired content statements are replaced.

Background types are not recomputed from repaired statements. This preserves the fixed validation setup and prevents a new repair statement from receiving a derived type merely because it was added by the repair model.

For cardinality cases, the same controlled OWL restriction used in the original case is restored in the symbolic graph. For temporal cases, the same HermiT `xsd:date` compatibility handling remains in use.

The grounding assessor receives only the source sentence and repaired content statements. It never receives symbolic scaffolding.

## Feedback at the initial repair step

The initial graph is the injected controlled graph.

At the first repair step, feedback is limited to violations associated with the controlled primary modification.

This keeps the experiment focused on the deliberate controlled change. Grounding findings that were already present in the clean baseline are recorded but are not sent to the repair model.

A grounding signal that was produced by the frozen assessor is not corrected using later human review before it is sent to the repair model. False positive and false negative behavior therefore remains part of the observed correction loop.

If the controlled modification receives no actionable feedback from any main validator, no repair request is made at that step.

## Feedback after a repair

After each repair, the new graph is validated again.

Feedback for the next repair step contains:

- any remaining violation associated with the controlled modification
- any new symbolic violation introduced by the repair
- any new grounding violation that was not already present in the clean baseline

Grounding findings that were already present in the clean baseline remain background results and are not sent as repair feedback.

This rule allows the loop to react to damage introduced by a repair without letting unrelated baseline grounding noise dominate the controlled experiment.

## Feedback format

Each feedback item records:

- validator
- violation identity
- error type when known
- focus when available
- message

For SHACL, the result path is retained when present.

The feedback describes the validator result without revealing the clean reference graph or the expected repair. A SHACL message may state the failed constraint, focus, and path. For example, a minimum count result may state that a director value is missing. It must not provide the missing director value from the clean reference graph.

The main repair experiment uses one combined feedback set containing all actionable feedback from the three main validators. The source validator is retained for every item. This records which validators contributed feedback, but it does not isolate the causal effect of each validator when several feedback items are shown together.

## Violation identity

### SHACL

Each SHACL result is kept as a separate violation. Its stable identity uses:

- `sourceConstraintComponent`
- `focusNode`
- `resultPath`, when present
- `value`, when present
- `sourceShape`

Missing fields are represented explicitly as missing values. Including the path, value, and shape prevents different violations on the same focus node from collapsing into one identity.

### Grounding

A grounding identity is based on the unsupported content assertion.

### OWL consistency

OWL consistency uses one stable inconsistency identity for each controlled case. The graph state is also required for the stalled and oscillation rules, so a changed graph cannot be treated as stalled merely because an OWL inconsistency remains. Because the main condition does not request a reasoner explanation, distinct OWL causes inside one repaired graph are not claimed to be separately localized.

## OWL feedback

HermiT supplies the consistency verdict for the main repair condition. The main condition does not use a reasoner explanation.

When the original controlled disjointness inconsistency is still present, the neutral OWL feedback contains the validator identity, the stable inconsistency identity, the focus entity involved in the controlled inconsistency, and the message `The graph is logically inconsistent.`

It does not state which assertion should be removed or added, which classes are disjoint, or what the expected repair is.

If a later repair produces an OWL inconsistency that cannot be identified as the original controlled inconsistency without an explanation, the feedback contains the inconsistency verdict without a focus entity. More detailed OWL explanations are reserved for the later feedback framing experiment.

## Repair rounds

The maximum number of repair rounds is five.

The injected graph is round 0 and is not counted as a repair round.

A repair request produces round 1. Further repair requests produce rounds 2 through 5.

No prompt, model, validator, case, or stop rule may be changed after the main repair run has started.

## Stop rules

A trajectory stops when one of the following conditions is met.

### Validated

No actionable violation remains.

This does not by itself mean that the graph was restored to the clean reference graph.

### Stalled

The asserted content triple set and the actionable violation identity set are both unchanged across two consecutive repair rounds.

The initial injected graph is not used by itself to declare a stalled trajectory. A model that makes no change in round 1 receives one more repair attempt. If the same state remains in round 2, the trajectory is stalled.

### Oscillation

A repair round returns to a graph state and actionable violation set that occurred in an earlier nonadjacent repair round.

### Maximum rounds

The trajectory reaches round 5 without meeting another stop rule.

## Main outcome measures

### End to end target resolution

Whether the controlled primary violation is resolved across all 50 controlled cases. Cases that receive no actionable feedback remain in this denominator.

### Target resolution given feedback

Whether the controlled primary violation is resolved among cases that received at least one actionable feedback item. This is reported separately from end to end target resolution.

### Reference recovery

Whether the final content triple set exactly matches the clean reference content graph for that controlled case.

Reference recovery is stricter than validator success. A graph can satisfy all validators without returning to the reference graph.

The clean graph is a controlled experimental reference. Exact recovery must not be described as complete recovery of every fact in the source sentence.

### Rounds to resolution

The first repair round in which the controlled violation is resolved.

### Validated state

Whether the trajectory reaches a state with no actionable violations.

### Collateral edits

Changes to statements that were not part of the controlled primary modification. The analysis records clean reference statements removed by the repair, new statements not present in the clean reference graph, and the total symmetric difference from the clean reference graph.

### New violations

Violations that were not present in the initial controlled feedback and appear after a repair. These are recorded by validator.

### Stalled trajectories

Trajectories that meet the stalled stop rule.

### Oscillation

Trajectories that revisit a previous nonadjacent repair state.

### Output failures

Parsing failures, empty outputs, and relations outside the allowed relation set are recorded separately and are not silently repaired by the evaluation code.

## Grounding interpretation

The grounding assessor is an empirical model and not ground truth.

Its feedback is used exactly as produced by the frozen assessor during the repair loop.

Known false positive and false negative behavior from the controlled validation study is retained in later interpretation. Human review is not used to edit the feedback before repair.

Grounding results during repair are therefore reported as assessor behavior unless a separate human review is explicitly stated.

## Controlled cases with no feedback

A controlled error may remain unresolved because no validator reports it.

Such a case is part of the end to end correction loop result. It is not removed from the denominator and is not given oracle feedback. Conditional repair results among cases that received feedback are reported separately.

## Analysis unit

The main analysis unit remains the controlled case.

Results are reported by controlled error category and by feedback source.

The experiment characterizes repair behavior on this controlled set. It does not estimate population performance.

## RQ3 boundary

The main repair experiment uses the combined feedback format defined above.

The later feedback framing analysis for OWL inconsistency is a separate experiment. It may compare different levels of OWL feedback, but it must not be used to change the main repair results after they are observed.

## Runner behavior

The main runner is `src/run_controlled_repair.py`.

Before a repair run starts, it verifies the repair prompt hash, the repair model digest, the frozen grounding prompt and model identity, and the required frozen result files.

The frozen clean and injected grounding judgments seed a cache for each controlled case. Assertions that already have a frozen judgment are not sent to the grounding model again. A new assertion created during repair is sent to the frozen grounding assessor the first time it appears. If the same assertion appears again later in the trajectory, its existing judgment is reused.

This keeps the initial controlled grounding result unchanged and avoids repeated judgments for an identical source sentence and assertion inside one trajectory.

The clean and injected graphs were assessed in separate grounding calls. The same unchanged assertion can receive two stored verdicts. When that happens, the injected verdict is used in the trajectory cache because round 0 is the injected graph. The clean verdict is still used to identify baseline grounding findings that must remain excluded from actionable feedback. Assertions that appear only in the clean graph keep that stored verdict if a repair brings them back. The grounding assessor is not called again to choose between the two stored verdicts.

A malformed, empty, out of vocabulary, or truncated repair output is recorded as an output failure. The model is not asked to regenerate a different answer after an output failure. Transport failures are treated as execution failures and abort the run rather than being counted as repair failures.

If round 0 has no actionable feedback, the trajectory stops with `no_feedback`. No repair request is made, and the case remains in the end to end denominator.

For every repair round, the result file records the rendered prompt, its SHA256 value, the raw model response, parsed content statements, validator results, actionable feedback, target resolution, reference comparison, new violation identities, and the final trajectory outcome.

The runner also provides `--preflight-only`. This mode checks the frozen files and local model identities but does not generate a repair or validate a controlled case.
