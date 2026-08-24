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

Using the same model keeps model identity fixed between extraction and repair. The experiment measures correction by the extraction model itself, rather than performance gained by replacing it with a different corrector.

## Repair input

Each repair request contains:

1. the original Text2KGBench task prompt for the case
2. the current content graph
3. structured validation feedback

The current content graph contains only statements that belong to the extracted or controlled content representation.

The repair model does not receive auxiliary RDF type assertions, OWL restrictions, SHACL shapes, or other symbolic scaffolding that was added only for validation.

The original Text2KGBench ontology description remains part of the task prompt. As a result, the repair model is given the same extraction vocabulary used in the baseline task.

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

## Feedback at the initial repair step

The initial graph is the injected controlled graph.

At the first repair step, feedback is limited to violations associated with the controlled primary modification.

This keeps the experiment focused on the deliberate controlled change. Grounding findings that were already present in the clean baseline are recorded but are not sent to the repair model.

A grounding signal produced by the frozen assessor is not revised based on later human review before it is sent to the repair model. As a result, false positive and false negative behavior remains part of the observed correction loop.

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
- focus
- message

The focus identifies the assertion, entity, or path involved in the reported problem.

The feedback must describe the validator result without revealing the clean reference graph or the expected repair.

The main repair experiment uses one combined feedback set containing all actionable feedback from the three main validators.

The source validator is retained for every item so that later analysis can separate SHACL, OWL, and grounding feedback.

## Violation identity

Violation identities must be stable enough to compare consecutive repair rounds.

SHACL identities are based on the reported constraint and focus.

Grounding identities are based on the unsupported assertion.

OWL consistency uses a stable case level inconsistency identity. The graph state is also required for convergence, so a changed graph cannot be treated as converged merely because an OWL inconsistency remains.

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

### Target resolution

Whether the controlled primary violation is resolved.

### Reference recovery

Whether the final content triple set exactly matches the clean reference content graph for that controlled case.

Reference recovery is stricter than validator success. A graph can satisfy all validators without returning to the reference graph.

The clean graph is a controlled experimental reference. Exact recovery must not be described as complete recovery of every fact in the source sentence.

### Rounds to resolution

The first repair round in which the controlled violation is resolved.

### Validated state

Whether the trajectory reaches a state with no actionable violations.

### Collateral edits

Changes to statements that were not part of the controlled primary modification.

The analysis records:

- clean reference statements removed by the repair
- new statements not present in the clean reference graph
- total symmetric difference from the clean reference graph

### New violations

Violations that were not present in the initial controlled feedback and appear after a repair.

These are recorded by validator.

### Stalled trajectories

Trajectories that meet the stalled stop rule.

### Oscillation

Trajectories that revisit a previous nonadjacent repair state.

### Output failures

Parsing failures, empty outputs, and relations outside the task vocabulary are recorded separately and are not silently repaired by the evaluation code.

## Grounding interpretation

The grounding assessor is an empirical model and not ground truth.

Its feedback is used exactly as produced by the frozen assessor during the repair loop.

Known false positive and false negative behavior from the controlled validation study is retained in later interpretation. Human review is not used to edit the feedback before repair.

Grounding results during repair are reported as assessor behavior unless a separate human review is explicitly stated.

## Controlled cases with no feedback

A controlled error may remain unresolved because no validator reports it.

Such a case is part of the correction loop result. It is not removed from the denominator and is not given oracle feedback.

## Analysis unit

The main analysis unit remains the controlled case.

Results are reported by controlled error category and by feedback source.

The experiment characterizes repair behavior on this controlled set. It does not estimate population performance.

## RQ3 boundary

The main repair experiment uses the combined feedback format defined above.

The later feedback framing analysis for OWL inconsistency is a separate experiment. It may compare different levels of OWL feedback, but it must not be used to change the main repair results after they are observed.
