# Anatomy of the Correction Loop

## Research objective

This project studies how SHACL, OWL 2 DL reasoning with HermiT, and a locked language model grounding assessor detect controlled extraction errors in knowledge graphs. It also studies how validation feedback affects later graph repair by the same extraction model.

## Current controlled study scope

The locked Movie and Music testbed contains 50 controlled cases, 25 from Movie and 25 from Music. Each domain has five cases for disjointness, domain and range, cardinality, temporal ordering, and grounding. Cases are constructed by controlled injection into Text2KGBench graphs. They are not a sample of naturally occurring extraction errors.

The repair model is `llama3.1:8b-instruct-q4_K_M`. Generation settings are temperature 0, seed 42, `num_ctx=4096`, and `num_predict=2048`. The grounding assessor is `qwen2.5:7b-instruct-q4_K_M`, digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, with prompt `experiments/grounding_judge_prompt_v3.txt`. There is no later assessor version.

The 40 case pilot annotation is retained as a development record. The study does not claim a human ground truth for natural extraction errors. It does not claim a natural error frequency.

## Exclusion of the abandoned 300 case annotation study

A 300 case annotation of natural extraction outputs was started and then fully abandoned. That study is not part of the current design. Its materials were removed and will not be restored. No result in this repository is a human ground truth over natural extraction errors.

## Completed preliminary components

RQ1 is controlled validation on the 50 Movie and Music cases. Symbolic validators produced the expected pattern on all 50 cases. The locked grounding assessor matched the expected primary modification outcome on 44 of 50 cases.

RQ2 is iterative repair of the same 50 cases, with a maximum of five rounds. Feedback comes from raw SHACL, OWL consistency, and locked version 3 grounding.

RQ3 is a paired OWL feedback framing study on the 10 disjointness cases. Each case is repaired once under each of three framings.

The existing RQ1, RQ2, and RQ3 runs are locked preliminary results. They are not a confirmatory study.

## Verified headline results

In RQ1, raw SHACL detected 40 of 50 cases. OWL inconsistency detected 10 of 50 cases, all disjointness. Grounding produced 34 observed detections. The assessor matched expected primary modification status on 44 of 50 cases, with 29 true positives, 5 false positives, 15 true negatives, and 1 false negative. The five false positives are Movie `narrative_location` assertions whose objects are countries. The false negative is Music case `ont_2_music_test_252`, `composer(Bei Mir Bistu Shein, Jacob Jacobs)`. That case received no initial actionable feedback in RQ2. HermiT is given a compatibility copy that strips `xsd:date`. Temporal order is checked by SHACL SPARQL, not by OWL.

In RQ2, end to end target resolution is 37 of 50. Ever resolved is 43 of 50. Last validated target resolution is 42 of 50. Exact clean reference recovery is 22 of 50. Validated stop is 30 of 50. Validated state is 31 of 50. Collateral edits occur in 26 of 50. New violation identities occur in 20 of 50. Output failure occurs in 9 of 50. First resolution occurs in round 1 for 41 cases, round 2 for 1 case, and round 3 for 1 case. Seven cases never resolve the target. Among the 43 cases that were ever resolved, one later lost the target in another valid graph, and five later ended in an unusable output.

Output failure, end to end outcome, last validated state, and ever resolved remain separate measures. They are not interchangeable.

In RQ3, the controlled target is removed in 8 of 10 verdict cases, 8 of 10 location cases, and 9 of 10 explanation cases. Unusable outputs occur in 2 of 10, 1 of 10, and 1 of 10 cases. Exact clean reference recovery occurs in 0 of 10, 0 of 10, and 1 of 10 cases. Among usable outputs, OWL consistency is 6 of 8, 6 of 9, and 7 of 9. Collateral edits are common. These are descriptive paired differences on ten controlled cases. They are not a population estimate.

## Verified quality and cost results

Primary clean reference F1 uses the 40 cases whose initial reference size is greater than 0. The 10 domain and range cases have empty clean references. An F1 of 0 against an empty reference is a scoring convention, not the primary quality estimate. Extra triples are the metric for those 10 cases.

On the 40 nonempty reference cases, mean F1 moved from 0.590 at the injected start to 0.833 at the last validated graph. The paired mean change is 0.244 and the median change is 0.127. Twenty five cases improved, six were unchanged, and nine worsened. A case bootstrap with 10000 resamples and seed 42 gives the interval [0.191, 0.294]. Bootstrap intervals describe resampling sensitivity of these controlled cases. They are not a population confidence interval.

On the same 40 cases, disjointness moved from 0.852 to 0.677, cardinality from 0.625 to 0.830, temporal from 0.000 to 0.907, and grounding from 0.881 to 0.919.

On the 10 empty reference cases, mean extra triples moved from 1.0 to 3.3, a paired mean change of 2.3. No case improved, two were unchanged, and eight worsened. Empty graph recovery is 0 of 10. End to end target resolution is 3 of 10. Output failure is 5 of 10. Validated state is 0 of 10.

A secondary 50 case F1 summary that still includes the empty reference convention moves from 0.472 to 0.667, a mean change of 0.195. It is not the primary quality result.

Clean reference F1 is a set comparison with the controlled clean graph. It is not human source faithfulness. It is not Text2KGBench benchmark F1.

Recorded model cost over 50 cases is 97 repair generations, 103 live grounding calls, and 584.4 seconds of Ollama duration, about 11.7 seconds per case. These figures are not money, wall clock runtime, reasoner time, or SHACL time.

## Locked artifacts and standing methodological rules

The locked Movie and Music result files remain unchanged. These include `results/controlled_repair_trajectories.jsonl`, the RQ1 validation outputs, the RQ3 framing outputs, the locked version 3 calibration and held out judge files, the ontologies and shapes, and `experiments/pilot_annotation.jsonl`.

The locked extractor, repair model, repair prompt, grounding assessor, and version 3 grounding prompt remain unchanged after those runs. The abandoned 300 case annotation is not restored.

Output failure, end to end outcome, last validated state, and ever resolved remain separate measures. Grounding labels are not treated as human ground truth. The 50 case set is not treated as a confirmatory sample.

## Known deviations from the original proposal

The 300 case human review of natural extraction errors was abandoned.

Functional property conflict remains in the taxonomy but is not instantiated for Movie or Music. The pinned ontologies do not support a defensible functional property for that injection.

HermiT in the recorded environment rejects `xsd:date`. The OWL procedure therefore strips date triples before reasoning. This is a limitation of the recorded reasoner setup, not a general claim about OWL 2.

The grounding assessor did not match expected primary modification status on 6 of 50 cases. Those mismatches remain in the locked record.

Primary F1 excludes the 10 empty reference domain and range cases. That exclusion was not part of the original quality plan.

## Preliminary versus confirmatory work

The completed Movie and Music RQ1, RQ2, and RQ3 runs, including the quality and cost extension, are a locked preliminary and development set. They are not a confirmatory experiment. A later confirmatory run must be specified and executed separately. It must not be pooled with these 50 cases as if they were one sample.

## Next planned extension

The next design step is a SOSA and SSN domain and an expanded controlled testbed. Error categories, source text sources, the eligible case pool, and sample size will be fixed only after the SOSA and SSN axioms and application profile rules are written. No sample size per category is locked yet.
