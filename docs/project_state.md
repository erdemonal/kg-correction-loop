# Project state

## Status

The final controlled experimental design and its planned analyses are complete. The repository holds the implementation, locked protocols, controlled cases, validators, repair runners, and analysis and reporting code. At commit `e529b89e04a3d2cfa0e3390e24e3e3ba7c4f470f`, the test suite contains 300 passing tests.

The remaining work is not experimental. Publish the result archives with their hashes, write the dissertation paper, prepare the presentation, and take in supervisor comments. No further model run is required for the reported research questions.

## Research objective

This project studies two linked questions about knowledge graph extraction that uses an ontology:

1. Which controlled extraction errors are detected by SHACL validation, OWL 2 DL consistency reasoning, and source grounding assessment?
2. When that feedback is given to a language model, does the model remove the target error without producing unusable output, adding new violations, or changing otherwise correct graph content?

A third analysis tests whether the amount of information in OWL inconsistency feedback changes repair after one step.

The study uses controlled fault injection. The clean graph, injected change, expected validator behavior, and repair target are known before execution. The results describe these cases. They do not estimate how often such errors occur in ordinary extraction.

## Completed studies

| Study | Data | Design | Status |
|---|---|---|---|
| Preliminary RQ1 and RQ2 | Movie and Music | 50 cases, 10 per condition across five conditions | Complete and locked |
| Preliminary RQ3 | Movie and Music disjointness | 10 paired cases, three feedback framings per case | Complete and locked |
| Preliminary quality and cost analysis | Movie and Music | Clean reference quality, side effects, model calls, tokens, and recorded generation duration | Complete and locked |
| Confirmatory RQ1 and RQ2 | SOSA and SSN hydrological observation profile | 180 unique cases, 30 per condition across six conditions | Complete and locked |
| Confirmatory RQ3 | SOSA and SSN disjointness | 30 paired cases, three feedback framings per case, 90 generations | Complete and locked |

The confirmatory sample contains 168 records from distinct USGS monitoring locations and 12 official W3C examples. It is one hydrological observation application profile. It is not a sample of the full SOSA and SSN ecosystem. Preliminary and confirmatory results are reported separately and are not pooled.

## Fixed models and validation methods

The extraction and repair model is `llama3.1:8b-instruct-q4_K_M`, digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`. The source grounding assessor is `qwen2.5:7b-instruct-q4_K_M`, digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, with the locked version 3 prompt. Both use temperature 0 and seed 42.

The symbolic methods are raw SHACL validation and OWL 2 DL consistency reasoning with HermiT. SHACL and OWL answer different questions. Their findings are not treated as interchangeable. The grounding assessor is an automated instrument, not human ground truth.

## RQ1: validator coverage

### Preliminary Movie and Music study

On the 50 controlled cases, raw SHACL detected 40 cases. OWL inconsistency detected the 10 disjointness cases. The grounding assessor matched the expected primary modification status on 44 cases: 29 true positives, 5 false positives, 15 true negatives, and 1 false negative.

### Confirmatory SOSA and SSN study

| Validator outcome | Cases | Rate |
|---|---:|---:|
| Raw SHACL target detected | 150/180 | 83.3% |
| OWL inconsistency | 60/180 | 33.3% |
| Grounding target detected | 120/180 | 66.7% |

The methods cover different cases. The overlap was 58 cases for all three methods, 32 for SHACL plus grounding, 2 for SHACL plus OWL, 58 for SHACL alone, and 30 for grounding alone.

For the target modification, the grounding counts were 120 true positives, 30 false negatives, 30 true negatives, and 0 false positives. That is a result about the injected target. It is not graph level grounding accuracy. The assessor also marked unsupported assertions on every clean graph and every injected graph. That baseline noise is part of the record.

Coverage matched the semantics fixed before the run. SHACL covered explicit application profile constraints. OWL inconsistency exposed the W3C disjointness and functional property conflicts. Grounding detected contradictions with the source text but could not detect an omitted triple in the cardinality condition.

## RQ2: iterative repair

The repair loop used combined SHACL, OWL, and grounding feedback for at most five rounds. Target removal, usable output, overall validation, exact recovery, and collateral change are reported separately.

### Preliminary Movie and Music study

| Outcome | Cases |
|---|---:|
| Target resolved in final output | 37/50 |
| Target resolved at least once | 43/50 |
| Target resolved in last valid graph | 42/50 |
| Final graph in validated state | 31/50 |
| Exact clean reference recovery | 22/50 |
| Output failure | 9/50 |
| Collateral edit | 26/50 |
| New violation identity | 20/50 |

### Confirmatory SOSA and SSN study

| Outcome | Cases | Rate |
|---|---:|---:|
| Target resolved at least once | 166/180 | 92.2% |
| Target resolved in last valid graph | 165/180 | 91.7% |
| Target resolved in final output | 117/180 | 65.0% |
| Final graph in validated state | 87/180 | 48.3% |
| Exact clean reference recovery | 81/180 | 45.0% |
| Output failure | 63/180 | 35.0% |
| Collateral edit | 99/180 | 55.0% |
| New violation identity | 111/180 | 61.7% |

The sequence `166 -> 165 -> 117` is the main result. The model removed the target at least once in 166 trajectories. One later valid graph put the target back. Of the 165 trajectories whose last valid graph had the target removed, 48 then ended with unusable output. Repair with validator feedback was often successful at some point in the trajectory. It was not reliably successful at the end.

| Condition | Final target resolved | Exact recovery | Output failure |
|---|---:|---:|---:|
| Cardinality | 24/30 | 17/30 | 6/30 |
| Disjointness | 12/30 | 5/30 | 18/30 |
| Domain and range | 24/30 | 23/30 | 6/30 |
| Functional property conflict | 20/30 | 12/30 | 10/30 |
| Grounding | 14/30 | 5/30 | 16/30 |
| Temporal | 23/30 | 19/30 | 7/30 |

Disjointness had the lowest final target resolution. The functional property condition exists only because SOSA and SSN supplies those axioms. It was neither the easiest nor the hardest condition.

## RQ3: feedback framing

The paired experiment compares three messages for the same OWL inconsistency: a verdict, a verdict plus the focus entity, and a controlled explanation naming the disjoint classes. Every condition starts from the same injected graph and receives one repair step.

| Study | Verdict | Location | Explanation |
|---|---:|---:|---:|
| Preliminary target removal | 8/10 | 8/10 | 9/10 |
| Confirmatory target removal | 0/30 | 0/30 | 19/30 |
| Confirmatory exact recovery | 0/30 | 0/30 | 8/30 |
| Confirmatory output failure | 1/30 | 5/30 | 10/30 |

For the confirmatory paired sample, Cochran's Q was 38.0 with `p = 5.60e-9`. Exact two sided McNemar tests comparing explanation with verdict and with location each had raw `p = 3.81e-6` and Holm adjusted `p = 1.14e-5`. The explanation removed the target more often in this locked sample. It also produced more unusable outputs and did not guarantee exact recovery. These findings apply to 30 disjointness cases in one hydrological application profile. They do not apply to all feedback, models, or SOSA and SSN applications.

## Graph quality and recorded cost

In the preliminary study's 40 cases with nonempty clean references, mean clean reference F1 changed from 0.590 to 0.833 at the last validated graph. The paired mean change was 0.244, with 25 improved, 6 unchanged, and 9 worsened cases. A case bootstrap with 10,000 resamples and seed 42 produced `[0.191, 0.294]`. This interval describes the controlled cases. It is not a population confidence interval.

The 10 domain and range cases had empty clean references and were analyzed separately through extra triples. Mean extra triples increased from 1.0 to 3.3. No case improved and 8 worsened. Clean reference F1 is a controlled graph similarity measure, not an independent human assessment of source faithfulness.

The preliminary repair study recorded 97 repair generations, 103 grounding calls, and 584.4 seconds of Ollama generation duration. The confirmatory repair phase recorded 429 repair calls, 252 live grounding calls, and 5,552.4 seconds of model duration. These are model use measures. SHACL and reasoner runtimes were not instrumented.

## Interpretation

Within the controlled study, the results support four points:

1. SHACL, OWL consistency reasoning, and source grounding catch different error classes. No one method covers the others.
2. Removing the injected target is easier than producing a usable, fully validated, or exactly recovered graph.
3. Repair failures include semantic side effects and output format failures. They must be counted separately.
4. A more detailed inconsistency explanation can help the model find the target. Richer feedback can also increase output failure and collateral change.

These results come from controlled cases. They are not claims about natural error rates, all language models, or the full SOSA and SSN ecosystem.

## Proposal changes and limitations

The final design differs from the original proposal in documented ways. The abandoned human annotation study is excluded. Independent human faithfulness assessment was not performed. A formal Pareto frontier was not defined. SHACL and reasoner runtime were not instrumented. Controlled ontology explanations replaced Pellet generated explanations in RQ3. The full record is in [proposal_deviations.md](proposal_deviations.md).

## Provenance

| Component | Locked commit or result hash |
|---|---|
| Preliminary RQ1 implementation | `f55d42c` |
| Preliminary RQ2 run base | `cd6d912` |
| Preliminary RQ2 analysis | `6ba4cf` |
| Preliminary RQ3 run | `0d9b4d1` |
| Preliminary quality and cost closure | `75e40e2` |
| Confirmatory scope accepted | `c18bb8aebc595bb314628a60efaa7a9b66369210` |
| Confirmatory run commit | `46bf351ba0add65c28599d5dd9c7c0844ace3fe1` |
| Confirmatory repair results SHA-256 | `37b7fd73fd4b00a6055d9442928bb9b812d2dd38b53f4a7bf99db96206019320` |
| Confirmatory grounding results SHA-256 | `6f68534759095c0763551f6df3de9ae143e3da40ae1ab3ede5dab266d0a4eec5` |
| Confirmatory RQ3 run commit | `533a58dfbdda5270190d8b2a86ac2b3c59db8e94` |
| Confirmatory RQ3 results SHA-256 | `85579656ebd3af3ed163c968c5de68b543e4577b29c08201f30a9848cacfaef9` |
| Confirmatory RQ3 analysis and reporting | `e529b89e04a3d2cfa0e3390e24e3e3ba7c4f470f` |
| Confirmatory reporting scope SHA-256 | `30d9382fabd34e0e7a840e6f823a5f0fa5d54c46bb780c26bfd5c2176508e3ee` |

The raw result archives are kept out of ordinary Git history. They must be published as release artifacts with a SHA-256 manifest. The literature search and synthesis are under [`literature/`](../literature/).
