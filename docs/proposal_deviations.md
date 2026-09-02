# Changes from the proposal

The proposal set the first research plan. The final design changed when calibration, ontology inspection, and implementation showed that some assumptions could not be supported. This document records those changes so the paper can tell planned methods from executed methods.

| Proposal plan | Executed design | Reason and reporting consequence |
|---|---|---|
| Manually annotate 300 natural extraction outputs. | The annotation study was abandoned and excluded. The experiments use controlled fault injection with known targets. | One researcher could not provide a credible independent human ground truth. Automated labels were not presented as human annotation. The study does not estimate natural error prevalence. |
| Measure source faithfulness through human validation. | Clean reference comparison and a locked source grounding assessor provide controlled proxy measures. | Independent human source faithfulness evaluation was removed from scope. Clean reference recovery and grounding assessor findings were retained as controlled proxy measurements, but they are not interpreted as human faithfulness judgments. |
| Plot a formal conformance versus faithfulness Pareto frontier. | Report descriptive clean reference quality, target resolution, validation state, collateral change, and grounding findings separately. | A formal Pareto frontier would require a validated independent faithfulness axis that the final study does not have. The reported trade-offs are descriptive. They are not called a Pareto frontier. |
| Record SHACL and reasoner runtime at every iteration. | Record model calls, prompt and generated tokens, Ollama generation duration, and wall time for the confirmatory run. | Validator runtimes were not instrumented in the completed runs. The cost analysis was therefore restricted to recorded model calls, tokens, and Ollama durations. This is a scope reduction and a limitation. |
| Use Pellet generated explanations for the feedback framing experiment. | Use a fixed format controlled explanation derived from the locked ontology axiom. | The experiment isolates the information in the explanation. It does not depend on how Pellet renders explanations. The paper must call it a controlled ontology explanation, not a Pellet generated explanation. |
| Include functional property conflicts in the initial taxonomy. | Do not inject this condition in Movie or Music. Add 30 confirmatory cases using the W3C functional properties `sosa:hasResult` and `sosa:hasSimpleResult`. | The pinned Movie and Music ontologies did not contain a defensible functional property for this injection. SOSA and SSN supplied a standards based condition without inventing an axiom. |
| Apply OWL reasoning to the recorded graphs without a datatype compatibility step. | Remove `xsd:date` triples only from the HermiT compatibility copy. Check temporal order with SHACL SPARQL. | HermiT rejected `xsd:date` in the recorded environment. This is a limit of that environment, not a general limit of OWL 2. |
| Use one F1 analysis for every controlled case. | Use clean reference F1 for 40 nonempty reference cases and extra triple counts for 10 empty reference domain and range cases. | F1 against an empty reference is dominated by a scoring convention. It does not measure meaningful recovery. The two groups are reported separately. |
| Use 100 calibration cases and 200 human labeled held out cases for grounding. | Keep the 40 case development record and 15 case reserved evaluator check, then lock grounding prompt version 3. | No per triple human accuracy estimate is claimed. The grounding assessor is a fixed automated instrument, not human ground truth. Its mismatches and baseline noise remain visible. |
| Use SOSA and SSN as a broad external validity extension. | Study 168 USGS hydrological observations and 12 official W3C examples within one hydrological observation application profile. | The data support a third standards based domain. They do not support claims about the entire SOSA and SSN ecosystem, source family comparisons, or environmental data prevalence. EPA data were deferred and are not represented. |

## Unchanged principles

The revisions did not change the main comparison. The same controlled target is evaluated by SHACL, OWL consistency reasoning, and source grounding. Their feedback is then used in a repair loop. Preliminary and confirmatory results remain separate. Model identities, prompts, controlled cases, and analysis rules were locked before the corresponding runs.

## Required language in the final paper

The final paper should state that:

- The study reports controlled faults, not naturally occurring error frequencies.
- Grounding is an automated assessor and not human ground truth.
- Clean reference F1 is a graph similarity proxy and not independent source faithfulness validation.
- The confirmatory sample is one hydrological observation application profile.
- Feedback framing conclusions apply to the paired disjointness sample and the tested model.
- SHACL and reasoner runtime were not instrumented.
- The SOSA and SSN 2023 Edition input is a pinned draft snapshot, not a W3C Recommendation.
