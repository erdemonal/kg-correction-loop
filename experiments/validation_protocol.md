# Controlled validation protocol

## Status

The controlled validation setup was fixed before the repair experiments began. This record states the method, known limits, software environment, and file hashes. It does not run any experiment again.

The repository state before this documentation step is commit `f55d42e21bab1417fa6ab93bec00d8c668a2b035` (`Add frozen grounding validation`).

The controlled study contains 50 primary modifications: 25 Movie cases and 25 Music cases. Each domain has five cases for disjointness, domain and range, cardinality, temporal ordering, and grounding.

The symbolic validation code produced the expected pattern for all 50 controlled cases. This confirms that the validation code behaves as intended on these cases. It does not mean that the validators are perfectly accurate.

## Grounding assessor

The frozen grounding assessor is `qwen2.5:7b-instruct-q4_K_M`.

Model digest:

`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`

Frozen v3 prompt SHA256:

`7fa341845f64e6c2b7079026bc4567e33f0ef3910c7e02f3efc654559223e73b`

The model and prompt were fixed before the reserved evaluation set and the controlled evaluation. There is no v4. Neither the prompt nor the model was changed after either evaluation.

The reserved evaluation set used 15 natural pilot cases. It produced 9 true positives, 3 false positives, 3 true negatives, and 0 false negatives at the case level. This is a small check on natural examples, not a full benchmark.

### Calibration labels

The human pilot labels used for calibration and the reserved evaluation set are case level labels. A case is marked as a grounding error when at least one extracted assertion is unsupported.

The grounding assessor returns a decision for each asserted triple. In the controlled RQ1 analysis, the relevant unit is the assertion or assertions changed by the controlled modification.

The pilot did not include a human label for every individual triple. The study therefore did not measure the assessor's accuracy separately for each triple. The controlled results show how the frozen assessor behaved in this study. They should not be read as a general estimate of grounding accuracy.

## Controlled grounding results

The main unit of analysis is the controlled modification. This unit was defined before the controlled grounding results were interpreted. Grounding findings on unchanged baseline assertions are kept as separate background results.

For addition cases, the verdict used is the one on the added assertion. For temporal cases, the verdicts used are those on the two new assertions created by the date swap. For cardinality cases, the modification removes an assertion. The grounding assessor checks assertions that are present, so it cannot detect the omission itself.

The frozen grounding result matched the expected behavior for 44 of the 50 controlled modifications:

- disjointness: 10/10 matches, with 10/10 detections
- domain and range: 5/10 matches, with 5/10 detections
- cardinality: 10/10 matches, with 0/10 detections
- temporal ordering: 10/10 matches, with 10/10 detections
- grounding: 9/10 matches, with 9/10 detections

The original graph results are also kept. Only 29 of the 50 clean graphs matched the expectation of no grounding error. The assessor marked at least one assertion as unsupported in 21 of the 50 clean graphs.

After the controlled modification was separated from unchanged assertions, 19 of the 50 clean cases still contained an unsupported background assertion according to the assessor. These results are not discarded. They are also not automatically called false positives because the controlled cases were selected for the symbolic study, not for full human review of every baseline assertion.

### Movie narrative location cases

Five false positives occurred in the Movie domain for `narrative_location` assertions whose objects were countries that were clearly supported by the source sentence: Israel, Japan, Italy, Cuba, and South Africa.

The frozen grounding assessor rejected all five assertions even though the source supported the location relation.

This shows a consistent false positive pattern in the tested Movie `narrative_location` cases with Country objects. It does not show that the assessor fails on domain and range cases in general. All five Music `record_label` cases behaved as expected.

One false negative occurred for `composer(Bei Mir Bistu Shein, Jacob Jacobs)`. The source clearly identifies Jacob Jacobs as the lyricist and Sholom Secunda as the composer.

The six mismatches were reviewed after the frozen run. This review only explains the observed errors. It does not change the model, prompt, cases, expected outcomes, or the frozen 44/50 result.

## HermiT and xsd:date

The main OWL consistency engine is HermiT through Owlready2.

In the current environment, HermiT rejects `xsd:date` because that datatype is not supported by the datatype map used by the reasoner.

Before a graph is sent to HermiT, the code creates a compatibility copy. In that copy, it removes triples whose object is an `xsd:date` literal and triples that directly use the `xsd:date` datatype IRI.

This helper is used whenever such triples are present. It is not limited to the temporal cases.

SHACL and grounding still receive the original date assertions. The temporal cases are still checked by the HermiT procedure, but the OWL part of the study does not examine the date assertions or their order. Temporal order is checked with SHACL SPARQL.

This is a limitation of the reasoner setup used in this experiment. It must not be presented as a general limitation of OWL 2.

## SHACL with OWL RL inference

The supplementary SHACL condition uses the OWL RL inference option provided by pySHACL, with the ontology supplied during validation.

The paper should describe this as SHACL validation with pySHACL OWL RL inference enabled. It should not present this condition as equivalent to a general rule rewriting method or as a separate OWL RL materialization pipeline.

## Human review of grounding errors

The six grounding mismatches were reviewed only after the frozen run. The review is used to describe the assessor's behavior. It is not used to change the cases, labels, prompt, model, or controlled result.

## Statistical scope

Each domain and error category contains five controlled cases. The study therefore describes the detection patterns in this controlled set. It does not estimate how common these patterns are in a larger population.

Significance tests and confidence intervals are not used for groups of only five cases. Any later bootstrap analysis must use a larger and suitable unit of analysis.

## Model separation

The extraction model and grounding assessor come from different model families and are used as separate components. This does not mean that their errors are statistically independent. The models may still share training data and similar language model biases.

## Fixed setup before repair

The following elements are fixed before the repair experiments begin:

- error taxonomy version 1
- the 50 controlled case identities and primary modifications
- the generated clean and injected graphs
- ontology enrichment and SHACL shapes
- symbolic validator code
- grounding prompts v1, v2, and v3, with v3 fixed for evaluation
- grounding model identifier and full digest
- the rule used to assign grounding results to the controlled modification
- reserved evaluation and controlled grounding results
- the human review record
- software environment information in `experiments/environment.json`
- artifact hashes in `experiments/validation_manifest.json`

The repair experiments must not change these elements in response to RQ2 results.

The convergence rule is also fixed before RQ2:

> A repair trajectory is considered converged when both the violation identity set and the asserted triple set remain unchanged for two consecutive repair rounds.

The repair prompt and the maximum number of repair rounds are separate RQ2 choices. They must be fixed before the repair experiment is run.

## Claim boundaries

The following claims are supported by the controlled validation study:

- SHACL and OWL show the expected semantic differences on these controlled cases.
- The three validation methods show different and overlapping behavior across the 50 controlled modifications.
- The grounding assessor is treated as a component whose behavior is measured rather than assumed to be correct.
- The symbolic validation code produced the expected pattern on all 50 controlled cases.

The project must not claim:

- that SHACL is simply a closed world reasoner
- that OWL cannot express or reason about cardinality
- that the grounding assessor is ground truth
- that the validators achieved 100% accuracy
- that the three validation methods cover every form of semantic validation
- that Qwen and Llama are statistically independent
- that the grounding assessor reliably measures source support in general
- that five cases per group estimate population behavior
- that reproducing the expected symbolic pattern proves that the validators are correct

## Provenance

Run:

`python -m src.capture_validation_environment`

before the commit that records this protocol.

The command records Python, the operating system, Java, Ollama, relevant Python package versions, HermiT JAR hashes, Git state, controlled output hashes, and result hashes. It does not run any validator or grounding model.
