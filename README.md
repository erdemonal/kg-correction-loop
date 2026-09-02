# Anatomy of the Correction Loop

This project studies how SHACL, OWL 2 DL reasoning with HermiT, and a locked language model grounding assessor detect controlled extraction errors in knowledge graphs. It also studies how validation feedback affects later graph repair by the same extraction model.

## Research questions

1. Which controlled error classes are detected by each validation method?
2. Does validator feedback produce reliable repair, and what happens after the known error is removed?
3. Does the amount of information in OWL inconsistency feedback change repair after one step?

Detection or removal of a known target is scored separately from whether the resulting graph is usable, validated, and free of collateral edits.

## Study design

| Component | Scope |
|---|---|
| Preliminary study | 50 controlled Movie and Music cases, 10 per condition, covering disjointness, domain and range, cardinality, temporal order, and grounding |
| Confirmatory study | 180 unique SOSA and SSN cases, 30 per condition, including a functional property conflict taken from the W3C ontology |
| Confirmatory sources | 168 distinct USGS monitoring locations and 12 official W3C examples, all in one hydrological observation application profile |
| Validators | Raw SHACL, OWL 2 DL consistency with HermiT, and a locked language model assessor for source grounding |
| Repair | At most five rounds with combined validator feedback |
| Feedback framing | Paired verdict, location, and controlled explanation conditions on 30 confirmatory disjointness cases |

Each case is a known clean graph with one injected fault. The results describe validator coverage and repair on these cases. They do not estimate how often such errors occur in unconstrained extraction.

## Results

In the confirmatory sample of 180 cases, SHACL detected the controlled target in 150 cases, OWL inconsistency in 60, and source grounding in 120. The three methods overlap only in part, so they are not substitutes for one another.

The repair model removed the target at least once in 166 of 180 trajectories. The last valid graph still had the target removed in 165 cases, but only 117 final outputs did. Output failure ended 63 trajectories. Collateral edits occurred in 99. Exact recovery of the clean reference occurred in 81. Feedback often made the target removable at some point in the trajectory without producing a reliable final graph.

In the confirmatory paired experiment of 30 disjointness cases, verdict only and location feedback removed the target in 0 of 30 cases. A controlled explanation that named the disjoint classes removed it in 19 of 30. Cochran's Q was 38.0 with `p = 5.60e-9`. Holm adjusted exact McNemar tests comparing explanation with each shorter framing had `p = 1.14e-5`. The explanation increased target removal in this sample. It also produced 10 output failures, and exact recovery occurred in only 8 cases.

Condition tables, scope limits, and provenance are in [docs/project_state.md](docs/project_state.md).

## Repository layout

| Path | Contents |
|---|---|
| `experiments/` | Locked protocols, prompts, controlled cases, sampling records, specifications, and input hashes |
| `validation/ontologies/` | Movie and Music enrichments and the pinned SOSA and SSN 2023 Edition snapshot |
| `validation/shapes/` | SHACL shapes for the preliminary and confirmatory studies |
| `src/` | Dataset construction, validation, model runners, analysis, and report generation |
| `tests/` | Semantic, protocol, hash, runner, analysis, and reporting checks |
| `literature/` | Literature search record and synthesis |
| `docs/project_state.md` | Study status and verified headline counts |
| `docs/proposal_deviations.md` | Differences between the proposal and the executed design |

## Checks and reproduction

In the recorded project environment:

```bash
pytest -q
```

The expected result is `300 passed`. That count was verified at code commit `e529b89e04a3d2cfa0e3390e24e3e3ba7c4f470f`. Later documentation commits did not change the code or tests.

The confirmatory symbolic and offline protocol checks do not call a model:

```bash
python -m src.validate_sosa_ssn_symbolic
python -m src.preflight_sosa_ssn_confirmatory
python -m src.run_sosa_ssn_feedback_framing --preflight-only
```

After the locked raw result files are placed under `results/`, the confirmatory analyses and reports can be reproduced with:

```bash
python -m src.analyze_sosa_ssn_confirmatory
python -m src.report_sosa_ssn_confirmatory
python -m src.analyze_sosa_ssn_feedback_framing
python -m src.report_sosa_ssn_feedback_framing
```

Raw results are excluded from ordinary Git history and will be published as release artifacts with SHA-256 hashes. Reproducing the analysis does not require repeating the model runs. The recorded environment is in `experiments/environment.json`. Each experiment specification records its inputs, prompts, models, and generation settings.

## Scope

The grounding assessor is an automated instrument, not human ground truth. Clean reference F1 measures graph similarity and is not an independent human judgment of source faithfulness. The confirmatory data come from one hydrological observation application profile, not from the SOSA and SSN ecosystem as a whole. The SOSA and SSN 2023 Edition files are a pinned draft snapshot, not a W3C Recommendation. Repair used one local model and fixed generation settings. SHACL and reasoner runtimes were not recorded. Preliminary and confirmatory results are reported separately and are not pooled.

Changes from the original proposal are listed in [docs/proposal_deviations.md](docs/proposal_deviations.md).

## Literature

The review covers 22 papers from 2023 to 2026 on extraction that uses an ontology, SHACL and OWL validation, language model grounding, graph repair, and self correction. The search record is [literature/search.md](literature/search.md). The synthesis is [literature/synthesis.md](literature/synthesis.md). Venue and author details should be checked again when this record is converted into the paper bibliography.

The methods themselves are not claimed as new. The contribution is to compare them on the same controlled faults and to follow those cases through repair, output failure, collateral change, clean reference recovery, and feedback framing.

## Status

The experiments and analyses in the final controlled design are complete. Remaining work is to publish the hashed result archives, write the dissertation paper, and prepare the presentation.

The earlier SemanticGAN codebase is in the `legacy/semanticgan` branch and under the `v0.1.1` tag.
