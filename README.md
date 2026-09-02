# Anatomy of the Correction Loop

This repository reports a controlled Semantic Web experiment. It compares SHACL, OWL 2 DL consistency reasoning, and a language model source grounding assessor on known extraction errors in knowledge graphs. It then measures how their feedback affects later repair.

## Research questions

1. Which controlled error classes does each validation method detect?
2. Does validator feedback produce reliable repair, and what happens after the known error is removed?
3. Does the amount of information in OWL inconsistency feedback change repair after one step?

The study measures two things separately: whether a known target is detected or removed, and whether the resulting graph is usable, validated, and free of collateral damage.

## Study design

| Component | Scope |
|---|---|
| Preliminary study | 50 controlled Movie and Music cases, 10 per condition across disjointness, domain and range, cardinality, temporal, and grounding errors |
| Confirmatory study | 180 unique SOSA and SSN cases, 30 per condition, with an added functional property conflict taken from the W3C ontology |
| Confirmatory sources | 168 distinct USGS monitoring locations and 12 official W3C examples in one hydrological observation application profile |
| Validators | Raw SHACL, OWL 2 DL consistency with HermiT, and a locked source grounding language model assessor |
| Repair | Up to five rounds with combined validator feedback |
| Feedback framing | Paired verdict, location, and controlled explanation conditions on 30 confirmatory disjointness cases |

Cases are built by injecting a known fault into a clean graph. The study reports validator coverage and repair behavior under those conditions. It does not estimate how often such errors occur in ordinary extraction.

## Main results

### Validator coverage

In the 180 case confirmatory study, SHACL detected the controlled target in 150 cases, OWL inconsistency in 60, and source grounding in 120. The methods catch different failures. They are not interchangeable.

### Local repair did not imply a reliable final graph

The repair model removed the target at least once in 166 of 180 trajectories. The last valid graph still had the target removed in 165 cases. Only 117 final outputs still had the target removed. 63 trajectories ended in output failure, 99 contained collateral edits, and 81 recovered the clean reference exactly.

The sequence `166 -> 165 -> 117` is the main repair result. Feedback often made the target removable. Later generation failures still blocked a reliable final graph.

### Feedback detail changed repair

In the 30 case confirmatory paired experiment, verdict only and location feedback removed the target in 0 of 30 cases. A controlled explanation naming the disjoint classes removed it in 19 of 30. Cochran's Q was 38.0 with `p = 5.60e-9`. Holm adjusted exact McNemar comparisons between explanation and each shorter framing had `p = 1.14e-5`.

The explanation helped the model find the target in this sample. It also caused 10 output failures and recovered the clean graph in only 8 cases. More detail helped. It did not make repair safe.

Tables by condition, scope limits, and provenance are in [docs/project_state.md](docs/project_state.md).

## Repository guide

| Path | Contents |
|---|---|
| `experiments/` | Locked protocols, prompts, controlled cases, sampling records, specifications, and input hashes |
| `validation/ontologies/` | Movie and Music enrichments and the pinned SOSA and SSN 2023 Edition snapshot |
| `validation/shapes/` | SHACL shapes for the preliminary and confirmatory studies |
| `src/` | Dataset preparation, validation, model runners, analysis, and report generation |
| `tests/` | Semantic, protocol, hash, runner, analysis, and reporting checks |
| `literature/` | Literature search record and synthesis |
| `docs/project_state.md` | Study status and verified headline results |
| `docs/proposal_deviations.md` | Differences between the proposal and the executed design |

## Verification and reproduction

Run the repository checks in the recorded project environment:

```bash
pytest -q
```

At commit `e529b89e04a3d2cfa0e3390e24e3e3ba7c4f470f`, the expected result is `300 passed`.

The confirmatory symbolic and offline protocol checks can be repeated without calling a model:

```bash
python -m src.validate_sosa_ssn_symbolic
python -m src.preflight_sosa_ssn_confirmatory
python -m src.run_sosa_ssn_feedback_framing --preflight-only
```

After placing the locked raw result files under `results/`, reproduce the confirmatory analyses and reports with:

```bash
python -m src.analyze_sosa_ssn_confirmatory
python -m src.report_sosa_ssn_confirmatory
python -m src.analyze_sosa_ssn_feedback_framing
python -m src.report_sosa_ssn_feedback_framing
```

The raw results are excluded from ordinary Git history. They will be published as release artifacts with SHA-256 hashes. Do not rerun the models only to reproduce the analysis. The recorded environment is in `experiments/environment.json`. Each experiment specification pins its inputs, prompts, models, and generation settings.

## Scope and limitations

- The grounding assessor is an automated instrument, not human ground truth.
- Clean reference F1 measures graph similarity. It is not independent human source faithfulness validation.
- The confirmatory data come from one hydrological observation application profile, not the full SOSA and SSN ecosystem.
- The SOSA and SSN 2023 Edition input is a pinned draft snapshot, not a W3C Recommendation.
- The repair results use one local model and fixed generation settings.
- SHACL and reasoner runtimes were not instrumented.
- Preliminary and confirmatory results are reported separately and are not pooled.

The full record of changes from the original proposal is in [docs/proposal_deviations.md](docs/proposal_deviations.md).

## Literature and contribution

The literature review covers 22 related works from 2023 to 2026 on extraction that uses an ontology, SHACL and OWL validation, language model grounding, graph repair, and self correction. The search record is in [literature/search.md](literature/search.md). The synthesis is in [literature/synthesis.md](literature/synthesis.md). Venue and author metadata should be checked again when this record is turned into the paper bibliography.

The separate pieces are not claimed as new. The contribution is to run them on the same controlled faults and to follow those cases through repair, output failure, collateral change, clean reference recovery, and feedback framing.

## Status

The experiments and analyses in the final controlled design are complete. What remains is to publish the hashed result archives, write the dissertation paper, prepare the presentation, and respond to supervisor comments.

The earlier SemanticGAN codebase is kept in the `legacy/semanticgan` branch and under the `v0.1.1` tag.
