# Requested audit before confirmatory execution: SOSA and SSN

Audit the exact commit supplied with this request. Do not audit an uncommitted
working tree and do not run either model.

Please inspect the locked source, case, and symbolic chain and the new
grounding and repair runner package. In particular, verify:

1. the 180 unique cases and 30 cases per condition cannot drift
2. hashes bind the runners to the committed case and symbolic preflight files
3. the same locked Llama repair model, Qwen grounding assessor, and prompts are
   retained from the preliminary study
4. scaffold and clean reference triples are never shown to either model
5. grounding target expectations follow injection metadata and cardinality is
   correctly undetectable by assertion grounding
6. assessor outputs are not treated as human ground truth or used to change the
   prompt or cases
7. SHACL and OWL reconstruction and HermiT offline and date compatibility
   preserve the locked semantic profile
8. target resolution, validated state, reference recovery, output failure, and
   ever resolved dynamics remain distinct
9. complete graph parsing, object kind reconstruction, retries, resume, and
   cost logging cannot silently alter or duplicate cases
10. the pending audit gate prevents grounding or repair execution before an
    accepted verdict

Return one verdict:

- **A — accepted for confirmatory execution**
- **B — revision required before execution**, with exact blocking issues
- **C — design invalid**, with the methodological reason

State the full audited commit SHA in the verdict. Do not recommend changing the
design after inspecting model outcomes because no confirmatory model outcome
exists at audit time.
