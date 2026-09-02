# Literature synthesis

Recent work on knowledge graph extraction, validation, and repair often treats these tasks separately.

Text2KGBench is an important benchmark for extracting knowledge graphs from text using an ontology. It scores extraction accuracy, ontology conformance, and hallucination as separate parts of output quality. Later systems such as iText2KG, KGGen, OntoLogX, ANCHOR, and Wikontic also show that ontology information and validation matter in graph construction.

SHACL and OWL answer different questions. SHACL checks whether an RDF graph meets explicit constraints. OWL reasoning follows the open world assumption and can infer facts that are not written in the graph. Oudshoorn et al. stress this difference and warn against treating SHACL validation as OWL reasoning.

Language models offer another way to judge graph statements. Regino and dos Reis study language models as RDF triple validators. Adam and Kliegr compare graph statements with supporting text. Tsaneva et al. and Dechtiar et al. look at combining several forms of validation rather than relying on a language model alone.

Graph repair appears in several studies. Lin et al. test whether language models can repair controlled SHACL violations. Terdalkar et al. compare several models on graph repair. Kim et al. study factual errors in generated knowledge graphs and ways to correct them. OntoLogX and related systems use validation outcomes to guide iterative correction.

Work on language model correction shows that feedback can change repair. Huang et al. report limits of self correction without reliable external feedback. Kamoi et al. find that external feedback is an important factor in successful correction. Qi et al. show that how feedback is written can also change correction.

The literature covers the parts of this study, usually one at a time. This study asks how different validators behave on the same controlled extraction errors and how their feedback affects repeated graph repair. It compares SHACL, OWL 2 DL reasoning, and automated source grounding on the same cases. It then tracks target removal, convergence, oscillation, output failure, unintended graph changes, clean reference similarity, and grounding findings during repair. Clean reference comparison and automated grounding are controlled proxies. They are not an independent human assessment of source faithfulness.
