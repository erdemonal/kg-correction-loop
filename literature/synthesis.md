# Literature synthesis

Recent research explores knowledge graph extraction, validation, and repair, but these aspects are often treated independently.

Text2KGBench provides an important benchmark for extracting knowledge graphs from text using an ontology. It evaluates extraction accuracy, ontology conformance, and hallucination as separate aspects of output quality. Later systems such as iText2KG, KGGen, OntoLogX, ANCHOR, and Wikontic also show the role of ontology information and validation in knowledge graph construction.

SHACL and OWL address distinct requirements. SHACL verifies whether an RDF graph meets explicit constraints, whereas OWL reasoning applies open world assumption and can infer information not explicitly stated in the graph. Oudshoorn et al. emphasize this difference and argue against equating SHACL validation with OWL reasoning.

Language models provide another way to assess graph statements. Regino and dos Reis study the use of language models to validate RDF triples. Adam and Kliegr evaluate graph statements against supporting text. Tsaneva et al. and Dechtiar et al. also examine approaches that combine different forms of validation rather than relying on a language model alone.

Graph repair is addressed in several studies. Lin et al. assess the ability of language models to repair controlled SHACL violations. Terdalkar et al. compare the performance of multiple language models on graph repair tasks. Kim et al. investigate factual errors in generated knowledge graphs and strategies for their correction. OntoLogX and related systems use validation outcomes to support iterative correction.

Research on language model correction shows that feedback can affect the outcome of repair. Huang et al. report important limits of self correction without reliable external feedback. Kamoi et al. identify external feedback as an important factor in successful correction. Qi et al. show that the way feedback is presented can also change correction behavior.

The literature covers the main components of this study, but usually examines them separately. This study asks how different forms of validation behave on the same controlled extraction errors and how their feedback affects repeated graph repair. It compares SHACL, OWL 2 DL reasoning, and language model assessment against the source text. It then examines error reduction, convergence, oscillation, unintended graph changes, and faithfulness to the original text during repair.
