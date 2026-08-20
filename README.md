# Anatomy of the Correction Loop

An empirical study of validation and feedback in knowledge graph extraction from text.

This project examines how SHACL, OWL 2 DL reasoning, and language model assessment identify different kinds of errors in extracted knowledge graphs. It also studies how validation feedback affects iterative graph repair, with attention to convergence, oscillation, collateral damage, and faithfulness to the source text.

The experiments use Text2KGBench with controlled ontology enrichment and error injection.

## Repository history

This repository previously hosted SemanticGAN, a Wasserstein GAN approach to knowledge graph completion. The current project grew out of that earlier work and shifts the focus from graph completion to validation and repair in knowledge graph extraction.

The final SemanticGAN codebase is preserved in the `legacy/semanticgan` branch and under the `v0.1.1` tag.