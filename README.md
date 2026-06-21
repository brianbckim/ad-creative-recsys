# Adgorithm Ad Creative Recsys Pipeline (Talkwalker proxy → Two‑Tower → DIN listwise KD → Ad Copy Scoring)

**Created by:** Brian Kim (Byungchan Kim), Alina Vasina  
**Affiliation:** Adgorithm Club at the University of Florida  
**Contact:** For questions about this project, reach out at bckim.contact@gmail.com.

**Team roles.**
- **Brian Kim (Team Lead).**
	- Built an end-to-end ad creative recsys pipeline: Talkwalker proxy → Two-Tower retrieval → DIN listwise KD → ad copy scoring.
	- Owned data contracts and reproducible artifacts (proxy schema, datasets, checkpoints) and end-to-end training + offline scoring tooling.
	- Implemented Two‑Tower training and candidate generation, plus DIN listwise KD (dataset construction, training loop, embedding export) for sequence-aware reranking.
- **Alina Vasina.**
	- Contributed to early-stage architecture discussions and literature review.
	- Supported the project through research material collection and feedback.

## Documentation

1. [Executive Summary](docs/01-executive-summary.md): project goal, key artifacts, and pipeline snapshot.
2. [System Architecture](docs/02-system-architecture.md): components, data flow, dependency graph, and external requirements.
3. [Data Contracts](docs/03-data-contracts.md): schemas, join keys, ID/index invariants, copy catalog requirements, and versioning conventions.
4. [Environment & Tooling](docs/04-environment-tooling.md): dependencies, hardware assumptions, CLI patterns, and provenance.
5. [End-to-End Pipeline](docs/05-end-to-end-pipeline.md): runnable stage-by-stage workflow.
6. [Model Training & Evaluation](docs/06-model-training-evaluation.md): Two-Tower and DIN KD training, evaluation, metrics, and monitoring.
7. [Copy Intelligence Layer](docs/07-copy-intelligence-layer.md): copy embeddings, theme assignment, item-to-copy mapping, copy-head training, and scoring.
8. [Validation & QA](docs/08-validation-qa.md): regression checks, artifact consistency, sanity dashboards, and ranking diffs.
9. [Troubleshooting & FAQ](docs/09-troubleshooting-faq.md): common environment, contract, and drift issues.
10. [Appendix](docs/10-appendix.md): command reference, config templates, and glossary.
