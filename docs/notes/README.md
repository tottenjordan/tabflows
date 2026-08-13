# Session & Topic Notes Index

This directory contains persistent notes on architecture, libraries, environment quirks, and non-obvious technical learnings accumulated across sessions.

## Topic Notes
- [Tabular Workflows](file:///usr/local/google/home/jordantotten/antigravity/tabflows/docs/notes/tabular_workflows.md): Architecture, component task names, GCP v1 pipeline APIs, skip-architecture search, and transformation configs.
- [Feature Transform Engine](file:///usr/local/google/home/jordantotten/antigravity/tabflows/docs/notes/feature_transform_engine.md): Feature Transform Engine (FTE) column types (`categorical`, `numeric`, `timestamp`, `text_embedding`, `auto`), Python SDK functions, and CLI usage.
- [Skip Architecture Search & Benchmarks](file:///usr/local/google/home/jordantotten/antigravity/tabflows/docs/notes/skip_architecture_search_benchmarks.md): Stage 1 vs Stage 2 performance benchmarks (-79.6% execution time, -80.4% node-hour savings), node-hour metrics, and `tuning_result_output` artifact reuse.
- [Inference & Serving](file:///usr/local/google/home/jordantotten/antigravity/tabflows/docs/notes/inference.md): Online real-time endpoint deployment, JSON instance prediction, batch inference execution, and endpoint resource cleanup.
- [Evaluation & Experiment Tracking](file:///usr/local/google/home/jordantotten/antigravity/tabflows/docs/notes/evaluation_and_experiments.md): Model evaluation metrics (Log Loss, AUC-ROC, confusion matrix), feature importance attributions, and Vertex AI Experiments run tracking.
- [Tooling & Environment Management](file:///usr/local/google/home/jordantotten/antigravity/tabflows/docs/notes/tooling.md): Package resolution with `uv`, PyPI default index configuration, `ruff`, `ty`, and `pytest` integration.

## Key Repository Files
- [CODE_STANDARDS.md](file:///usr/local/google/home/jordantotten/antigravity/tabflows/CODE_STANDARDS.md): Mandatory coding standards and project rules.
- [GEMINI.md](file:///usr/local/google/home/jordantotten/antigravity/tabflows/GEMINI.md): Agent context document.
- [pyproject.toml](file:///usr/local/google/home/jordantotten/antigravity/tabflows/pyproject.toml): Build system and dependency specifications.
