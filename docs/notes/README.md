# Session & Topic Notes Index

This directory contains persistent notes on architecture, libraries, environment quirks, and non-obvious technical learnings accumulated across sessions.

## Topic Notes
- [Tabular Workflows](tabular_workflows.md): Architecture, component task names, GCP v1 pipeline APIs, skip-architecture search, and transformation configs.
- [Feature Transform Engine](feature_transform_engine.md): Feature Transform Engine (FTE) column types (`categorical`, `numeric`, `timestamp`, `text_embedding`, `auto`), Python SDK functions, and CLI usage.
- [Skip Architecture Search & Benchmarks](skip_architecture_search_benchmarks.md): Stage 1 vs Stage 2 performance benchmarks (-79.6% execution time, -80.4% node-hour savings), node-hour metrics, and `tuning_result_output` artifact reuse.
- [Inference & Serving](inference.md): Online real-time endpoint deployment, JSON instance prediction, batch inference execution, and endpoint resource cleanup.
- [Evaluation & Experiment Tracking](evaluation_and_experiments.md): Model evaluation metrics (Log Loss, AUC-ROC, confusion matrix), feature importance attributions, full-lifecycle experiment tracking across pipelines, DoE campaigns, deployments, batch predictions, and `log-experiment` CLI usage.
- [Tooling & Environment Management](tooling.md): Package resolution with `uv`, PyPI default index configuration, `ruff`, `ty`, and `pytest` integration.

## Key Repository Files
- [CODE_STANDARDS.md](../../CODE_STANDARDS.md): Mandatory coding standards and project rules.
- [GEMINI.md](../../GEMINI.md): Agent context document.
- [pyproject.toml](../../pyproject.toml): Build system and dependency specifications.
