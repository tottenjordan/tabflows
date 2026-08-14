#!/usr/bin/env python3
"""Script to compile and verify Vertex AI AutoML Tabular pipeline templates for CI/CD."""

import os
import sys

from tabflows.config import TabularPipelineConfig
from tabflows.pipeline import (
    build_automl_tabular_pipeline,
    build_skip_architecture_search_pipeline,
    create_tabular_pipeline_job,
    run_skip_architecture_search_pipeline,
)


def compile_pipelines() -> None:
    """Compile and verify AutoML Tabular pipeline job templates."""
    print("Initializing mock TabularPipelineConfig for CI compilation...")
    config = TabularPipelineConfig(
        project_id="ci-test-project",
        bucket_uri="gs://ci-test-bucket",
        tuning_result_output="gs://ci-test-bucket/automl_tabular_pipeline/tuning_result",
    )

    print("Compiling Stage 1 Full Search pipeline job...")
    job1 = create_tabular_pipeline_job(config, job_id="ci-full-search", log_experiment=False)
    template_path1, _ = build_automl_tabular_pipeline(config)

    print("Compiling Stage 2 Skip Architecture Search pipeline job...")
    job2 = run_skip_architecture_search_pipeline(
        config,
        tuning_result_artifact_uri="gs://ci-test-bucket/automl_tabular_pipeline/tuning_result",
        job_id="ci-skip-search",
        log_experiment=False,
    )
    template_path2, _ = build_skip_architecture_search_pipeline(
        config,
        stage_1_tuning_result_artifact_uri="gs://ci-test-bucket/automl_tabular_pipeline/tuning_result",
    )

    print(f"Stage 1 Pipeline Job created: {type(job1).__name__}")
    print(f"Stage 1 Pipeline Template Path: {template_path1}")
    print(f"Stage 2 Pipeline Job created: {type(job2).__name__}")
    print(f"Stage 2 Pipeline Template Path: {template_path2}")

    if not os.path.exists(template_path1):
        print(f"ERROR: Template path does not exist: {template_path1}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(template_path2):
        print(f"ERROR: Template path does not exist: {template_path2}", file=sys.stderr)
        sys.exit(1)

    print("\nSUCCESS: All AutoML Tabular pipeline templates compiled and verified successfully.")


def main() -> None:
    """Execute pipeline compilation."""
    compile_pipelines()


if __name__ == "__main__":
    main()
