"""CLI interface for AutoML Tabular workflows."""

import click
from google.cloud import aiplatform

from tabflows.config import TabularPipelineConfig
from tabflows.pipeline import (
    build_automl_tabular_pipeline,
    build_skip_architecture_search_pipeline,
)


@click.group()
def main() -> None:
    """Tabflows CLI for AutoML Tabular classification pipelines."""


@main.command()
@click.option("--project", required=True, help="GCP Project ID")
@click.option("--location", default="us-central1", help="GCP Region")
@click.option("--bucket", required=True, help="GCS Bucket URI (gs://...)")
@click.option("--job-id", default="automl-tabular-run", help="Pipeline Job ID")
@click.option("--compile-only", is_flag=True, help="Only compile template without submitting job")
def run_automl(project: str, location: str, bucket: str, job_id: str, compile_only: bool) -> None:
    """Build and submit standard AutoML Tabular pipeline."""
    config = TabularPipelineConfig(project_id=project, location=location, bucket_uri=bucket)

    click.echo(f"Building AutoML Tabular Pipeline for project {project} in {location}...")
    template_path, parameter_values = build_automl_tabular_pipeline(config)
    click.echo(f"Template compiled at: {template_path}")

    if compile_only:
        click.echo("Compile-only mode requested. Skipping job submission.")
        return

    aiplatform.init(project=project, location=location)
    job = aiplatform.PipelineJob(
        display_name=job_id,
        location=location,
        template_path=template_path,
        job_id=job_id,
        pipeline_root=config.root_dir,
        parameter_values=parameter_values,
        enable_caching=False,
    )
    click.echo("Submitting pipeline job to Vertex AI...")
    job.run()


@main.command()
@click.option("--project", required=True, help="GCP Project ID")
@click.option("--location", default="us-central1", help="GCP Region")
@click.option("--bucket", required=True, help="GCS Bucket URI (gs://...)")
@click.option(
    "--tuning-artifact-uri",
    required=True,
    help="URI of tuning_result_output artifact from Stage 1",
)
@click.option("--job-id", default="automl-tabular-skip-search-run", help="Pipeline Job ID")
@click.option("--compile-only", is_flag=True, help="Only compile template without submitting job")
def run_skip_search(
    project: str,
    location: str,
    bucket: str,
    tuning_artifact_uri: str,
    job_id: str,
    compile_only: bool,
) -> None:
    """Build and submit Skip Architecture Search AutoML Tabular pipeline."""
    config = TabularPipelineConfig(project_id=project, location=location, bucket_uri=bucket)

    click.echo(f"Building Skip Architecture Search Pipeline for project {project}...")
    template_path, parameter_values = build_skip_architecture_search_pipeline(
        config, tuning_artifact_uri
    )
    click.echo(f"Template compiled at: {template_path}")

    if compile_only:
        click.echo("Compile-only mode requested. Skipping job submission.")
        return

    aiplatform.init(project=project, location=location)
    job = aiplatform.PipelineJob(
        display_name=job_id,
        location=location,
        template_path=template_path,
        job_id=job_id,
        pipeline_root=config.root_dir,
        parameter_values=parameter_values,
        enable_caching=False,
    )
    click.echo("Submitting pipeline job to Vertex AI...")
    job.run()


if __name__ == "__main__":
    main()
