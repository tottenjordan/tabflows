"""CLI interface for AutoML Tabular workflows."""

import datetime
from typing import Any

import click
from google.cloud import aiplatform, storage

from tabflows.config import TabularPipelineConfig
from tabflows.pipeline import (
    build_automl_tabular_pipeline,
    build_skip_architecture_search_pipeline,
    setup_gcp_resources,
)


def _get_config(
    project: str | None, location: str | None, bucket: str | None
) -> TabularPipelineConfig:
    kwargs: dict[str, Any] = {}
    if project:
        kwargs["project_id"] = project
    if location:
        kwargs["location"] = location
    if bucket:
        kwargs["bucket_uri"] = bucket
    return TabularPipelineConfig(**kwargs)


@click.group()
def main() -> None:
    """Tabflows CLI for AutoML Tabular classification pipelines."""


@main.command()
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
@click.option("--bucket", default=None, help="GCS Bucket URI (defaults to .env)")
def setup(project: str | None, location: str | None, bucket: str | None) -> None:
    """Provision GCS bucket and upload transform configuration JSON asset."""
    config = _get_config(project, location, bucket)

    if not config.project_id or not config.bucket_uri:
        click.echo("Error: Both project_id and bucket_uri must be provided or set in .env")
        raise click.Abort()

    click.echo(f"Setting up GCP resources for project '{config.project_id}'...")
    storage_client = storage.Client(project=config.project_id)
    summary = setup_gcp_resources(config, storage_client=storage_client)

    click.echo(f"GCS Bucket: {summary['bucket_uri']} (created: {summary['bucket_created']})")
    click.echo(f"Transform config uploaded to: {summary['transform_config_path']}")
    click.echo("Setup completed successfully.")


@main.command()
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
@click.option("--bucket", default=None, help="GCS Bucket URI (defaults to .env)")
@click.option("--job-id", default=None, help="Pipeline Job ID")
@click.option("--compile-only", is_flag=True, help="Only compile template without submitting job")
@click.option(
    "--async-mode",
    is_flag=True,
    help="Submit job asynchronously to Vertex AI without blocking",
)
def run_automl(
    project: str | None,
    location: str | None,
    bucket: str | None,
    job_id: str | None,
    compile_only: bool,
    async_mode: bool,
) -> None:
    """Build and submit standard AutoML Tabular pipeline."""
    config = _get_config(project, location, bucket)

    if not config.project_id or not config.bucket_uri:
        click.echo("Error: Both project_id and bucket_uri must be provided or set in .env")
        raise click.Abort()

    if not job_id:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        job_id = f"automl-tabular-{timestamp}"

    click.echo(
        f"Building AutoML Tabular Pipeline for project {config.project_id} in {config.location}..."
    )
    template_path, parameter_values = build_automl_tabular_pipeline(config)
    click.echo(f"Template compiled at: {template_path}")

    if compile_only:
        click.echo("Compile-only mode requested. Skipping job submission.")
        return

    aiplatform.init(project=config.project_id, location=config.location)
    job = aiplatform.PipelineJob(
        display_name=job_id,
        location=config.location,
        template_path=template_path,
        job_id=job_id,
        pipeline_root=config.root_dir,
        parameter_values=parameter_values,
        enable_caching=False,
    )

    if async_mode:
        click.echo(f"Submitting pipeline job '{job_id}' asynchronously to Vertex AI...")
        job.submit()
        click.echo("Pipeline job submitted successfully.")
    else:
        click.echo(f"Submitting pipeline job '{job_id}' to Vertex AI...")
        job.run()


@main.command()
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
@click.option("--bucket", default=None, help="GCS Bucket URI (defaults to .env)")
@click.option(
    "--tuning-artifact-uri",
    required=True,
    help="URI of tuning_result_output artifact from Stage 1",
)
@click.option("--job-id", default=None, help="Pipeline Job ID")
@click.option("--compile-only", is_flag=True, help="Only compile template without submitting job")
@click.option(
    "--async-mode",
    is_flag=True,
    help="Submit job asynchronously to Vertex AI without blocking",
)
def run_skip_search(
    project: str | None,
    location: str | None,
    bucket: str | None,
    tuning_artifact_uri: str,
    job_id: str | None,
    compile_only: bool,
    async_mode: bool,
) -> None:
    """Build and submit Skip Architecture Search AutoML Tabular pipeline."""
    config = _get_config(project, location, bucket)

    if not config.project_id or not config.bucket_uri:
        click.echo("Error: Both project_id and bucket_uri must be provided or set in .env")
        raise click.Abort()

    if not job_id:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        job_id = f"automl-skip-search-{timestamp}"

    click.echo(f"Building Skip Architecture Search Pipeline for project {config.project_id}...")
    template_path, parameter_values = build_skip_architecture_search_pipeline(
        config, tuning_artifact_uri
    )
    click.echo(f"Template compiled at: {template_path}")

    if compile_only:
        click.echo("Compile-only mode requested. Skipping job submission.")
        return

    aiplatform.init(project=config.project_id, location=config.location)
    job = aiplatform.PipelineJob(
        display_name=job_id,
        location=config.location,
        template_path=template_path,
        job_id=job_id,
        pipeline_root=config.root_dir,
        parameter_values=parameter_values,
        enable_caching=False,
    )

    if async_mode:
        click.echo(f"Submitting pipeline job '{job_id}' asynchronously to Vertex AI...")
        job.submit()
        click.echo("Pipeline job submitted successfully.")
    else:
        click.echo(f"Submitting pipeline job '{job_id}' to Vertex AI...")
        job.run()


if __name__ == "__main__":
    main()
