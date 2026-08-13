"""CLI interface for AutoML Tabular workflows."""

import datetime
import json
import pathlib
from typing import Any

import click
from google.cloud import aiplatform, storage

from tabflows.config import TabularPipelineConfig
from tabflows.experiments import (
    get_model_evaluation_metrics,
    get_model_feature_attributions,
    list_experiment_runs,
    log_experiment_run,
)
from tabflows.inference import (
    deploy_model_to_endpoint,
    run_batch_prediction,
)
from tabflows.inference import (
    list_models as sdk_list_models,
)
from tabflows.inference import (
    predict_online as sdk_predict_online,
)
from tabflows.pipeline import (
    build_automl_tabular_pipeline,
    build_skip_architecture_search_pipeline,
    generate_fte_transformations,
    setup_gcp_resources,
    write_to_gcs,
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
    """Tabflows CLI for AutoML Tabular classification pipelines & inference."""


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
@click.option(
    "--distill / --no-distill", default=False, help="Enable or disable model distillation"
)
def run_automl(
    project: str | None,
    location: str | None,
    bucket: str | None,
    job_id: str | None,
    compile_only: bool,
    async_mode: bool,
    distill: bool,
) -> None:
    """Build and submit standard AutoML Tabular pipeline."""
    config = _get_config(project, location, bucket)
    config.run_distillation = distill

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


@main.command()
@click.option("--limit", default=5, help="Maximum number of recent models to list")
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
def list_models(
    limit: int,
    project: str | None,
    location: str | None,
) -> None:
    """List recent trained models in Vertex AI Model Registry."""
    config = _get_config(project, location, None)
    if not config.project_id:
        click.echo("Error: project_id must be provided or set in .env")
        raise click.Abort()

    click.echo(f"Fetching up to {limit} recent models for project '{config.project_id}'...")
    models = sdk_list_models(config=config, limit=limit)

    if not models:
        click.echo("No models found in Vertex AI Model Registry.")
        return

    click.echo(f"\nFound {len(models)} model(s):")
    for m in models:
        click.echo(
            f"- Display Name: {m.display_name}\n"
            f"  Resource Name: {m.resource_name}\n"
            f"  Created: {m.create_time}\n"
        )


@main.command()
@click.option("--model", required=True, help="Model resource name or ID")
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
def get_evaluation(
    model: str,
    project: str | None,
    location: str | None,
) -> None:
    """Retrieve evaluation metrics and feature attributions for a trained model."""
    config = _get_config(project, location, None)

    click.echo(f"Fetching evaluation metrics for model '{model}'...")
    metrics = get_model_evaluation_metrics(model=model, config=config)
    attributions = get_model_feature_attributions(model=model, config=config)

    if not metrics:
        click.echo("No evaluation metrics found for specified model.")
        return

    click.echo("\n--- Evaluation Metrics ---")
    click.echo(json.dumps(metrics, indent=2))

    if attributions:
        click.echo("\n--- Feature Attributions ---")
        click.echo(json.dumps(attributions, indent=2))


@main.command()
@click.option(
    "--experiment", default=None, help="Experiment name (defaults to .env/EXPERIMENT_NAME)"
)
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
def list_experiments(
    experiment: str | None,
    project: str | None,
    location: str | None,
) -> None:
    """List pipeline runs and metrics logged to a Vertex AI Experiment."""
    config = _get_config(project, location, None)
    exp_name = experiment or config.experiment_name

    click.echo(f"Fetching experiment runs for '{exp_name}' in project '{config.project_id}'...")
    df = list_experiment_runs(experiment_name=exp_name, config=config)

    if df is None or df.empty:
        click.echo(f"No runs found for experiment '{exp_name}'.")
        return

    click.echo(f"\nFound {len(df)} experiment run(s):")
    click.echo(df.to_string())


@main.command()
@click.option("--run-name", required=True, help="Experiment run name")
@click.option("--model-resource-name", default=None, help="Model resource name or ID")
@click.option("--pipeline-job-id", default=None, help="Pipeline Job ID or resource name")
@click.option("--params-json", default=None, help="JSON string of parameter key-value pairs")
@click.option("--metrics-json", default=None, help="JSON string of metric key-value pairs")
@click.option(
    "--experiment", default=None, help="Experiment name (defaults to .env/EXPERIMENT_NAME)"
)
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
def log_experiment(
    run_name: str,
    model_resource_name: str | None,
    pipeline_job_id: str | None,
    params_json: str | None,
    metrics_json: str | None,
    experiment: str | None,
    project: str | None,
    location: str | None,
) -> None:
    """Log an experiment run with parameters, metrics, model, and pipeline job info."""
    config = _get_config(project, location, None)
    if experiment:
        config.experiment_name = experiment

    params: dict[str, Any] | None = None
    if params_json:
        try:
            parsed = json.loads(params_json)
            if not isinstance(parsed, dict):
                click.echo("Error: --params-json must be a JSON object.")
                raise click.Abort()
            params = parsed
        except json.JSONDecodeError as e:
            click.echo(f"Error parsing --params-json JSON: {e}")
            raise click.Abort() from e

    metrics: dict[str, Any] | None = None
    if metrics_json:
        try:
            parsed = json.loads(metrics_json)
            if not isinstance(parsed, dict):
                click.echo("Error: --metrics-json must be a JSON object.")
                raise click.Abort()
            metrics = parsed
        except json.JSONDecodeError as e:
            click.echo(f"Error parsing --metrics-json JSON: {e}")
            raise click.Abort() from e

    log_experiment_run(
        run_name=run_name,
        params=params,
        metrics=metrics,
        model=model_resource_name,
        pipeline_job=pipeline_job_id,
        config=config,
    )
    click.echo(
        f"Successfully logged experiment run '{run_name}' to experiment '{config.experiment_name}'."
    )


@main.command()
@click.option("--model", required=True, help="Model resource name or ID")
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
@click.option("--bucket", default=None, help="GCS Bucket URI (defaults to .env)")
@click.option("--machine-type", default="n1-standard-4", help="Serving machine type")
@click.option("--endpoint-name", default=None, help="Endpoint display name")
@click.option(
    "--async-mode",
    is_flag=True,
    help="Deploy endpoint asynchronously without blocking",
)
def deploy_endpoint(
    model: str,
    project: str | None,
    location: str | None,
    bucket: str | None,
    machine_type: str,
    endpoint_name: str | None,
    async_mode: bool,
) -> None:
    """Deploy an AutoML Tabular model to a real-time Vertex AI Endpoint."""
    config = _get_config(project, location, bucket)
    config.serving_machine_type = machine_type

    click.echo(f"Deploying model '{model}' to endpoint...")
    endpoint = deploy_model_to_endpoint(
        model=model,
        config=config,
        endpoint_display_name=endpoint_name,
        sync=not async_mode,
    )
    click.echo(f"Endpoint deployment initiated: {endpoint.resource_name}")


@main.command()
@click.option("--endpoint", required=True, help="Endpoint resource name or ID")
@click.option(
    "--json-instance",
    required=True,
    help='JSON string of feature key-value pairs (e.g. \'{"age": "30", ...}\')',
)
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
def predict_online(
    endpoint: str,
    json_instance: str,
    project: str | None,
    location: str | None,
) -> None:
    """Execute real-time online inference against a deployed Vertex AI Endpoint."""
    config = _get_config(project, location, None)
    if config.project_id:
        aiplatform.init(project=config.project_id, location=config.location)

    try:
        parsed = json.loads(json_instance)
        instances = [parsed] if isinstance(parsed, dict) else parsed
    except Exception as e:
        click.echo(f"Error parsing --json-instance JSON: {e}")
        raise click.Abort() from e

    click.echo(f"Sending online prediction request to endpoint '{endpoint}'...")
    results = sdk_predict_online(endpoint=endpoint, instances=instances)
    click.echo(f"Predictions: {json.dumps(results, indent=2)}")


@main.command()
@click.option("--model", required=True, help="Model resource name or ID")
@click.option("--gcs-source", required=True, help="GCS source URI (gs://...) for input data")
@click.option(
    "--gcs-dest", default=None, help="GCS destination prefix (gs://...) for output predictions"
)
@click.option("--project", default=None, help="GCP Project ID (defaults to .env)")
@click.option("--location", default=None, help="GCP Region (defaults to .env)")
@click.option("--bucket", default=None, help="GCS Bucket URI (defaults to .env)")
def run_batch_predict(
    model: str,
    gcs_source: str,
    gcs_dest: str | None,
    project: str | None,
    location: str | None,
    bucket: str | None,
) -> None:
    """Submit a Batch Prediction job for an AutoML Tabular model."""
    config = _get_config(project, location, bucket)

    click.echo(f"Submitting batch prediction job for model '{model}'...")
    batch_job = run_batch_prediction(
        model=model,
        config=config,
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_dest,
    )
    click.echo(f"Batch prediction job created successfully: {batch_job.resource_name}")


@main.command()
@click.option(
    "--output-path",
    required=True,
    help="Path or GCS URI (gs://...) to write FTE config JSON",
)
@click.option(
    "--columns-json",
    required=True,
    help="JSON string mapping column names to transformation types",
)
@click.option("--bucket", default=None, help="GCS Bucket URI (defaults to .env)")
def generate_fte_config(
    output_path: str,
    columns_json: str,
    bucket: str | None,
) -> None:
    """Generate Feature Transform Engine (FTE) transformations configuration JSON."""
    try:
        column_types = json.loads(columns_json)
    except json.JSONDecodeError as e:
        click.echo(f"Error parsing --columns-json JSON: {e}")
        raise click.Abort() from e

    if not isinstance(column_types, dict):
        click.echo("Error: --columns-json must be a JSON object mapping column names to types.")
        raise click.Abort()

    try:
        transformations = generate_fte_transformations(column_types)
    except ValueError as e:
        click.echo(f"Error generating FTE transformations: {e}")
        raise click.Abort() from e

    content = json.dumps(transformations, indent=2)

    if output_path.startswith("gs://"):
        config = _get_config(None, None, bucket)
        storage_client = storage.Client(project=config.project_id or None)
        write_to_gcs(storage_client, output_path, content)
        click.echo(f"FTE configuration uploaded to GCS: {output_path}")
    else:
        path = pathlib.Path(output_path)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        click.echo(f"FTE configuration written to: {output_path}")


if __name__ == "__main__":
    main()
