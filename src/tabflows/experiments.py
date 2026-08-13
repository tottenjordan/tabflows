"""Evaluation and Vertex AI Experiment tracking utilities for AutoML Tabular models."""

from typing import Any

from google.cloud import aiplatform

from tabflows.config import TabularPipelineConfig


def get_model_evaluation_metrics(
    model: str | aiplatform.Model,
    config: TabularPipelineConfig | None = None,
) -> dict[str, Any]:
    """Retrieve evaluation metrics dictionary for a trained Vertex AI Model."""
    if config is None:
        config = TabularPipelineConfig()

    if isinstance(model, str):
        aiplatform.init(project=config.project_id, location=config.location)
        model = aiplatform.Model(model_name=model)

    evaluations = model.list_model_evaluations()
    if not evaluations:
        return {}

    # Return metrics dictionary from primary evaluation
    eval_obj = evaluations[0]
    return dict(eval_obj.metrics)


def get_model_feature_attributions(
    model: str | aiplatform.Model,
    config: TabularPipelineConfig | None = None,
) -> dict[str, float]:
    """Retrieve global feature attributions / importance weights for a trained Vertex AI Model."""
    if config is None:
        config = TabularPipelineConfig()

    if isinstance(model, str):
        aiplatform.init(project=config.project_id, location=config.location)
        model = aiplatform.Model(model_name=model)

    evaluations = model.list_model_evaluations()
    if not evaluations:
        return {}

    eval_obj = evaluations[0]
    metrics = eval_obj.metrics
    if not metrics or "modelExplanation" not in metrics:
        return {}

    explanations = metrics["modelExplanation"]
    mean_attributions = explanations.get("meanAttributions", {}).get("featureAttributions", {})
    return {k: float(v) for k, v in mean_attributions.items()}


def list_experiment_runs(
    experiment_name: str | None = None,
    config: TabularPipelineConfig | None = None,
) -> Any:
    """Retrieve pandas DataFrame of pipeline runs logged to a Vertex AI Experiment."""
    if config is None:
        config = TabularPipelineConfig()

    if not experiment_name:
        experiment_name = config.experiment_name

    aiplatform.init(project=config.project_id, location=config.location)
    return aiplatform.get_experiment_df(experiment=experiment_name)


def log_experiment_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    model: str | aiplatform.Model | None = None,
    pipeline_job: str | aiplatform.PipelineJob | None = None,
    config: TabularPipelineConfig | None = None,
) -> Any:
    """Log an experiment run with parameters, metrics, evaluation, and pipeline job info."""
    if config is None:
        config = TabularPipelineConfig()

    aiplatform.init(
        project=config.project_id,
        location=config.location,
        experiment=config.experiment_name,
    )

    run = aiplatform.start_run(run_name=run_name)

    if params:
        aiplatform.log_params(params)

    if metrics:
        aiplatform.log_metrics(metrics)

    if model is not None:
        model_metrics = get_model_evaluation_metrics(model=model, config=config)
        if model_metrics:
            aiplatform.log_metrics(model_metrics)

    if pipeline_job is not None:
        if isinstance(pipeline_job, str):
            job_resource_name = pipeline_job
        else:
            job_resource_name = getattr(pipeline_job, "resource_name", str(pipeline_job))

        aiplatform.log_params({"pipeline_job_resource_name": job_resource_name})

        if hasattr(run, "_log_pipeline_job") and not isinstance(pipeline_job, str):
            try:
                run._log_pipeline_job(pipeline_job)
            except Exception:
                pass

    return run


def run_doe_campaign(
    campaign_name: str,
    variants: list[dict[str, Any]],
    config: TabularPipelineConfig | None = None,
) -> list[dict[str, Any]]:
    """Execute Design of Experiments (DoE) campaign, logging experiment runs for each variant."""
    if config is None:
        config = TabularPipelineConfig()

    summary: list[dict[str, Any]] = []

    for i, variant in enumerate(variants):
        var_name = variant.get("name", f"variant_{i}")
        if campaign_name and not str(var_name).startswith(f"{campaign_name}-"):
            run_name = f"{campaign_name}-{var_name}"
        else:
            run_name = str(var_name)

        run = log_experiment_run(
            run_name=run_name,
            params=variant,
            config=config,
        )

        summary.append(
            {
                "campaign": campaign_name,
                "variant_name": var_name,
                "run_name": run_name,
                "params": variant,
                "run": run,
            }
        )

    return summary

