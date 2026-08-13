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
