#!/usr/bin/env python3
"""Bake-off Live Deployment & Latency Benchmarking Script.

Retrieves trained Bake-off models ('bakeoff-baseline-full-search' and
'bakeoff-distilled-student-v2'), deploys them to a Vertex AI Endpoint with a
90/10 Champion-Challenger traffic split, runs online prediction benchmarks,
reports latency percentiles and QPS, and cleans up endpoint resources.
"""

import argparse
import logging
import time
from typing import Any

import numpy as np
from google.cloud import aiplatform

from tabflows.config import TabularPipelineConfig
from tabflows.inference import (
    cleanup_endpoint,
    deploy_model_to_endpoint,
    predict_online,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Calculate latency statistics (p50, p90, p95, qps) from request durations in milliseconds."""
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "qps": 0.0}

    p50 = float(np.percentile(latencies_ms, 50))
    p90 = float(np.percentile(latencies_ms, 90))
    p95 = float(np.percentile(latencies_ms, 95))

    mean_ms = float(np.mean(latencies_ms))
    qps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "p95": round(p95, 2),
        "qps": round(qps, 2),
    }


def get_bakeoff_model(
    model_name: str,
    config: TabularPipelineConfig,
) -> aiplatform.Model:
    """Retrieve Vertex AI Model by display name or resource name."""
    aiplatform.init(project=config.project_id, location=config.location)

    if model_name.startswith("projects/"):
        return aiplatform.Model(model_name=model_name)

    models = aiplatform.Model.list(
        filter=f'display_name="{model_name}"',
        order_by="create_time desc",
    )
    if models:
        return models[0]

    all_models = aiplatform.Model.list(order_by="create_time desc")
    for m in all_models:
        if getattr(m, "display_name", "") == model_name:
            return m

    raise ValueError(f"Bake-off model '{model_name}' not found in Vertex AI Model Registry.")


def benchmark_deployment(
    config: TabularPipelineConfig | None = None,
    num_requests: int = 20,
    baseline_model_name: str = "bakeoff-baseline-full-search",
    student_model_name: str = "bakeoff-distilled-student-v2",
) -> dict[str, float]:
    """Deploy Bake-off Champion & Challenger models, run latency benchmarks, and clean up."""
    if config is None:
        config = TabularPipelineConfig()

    logger.info(
        "Initializing Vertex AI with project=%s, location=%s", config.project_id, config.location
    )

    logger.info("Retrieving Champion baseline model: %s", baseline_model_name)
    try:
        champion_model = get_bakeoff_model(baseline_model_name, config)
    except Exception as err:
        logger.warning(
            "Could not retrieve model '%s': %s. Creating dummy handle.", baseline_model_name, err
        )
        champion_model = aiplatform.Model(model_name=baseline_model_name)

    logger.info("Retrieving Challenger student model: %s", student_model_name)
    try:
        challenger_model = get_bakeoff_model(student_model_name, config)
    except Exception as err:
        logger.warning(
            "Could not retrieve model '%s': %s. Creating dummy handle.", student_model_name, err
        )
        challenger_model = aiplatform.Model(model_name=student_model_name)

    endpoint_display_name = f"bakeoff-latency-benchmark-{int(time.time())}"
    logger.info("Deploying Champion model to endpoint: %s", endpoint_display_name)
    endpoint = deploy_model_to_endpoint(
        model=champion_model,
        config=config,
        endpoint_display_name=endpoint_display_name,
        sync=True,
        log_experiment=True,
    )

    logger.info("Deploying Challenger model with 90/10 traffic split...")
    traffic_split = {"0": 90, "1": 10}
    endpoint = deploy_model_to_endpoint(
        model=challenger_model,
        config=config,
        endpoint=endpoint,
        traffic_split=traffic_split,
        sync=True,
        log_experiment=True,
    )

    sample_instance: dict[str, Any] = {
        "age": "35",
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": "1500",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": "15",
        "month": "may",
        "duration": "250",
        "campaign": "1",
        "pdays": "-1",
        "previous": "0",
        "poutcome": "unknown",
    }

    logger.info("Executing %d online prediction requests for benchmarking...", num_requests)
    latencies_ms: list[float] = []

    for i in range(num_requests):
        t0 = time.perf_counter()
        _ = predict_online(endpoint=endpoint, instances=[sample_instance])
        t1 = time.perf_counter()
        duration_ms = (t1 - t0) * 1000.0
        latencies_ms.append(duration_ms)
        logger.debug("Request %d/%d: %.2f ms", i + 1, num_requests, duration_ms)

    stats = calculate_latency_stats(latencies_ms)

    print("\n" + "=" * 70)
    print("Bake-off Deployment Latency Benchmark Report")
    print("=" * 70)
    print(f"Endpoint Display Name: {endpoint_display_name}")
    print("Traffic Split        : 90% Champion (Baseline) / 10% Challenger (Student)")
    print(f"Total Requests       : {num_requests}")
    print("-" * 70)
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 70)
    print(f"{'p50 Latency (ms)':<25} {stats['p50']:<15.2f}")
    print(f"{'p90 Latency (ms)':<25} {stats['p90']:<15.2f}")
    print(f"{'p95 Latency (ms)':<25} {stats['p95']:<15.2f}")
    print(f"{'Throughput (QPS)':<25} {stats['qps']:<15.2f}")
    print("=" * 70 + "\n")

    logger.info("Cleaning up benchmark endpoint resource...")
    cleanup_endpoint(endpoint=endpoint, delete_endpoint=True)
    logger.info("Benchmark cleanup completed successfully.")

    return stats


def main() -> None:
    """Parse arguments and execute latency benchmark script."""
    parser = argparse.ArgumentParser(description="Bake-off Deployment Latency Benchmarking")
    parser.add_argument(
        "--num-requests", type=int, default=20, help="Number of benchmark requests to send"
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="bakeoff-baseline-full-search",
        help="Champion baseline model display name or resource name",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="bakeoff-distilled-student-v2",
        help="Challenger student model display name or resource name",
    )
    args = parser.parse_args()

    benchmark_deployment(
        num_requests=args.num_requests,
        baseline_model_name=args.baseline_model,
        student_model_name=args.student_model,
    )


if __name__ == "__main__":
    main()
