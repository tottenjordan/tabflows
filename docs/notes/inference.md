# Topic Note: Vertex AI AutoML Tabular Inference (Online & Batch)

This note captures model serving patterns, online prediction payload specs, and batch prediction execution details for AutoML Tabular models in `tabflows`.

---

## 1. Online Inference (Endpoints)

- **SDK Functions**:
  - `deploy_model_to_endpoint(model, config)`: Deploys model to a real-time `aiplatform.Endpoint`.
  - `predict_online(endpoint, instances)`: Sends synchronous JSON feature requests.
  - `cleanup_endpoint(endpoint)`: Undeploys all deployed models and deletes the endpoint resource to avoid unnecessary serving charges.
- **CLI Commands**:
  - `uv run tabflows deploy-endpoint --model <MODEL_ID_OR_URI>`
  - `uv run tabflows predict-online --endpoint <ENDPOINT_ID> --json-instance '{"age": "30", "job": "blue-collar"}'`
- **Machine Type**: Default serving machine is `n1-standard-4` (`min_replica_count=1`, `max_replica_count=1`).

---

## 2. Batch Inference (Batch Jobs)

- **SDK Function**: `run_batch_prediction(model, config, gcs_source, gcs_destination_prefix)`
- **CLI Command**:
  - `uv run tabflows run-batch-predict --model <MODEL_ID> --gcs-source gs://bucket/test_instances.csv`
- **Supported Formats**:
  - Input: `csv`, `jsonl`, `bigquery`.
  - Output: `jsonl`, `csv`, `bigquery`.
- **Output Storage**: Results are exported asynchronously to GCS under `gs://<bucket>/<root_dir>/batch_predictions/prediction-model-YYYY_MM_DD.../`.
