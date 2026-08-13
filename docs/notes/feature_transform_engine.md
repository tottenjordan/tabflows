# Topic Note: Feature Transform Engine (FTE)

## Overview
The Feature Transform Engine (FTE) in Vertex AI Tabular Workflows provides column-level feature engineering and data preprocessing for tabular datasets prior to model training. It supports both automated transformations (`"auto"`) and explicit custom transformation specifications defined in JSON.

## Supported Column Transform Types
FTE supports five primary transformation types:

1. **`categorical`**: Generates categorical encodings (one-hot encoding, target encoding, or vocabulary index lookups). Maps to `{"categorical": {"column_name": "col"}}` in JSON config.
2. **`numeric`**: Standardizes numerical features using normalization and scaling (e.g. z-score, min-max scaling). Maps to `{"numeric": {"column_name": "col"}}` in JSON config.
3. **`timestamp`**: Extracts datetime components such as year, month, day, hour, day of week, and time since epoch. Maps to `{"timestamp": {"column_name": "col"}}` in JSON config.
4. **`text_embedding`**: Embeds free-form text columns using pre-trained language models or bag-of-words. Specified as `"text"` in SDK input and maps to `{"text_embedding": {"column_name": "col"}}` in JSON config.
5. **`auto`**: Allows Vertex AI to inspect column data types and statistics to automatically select the optimal transformation strategy. Maps to `{"auto": {"column_name": "col"}}` in JSON config.

## Python SDK Usage
The `tabflows.pipeline` module exports helper functions to construct and persist FTE transformation dictionaries:

- `generate_fte_transformations(column_types: dict[str, str]) -> list[dict[str, Any]]`:
  Accepts a dictionary mapping column names to type strings (`"categorical"`, `"numeric"`, `"timestamp"`, `"text"`, `"auto"`).

  ```python
  from tabflows import generate_fte_transformations

  fte_types = {
      "age": "numeric",
      "job": "categorical",
      "signup_date": "timestamp",
      "comments": "text",
  }
  transformations = generate_fte_transformations(fte_types)
  ```

- `write_fte_transformations(storage_client, uri: str, column_types: dict[str, str]) -> None`:
  Generates FTE transformation JSON and uploads directly to a Google Cloud Storage URI (`gs://bucket/path/fte_config.json`).

## CLI Usage
The `tabflows` CLI provides the `generate-fte-config` command to generate and save FTE configuration files locally or directly to GCS:

```bash
# Save FTE JSON config locally
uv run tabflows generate-fte-config \
    --output-path "./transform_config_fte.json" \
    --columns-json '{"age": "numeric", "job": "categorical", "signup_date": "timestamp", "notes": "text"}'

# Upload FTE JSON config directly to GCS
uv run tabflows generate-fte-config \
    --output-path "gs://my-bucket/config/transform_config_fte.json" \
    --columns-json '{"age": "numeric", "job": "categorical"}'
```
