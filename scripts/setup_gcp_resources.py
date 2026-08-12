#!/usr/bin/env python3
"""Setup script for provisioning Cloud Storage bucket and uploading pipeline assets."""

import sys

from google.cloud import storage

from tabflows.config import TabularPipelineConfig
from tabflows.pipeline import setup_gcp_resources


def main() -> None:
    """Execute GCP resource setup using environment settings."""
    config = TabularPipelineConfig()

    if not config.project_id or not config.bucket_uri:
        print("Error: GCP_PROJECT and GCP_BUCKET_URI must be configured in .env or environment.")
        sys.exit(1)

    print(f"Setting up GCP resources for project '{config.project_id}'...")
    print(f"Region: {config.location}")
    print(f"Target Bucket: {config.bucket_uri}")

    try:
        storage_client = storage.Client(project=config.project_id)
        summary = setup_gcp_resources(config, storage_client=storage_client)

        print("\n--- Asset Setup Summary ---")
        print(f"Bucket Name:           {summary['bucket_name']}")
        print(f"Bucket URI:            {summary['bucket_uri']}")
        print(f"Bucket Newly Created:  {summary['bucket_created']}")
        print(f"Transform Config Path: {summary['transform_config_path']}")
        print("\nSetup completed successfully!")
    except Exception as e:
        print(f"\nError setting up GCP resources: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
