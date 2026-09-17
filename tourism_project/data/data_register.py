"""
data_register.py
------------------------------------------------------------------
STEP 1 of the MLOps pipeline: DATA REGISTRATION.

Registers the raw `tourism.csv` dataset on the Hugging Face Hub as a
dataset repository so that every downstream stage (preparation,
training, deployment) pulls from one single, versioned source of truth
instead of a local file.

Run locally:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
    python tourism_project/data/data_register.py

In CI this is executed by .github/workflows/pipeline.yml with the
HF_TOKEN provided through GitHub repository secrets.
"""

import os
import sys

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# NOTE: replace <your-hf-username> with your actual Hugging Face username.
HF_USERNAME = os.getenv("HF_USERNAME", "<your-hf-username>")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
REPO_TYPE = "dataset"

# Path to the raw file that will be registered
LOCAL_FILE_PATH = os.path.join(os.path.dirname(__file__), "tourism.csv")
PATH_IN_REPO = "tourism.csv"


def main() -> None:
    """Create the dataset repo (if needed) and upload the raw CSV."""
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN environment variable is not set.")

    if not os.path.exists(LOCAL_FILE_PATH):
        sys.exit(f"ERROR: raw data file not found at {LOCAL_FILE_PATH}")

    api = HfApi(token=token)

    # ------------------------------------------------------------------
    # Create the dataset repository if it does not already exist.
    # `exist_ok=True` makes this step idempotent, which matters because
    # CI re-runs this script on every push to main.
    # ------------------------------------------------------------------
    try:
        api.repo_info(repo_id=DATASET_REPO_ID, repo_type=REPO_TYPE)
        print(f"Dataset repo already exists: {DATASET_REPO_ID}")
    except RepositoryNotFoundError:
        print(f"Creating dataset repo: {DATASET_REPO_ID}")
        create_repo(
            repo_id=DATASET_REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            token=token,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Upload the raw dataset file.
    # ------------------------------------------------------------------
    print(f"Uploading {LOCAL_FILE_PATH} -> {DATASET_REPO_ID}/{PATH_IN_REPO}")
    api.upload_file(
        path_or_fileobj=LOCAL_FILE_PATH,
        path_in_repo=PATH_IN_REPO,
        repo_id=DATASET_REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Register raw tourism dataset",
    )

    print("\nData registration complete.")
    print(f"View at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
