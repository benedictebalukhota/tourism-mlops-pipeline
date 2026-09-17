"""
hosting.py
------------------------------------------------------------------
STEP 5 of the MLOps pipeline: HOSTING.

Pushes all deployment files (app.py, Dockerfile, requirements.txt) into a
Hugging Face Space configured to run as a Docker container. Creating the
Space if it does not exist makes the script safe to re-run on every CI push.

Run locally:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
    python tourism_project/deployment/hosting.py
"""

import os
import sys

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "<your-hf-username>")
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-wellness-app"
REPO_TYPE = "space"
SPACE_SDK = "docker"

DEPLOYMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Files that make up the deployable application
FILES_TO_UPLOAD = ["app.py", "Dockerfile", "requirements.txt"]


def build_readme() -> str:
    """
    Generate the README.md that Hugging Face Spaces requires.

    The YAML front-matter tells the Space which SDK to use and which port
    the container listens on; without it the Space will not start.
    """
    return f"""---
title: Wellness Tourism Package Predictor
emoji: 🌴
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Wellness Tourism Package Predictor

Predicts whether a customer is likely to purchase the Wellness Tourism
Package offered by "Visit with Us".

- **Model:** XGBoost classifier inside a scikit-learn preprocessing pipeline
- **Model hub:** [{HF_USERNAME}/tourism-wellness-model](https://huggingface.co/{HF_USERNAME}/tourism-wellness-model)
- **Dataset:** [{HF_USERNAME}/tourism-wellness-dataset](https://huggingface.co/datasets/{HF_USERNAME}/tourism-wellness-dataset)

This Space is deployed automatically by a GitHub Actions MLOps pipeline.
"""


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN environment variable is not set.")

    api = HfApi(token=token)

    # ------------------------------------------------------------------
    # Create the Space if it does not already exist.
    # ------------------------------------------------------------------
    try:
        api.repo_info(repo_id=SPACE_REPO_ID, repo_type=REPO_TYPE)
        print(f"Space already exists: {SPACE_REPO_ID}")
    except RepositoryNotFoundError:
        print(f"Creating Space: {SPACE_REPO_ID}")
        create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type=REPO_TYPE,
            space_sdk=SPACE_SDK,
            private=False,
            token=token,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Write and upload the Space README (required front-matter).
    # ------------------------------------------------------------------
    readme_path = os.path.join(DEPLOYMENT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(build_readme())

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=SPACE_REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Update Space README",
    )
    print("Uploaded README.md")

    # ------------------------------------------------------------------
    # Upload each deployment file.
    # ------------------------------------------------------------------
    for filename in FILES_TO_UPLOAD:
        local_path = os.path.join(DEPLOYMENT_DIR, filename)
        if not os.path.exists(local_path):
            sys.exit(f"ERROR: expected deployment file not found: {local_path}")

        print(f"Uploading {filename} -> {SPACE_REPO_ID}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=SPACE_REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Deploy {filename}",
        )

    print("\nDeployment complete.")
    print(f"Space URL: https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
