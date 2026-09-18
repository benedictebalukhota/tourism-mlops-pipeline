"""
hosting.py
------------------------------------------------------------------
STEP 5 of the MLOps pipeline: HOSTING.

Pushes all deployment files (app.py, requirements.txt) into a Hugging Face
Space configured to run with the Gradio SDK. Creating the Space if it does
not exist makes the script safe to re-run on every CI push.

NOTE ON SDK CHOICE: as of mid-2026, Hugging Face requires a paid PRO plan
to create Docker or Streamlit-SDK Spaces on free personal accounts; a
Gradio SDK Space on the free ZeroGPU tier remains free (up to 2 per
account). This script therefore deploys the Gradio app (`app.py`) rather
than the Docker/Streamlit path. The `Dockerfile` is still versioned in
this folder to document how the app would be containerised on a paid plan,
but it is intentionally not uploaded to the Space itself, since a Gradio
SDK Space ignores it and builds directly from `app.py` + `requirements.txt`.

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
SPACE_SDK = "gradio"

# As of mid-2026, Gradio Spaces on the DEFAULT "cpu-basic" hardware require a
# paid plan on free personal accounts. The "zero-a10g" (ZeroGPU) hardware
# tier is the specific one free accounts can still use at no cost (up to 2
# Spaces per account, requiring a verified email and an account older than
# 30 days). This app does no GPU work itself - it simply runs on whatever
# CPU backs the ZeroGPU allocation when idle - so requesting this hardware
# is purely to stay within the free tier, not because a GPU is needed.
SPACE_HARDWARE = "zero-a10g"

DEPLOYMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Files that make up the deployable application.
# Dockerfile is deliberately excluded - see module docstring above.
FILES_TO_UPLOAD = ["app.py", "requirements.txt"]


def build_readme() -> str:
    """
    Generate the README.md that Hugging Face Spaces requires.

    The YAML front-matter tells the Space which SDK to use; without it the
    Space will not start.
    """
    return f"""---
title: Wellness Tourism Package Predictor
emoji: 🌴
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
suggested_hardware: zero-a10g
pinned: false
---

# Wellness Tourism Package Predictor

Predicts whether a customer is likely to purchase the Wellness Tourism
Package offered by "Visit with Us".

- **Model:** XGBoost classifier inside a scikit-learn preprocessing pipeline
- **Model hub:** [{HF_USERNAME}/tourism-wellness-model](https://huggingface.co/{HF_USERNAME}/tourism-wellness-model)
- **Dataset:** [{HF_USERNAME}/tourism-wellness-dataset](https://huggingface.co/datasets/{HF_USERNAME}/tourism-wellness-dataset)

Built with a Gradio interface running on Hugging Face's free tier.
A Dockerfile is included in the source repository for reference, showing
how this app would be containerised under a paid Hugging Face plan.

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
        print(f"Creating Space: {SPACE_REPO_ID} (hardware: {SPACE_HARDWARE})")
        create_repo(
            repo_id=SPACE_REPO_ID,
            repo_type=REPO_TYPE,
            space_sdk=SPACE_SDK,
            space_hardware=SPACE_HARDWARE,
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
