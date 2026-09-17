"""
prep.py
------------------------------------------------------------------
STEP 2 of the MLOps pipeline: DATA PREPARATION.

1. Loads the raw dataset directly from the Hugging Face dataset space.
2. Cleans it (drops identifier columns, fixes a known category typo).
3. Splits it into stratified train / test sets and saves them locally.
4. Uploads the resulting train/test files back to the same HF dataset space.

Run locally:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
    python tourism_project/model_building/prep.py
"""

import os
import sys

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "<your-hf-username>")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
REPO_TYPE = "dataset"

RAW_FILE = "tourism.csv"
TARGET = "ProdTaken"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Local output directory for the split files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_raw_from_hub(token: str) -> pd.DataFrame:
    """Download the registered raw CSV from the HF dataset space."""
    print(f"Loading raw data from HF dataset space: {DATASET_REPO_ID}")
    local_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=RAW_FILE,
        repo_type=REPO_TYPE,
        token=token,
    )
    # The raw export carries an unnamed pandas index column, so index_col=0
    # prevents it being read in as a spurious feature.
    df = pd.read_csv(local_path, index_col=0)
    print(f"Loaded raw data with shape {df.shape}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data cleaning steps and drop unnecessary columns."""
    df = df.copy()

    # ------------------------------------------------------------------
    # CustomerID is a unique identifier (one distinct value per row). It
    # carries no predictive signal and would only add noise, so we drop it.
    # ------------------------------------------------------------------
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
        print("Dropped identifier column: CustomerID")

    # ------------------------------------------------------------------
    # 'Fe Male' is a data-entry variant of 'Female'. Left unmerged it would
    # be one-hot encoded as a separate, meaningless third gender category.
    # ------------------------------------------------------------------
    if "Gender" in df.columns:
        n_before = (df["Gender"] == "Fe Male").sum()
        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
        print(f"Normalised {n_before} 'Fe Male' values to 'Female'")

    # Remove exact duplicate rows if any exist
    n_dupes = df.duplicated().sum()
    if n_dupes:
        df = df.drop_duplicates()
        print(f"Dropped {n_dupes} duplicate rows")

    return df


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN environment variable is not set.")

    # ---------------- Load + clean ----------------
    df = load_raw_from_hub(token)
    df = clean(df)

    # ---------------- Split ----------------
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # preserves the ~19% positive rate in both splits
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    print(f"\nTrain shape: {train_df.shape} | positive rate: {y_train.mean():.4f}")
    print(f"Test  shape: {test_df.shape} | positive rate: {y_test.mean():.4f}")

    # ---------------- Save locally ----------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nSaved locally:\n  {train_path}\n  {test_path}")

    # ---------------- Upload back to HF ----------------
    api = HfApi(token=token)
    for local_path, repo_path in [(train_path, "train.csv"), (test_path, "test.csv")]:
        print(f"Uploading {repo_path} -> {DATASET_REPO_ID}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=DATASET_REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Upload prepared {repo_path}",
        )

    print("\nData preparation complete.")
    print(f"View at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
