"""
train.py
------------------------------------------------------------------
STEP 3 of the MLOps pipeline: MODEL BUILDING WITH EXPERIMENT TRACKING.

1. Loads the prepared train/test data from the Hugging Face dataset space.
2. Defines an XGBoost classifier inside a preprocessing Pipeline.
3. Tunes hyperparameters with RandomizedSearchCV (5-fold stratified CV).
4. Logs every tuned parameter and evaluation metric to MLflow.
5. Evaluates the best model on the held-out test set.
6. Registers the best model on the Hugging Face model hub.

Run locally:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
    python tourism_project/model_building/train.py
"""

import json
import os
import sys
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "<your-hf-username>")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"

TARGET = "ProdTaken"
RANDOM_STATE = 42
N_ITER = 40          # number of hyperparameter combinations sampled
CV_FOLDS = 5
SCORING = "f1"       # F1 chosen because the target is imbalanced (~19% positive)

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_FILENAME = "tourism_model.joblib"


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_split(filename: str, token: str) -> pd.DataFrame:
    """Load a prepared split (train.csv / test.csv) from the HF dataset space."""
    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=filename,
        repo_type="dataset",
        token=token,
    )
    df = pd.read_csv(path)
    print(f"Loaded {filename} with shape {df.shape}")
    return df


# ----------------------------------------------------------------------
# Pipeline definition
# ----------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame, scale_pos_weight: float) -> Pipeline:
    """
    Build a preprocessing + XGBoost pipeline.

    Wrapping preprocessing inside the Pipeline (rather than transforming the
    data up front) guarantees that scaling/encoding are fitted only on the
    training fold during cross-validation, which prevents data leakage.
    """
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Passing the full category vocabulary explicitly means a rare category
    # (e.g. 'Free Lancer', which appears only twice) still produces a stable
    # column even when absent from a particular CV fold.
    categories = [sorted(X[c].dropna().unique().tolist()) for c in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    categories=categories,
                    handle_unknown="ignore",
                    drop="first",
                ),
                cat_cols,
            ),
        ]
    )

    classifier = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,  # counteracts class imbalance
        tree_method="hist",
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(model, X, y, split_name: str) -> dict:
    """Compute the standard classification metric suite for one split."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        f"{split_name}_accuracy": accuracy_score(y, preds),
        f"{split_name}_precision": precision_score(y, preds, zero_division=0),
        f"{split_name}_recall": recall_score(y, preds, zero_division=0),
        f"{split_name}_f1": f1_score(y, preds, zero_division=0),
        f"{split_name}_roc_auc": roc_auc_score(y, probs),
    }

    print(f"\n--- {split_name.upper()} METRICS ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


# ----------------------------------------------------------------------
# Model registration
# ----------------------------------------------------------------------
def register_model(token: str, model_path: str, metrics: dict, params: dict) -> None:
    """Push the trained model artifact to the Hugging Face model hub."""
    api = HfApi(token=token)

    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"\nModel repo already exists: {MODEL_REPO_ID}")
    except RepositoryNotFoundError:
        print(f"\nCreating model repo: {MODEL_REPO_ID}")
        create_repo(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            private=False,
            token=token,
            exist_ok=True,
        )

    # Upload the serialized pipeline
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=MODEL_FILENAME,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Register best tuned XGBoost pipeline",
    )

    # Upload a small metrics/params card alongside it for traceability
    meta_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    with open(meta_path, "w") as f:
        json.dump({"metrics": metrics, "best_params": params}, f, indent=2)

    api.upload_file(
        path_or_fileobj=meta_path,
        path_in_repo="metrics.json",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Add evaluation metrics and best hyperparameters",
    )

    print(f"Model registered at: https://huggingface.co/{MODEL_REPO_ID}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN environment variable is not set.")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ---------------- Load prepared data ----------------
    train_df = load_split("train.csv", token)
    test_df = load_split("test.csv", token)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # ---------------- Handle class imbalance ----------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nClass imbalance -> scale_pos_weight = {scale_pos_weight:.4f}")

    pipeline = build_pipeline(X_train, scale_pos_weight)

    # ---------------- Hyperparameter grid ----------------
    param_grid = {
        "classifier__n_estimators": [200, 300, 500, 700],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__gamma": [0, 0.1, 0.3],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    # ---------------- Train with MLflow tracking ----------------
    mlflow.set_experiment("tourism-wellness-package-prediction")

    with mlflow.start_run(run_name="xgboost_randomized_search"):
        print("\nStarting hyperparameter search...")
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_

        print(f"\nBest CV {SCORING}: {search.best_score_:.4f}")
        print("Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        # --- Log all tuned parameters ---
        mlflow.log_params(best_params)
        mlflow.log_param("n_iter", N_ITER)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("scoring", SCORING)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_metric(f"best_cv_{SCORING}", search.best_score_)

        # --- Log every individual search trial for full traceability ---
        cv_results = pd.DataFrame(search.cv_results_)
        cv_results_path = os.path.join(ARTIFACT_DIR, "cv_results.csv")
        cv_results.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path)

        # --- Evaluate ---
        train_metrics = evaluate(best_model, X_train, y_train, "train")
        test_metrics = evaluate(best_model, X_test, y_test, "test")
        all_metrics = {**train_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)

        print("\n--- TEST CONFUSION MATRIX ---")
        print(confusion_matrix(y_test, best_model.predict(X_test)))
        print("\n--- TEST CLASSIFICATION REPORT ---")
        print(classification_report(y_test, best_model.predict(X_test), digits=4))

        # --- Persist and register ---
        model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)
        print(f"\nModel saved to {model_path}")

        register_model(token, model_path, all_metrics, best_params)

    print("\nModel building complete.")


if __name__ == "__main__":
    main()
