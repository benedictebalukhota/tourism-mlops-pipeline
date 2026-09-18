# Visit with Us — Wellness Tourism Package Prediction (MLOps Pipeline)

An end-to-end MLOps pipeline that predicts whether a customer will purchase the
newly introduced **Wellness Tourism Package**, so that the sales team can target
customers *before* contacting them.

The pipeline is fully automated with **GitHub Actions** and uses the
**Hugging Face Hub** for data registration, model registry, and app hosting.

---

## Architecture

```
  GitHub push to main
          │
          ▼
  ┌───────────────────┐
  │ 1. Data           │  data_register.py
  │    Registration   │  ──► HF Dataset Space (raw tourism.csv)
  └─────────┬─────────┘
            ▼
  ┌───────────────────┐
  │ 2. Data           │  prep.py
  │    Preparation    │  ──► clean, split, ──► HF Dataset Space (train/test.csv)
  └─────────┬─────────┘
            ▼
  ┌───────────────────┐
  │ 3. Model Building │  train.py
  │    + Tracking     │  ──► MLflow logs, ──► HF Model Hub (tourism_model.joblib)
  └─────────┬─────────┘
            ▼
  ┌───────────────────┐
  │ 4. Deployment     │  hosting.py + Dockerfile + app.py
  │                   │  ──► HF Space (Streamlit, Docker SDK)
  └───────────────────┘
```

---

## Repository structure

```
.
├── .github/
│   └── workflows/
│       └── pipeline.yml              # GitHub Actions CI/CD workflow
├── tourism_project/
│   ├── data/
│   │   ├── tourism.csv               # raw dataset
│   │   └── data_register.py          # STEP 1: register data on HF
│   ├── model_building/
│   │   ├── prep.py                   # STEP 2: clean + split + upload
│   │   └── train.py                  # STEP 3: tune, track, register model
│   └── deployment/
│       ├── app.py                    # Streamlit inference app
│       ├── Dockerfile                # container definition (port 7860)
│       ├── requirements.txt          # deployment-only dependencies
│       └── hosting.py                # STEP 4: push app to HF Space
├── notebooks/
│   └── tourism_mlops_notebook.ipynb  # EDA + full pipeline walkthrough
├── requirements.txt                  # pipeline dependencies
└── README.md
```

---

## Setup

### 1. Prerequisites

- A [Hugging Face](https://huggingface.co) account
- A Hugging Face **access token** with `write` permission
  (Settings → Access Tokens → New token → role: `write`)
- A GitHub repository

### 2. Configure GitHub secrets

In your GitHub repo, go to
**Settings → Secrets and variables → Actions → New repository secret**
and add:

| Secret name   | Value                                   |
|---------------|-----------------------------------------|
| `HF_TOKEN`    | your Hugging Face write token           |
| `HF_USERNAME` | your Hugging Face username              |

### 3. Run locally (optional)

```bash
pip install -r requirements.txt

export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
export HF_USERNAME=your-hf-username

python tourism_project/data/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
python tourism_project/deployment/hosting.py
```

### 4. Trigger the automated pipeline

```bash
git add .
git commit -m "Initial MLOps pipeline"
git push origin main
```

The workflow runs automatically on every push to `main`. It can also be
triggered manually from the **Actions** tab (`workflow_dispatch`).

---

## Model

| Item | Value |
|---|---|
| Algorithm | XGBoost classifier (`XGBClassifier`) |
| Preprocessing | `StandardScaler` (numeric) + `OneHotEncoder` (categorical), inside a `Pipeline` |
| Tuning | `RandomizedSearchCV`, 40 candidates, 5-fold stratified CV |
| Optimised metric | F1 (target is imbalanced at ~19% positive) |
| Imbalance handling | `scale_pos_weight` |
| Tracking | MLflow (parameters, metrics, artifacts) |

### Test-set performance

| Metric | Score |
|---|---|
| Accuracy | 0.9455 |
| Precision | 0.9014 |
| Recall | 0.8050 |
| F1 | 0.8505 |
| ROC-AUC | 0.9651 |

---

## Links

- **Dataset:** `https://huggingface.co/datasets/<your-hf-username>/tourism-wellness-dataset`
- **Model:** `https://huggingface.co/<your-hf-username>/tourism-wellness-model`
- **App (Space):** `https://huggingface.co/spaces/<your-hf-username>/tourism-wellness-app`
