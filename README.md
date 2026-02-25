# Hydration Level Predictor (Flask + ML)

This is a beginner-friendly, production-ready Flask web app that loads a saved machine-learning model and predicts **Hydration Level**: **Good** or **Poor**.

## Project structure

```
curser/
  app.py
  train_model.py
  requirements.txt
  model/
    hydration_model.joblib          # created after training
  templates/
    index.html
  static/
    style.css
```

## 1) Setup (local)

Create a virtual environment (recommended), then install dependencies:

### Windows PowerShell

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Download the Kaggle dataset

You said your Kaggle dataset is **daily_water_intake**. Download the CSV from Kaggle and keep it somewhere on your machine, for example:

```
data/daily_water_intake.csv
```

### Required columns

Your CSV must contain these feature columns:

- `Age`
- `Weight`
- `Gender`
- `Daily Water Intake`
- `Physical Activity Level`
- `Weather Condition`

And a target column for the label (hydration level). The training script will look for one of these names:

- `Hydration Level` (recommended)
- `Hydration_Level`
- `hydration_level`
- `HydrationStatus`
- `Hydration Status`
- `Target`
- `Label`

## 3) Train and save the model (joblib)

Run:

```bash
python train_model.py --data data/daily_water_intake.csv --out model/hydration_model.joblib
```

This creates:

```
model/hydration_model.joblib
```

### What gets saved?

The saved file is a **single scikit-learn Pipeline** containing:

- preprocessing
  - numeric: missing value imputation (median)
  - categorical: missing value imputation (most frequent) + OneHotEncoding
- model: Logistic Regression classifier

Because preprocessing is inside the pipeline, the Flask app can call `model.predict(...)` directly.

## 4) Run the Flask app (local)

```bash
python app.py
```

Open:

- `http://127.0.0.1:5000`

Health check:

- `http://127.0.0.1:5000/health`

## 5) Common errors

### “Model file not found…”

You haven’t trained the model yet, or the file path is wrong.

Fix:

- run the training step above
- ensure the file exists at `model/hydration_model.joblib`

### Categorical values don’t match your dataset

The UI provides common choices (Gender / Activity / Weather). If your dataset uses different words (example: `Sunny` instead of `Hot`), update the choices in `app.py` so they match what the model saw during training.

## 6) Push to GitHub (before Render deployment)

1) Create a GitHub repository (example name: `hydration-flask-ml`).

2) In this project folder, run:

```bash
git init
git add .
git commit -m "Initial Flask ML hydration app"
```

3) Add your GitHub remote and push:

```bash
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/hydration-flask-ml.git
git push -u origin main
```

### Should I push the trained model file?

For a beginner project: **yes, you can** commit `model/hydration_model.joblib` so Render can load it.

For real production: store it in an artifact store/object storage (S3, etc.) and download it during deploy.

## 7) Deploy on Render (step-by-step)

### A) Create a Web Service

1) Go to Render Dashboard → **New** → **Web Service**
2) Connect your GitHub account and select your repo

### B) Configure build + start commands

- **Build Command**:
  - `pip install -r requirements.txt`
- **Start Command**:
  - `gunicorn app:app`

### C) Environment variables (recommended)

In Render → Environment:

- `SECRET_KEY`: set any long random value
- `MODEL_PATH`: `model/hydration_model.joblib` (optional; this is the default)

### D) Deploy

Click **Create Web Service** → wait for build + deploy → open your service URL.

## Notes (production readiness)

- Uses `gunicorn` on Render (recommended for production).
- Includes server health endpoint: `/health`
- Uses a trained `Pipeline` to ensure preprocessing at inference time matches training time.
- Handles common user input errors using Flask flash messages.

