# Regenerating the trained model

This zip excludes the pre-trained `dropout_risk_model.joblib` files (a common
false-positive trigger for antivirus/browser download scanners, since pickle-based
files are flagged by heuristics regardless of actual content).

To regenerate them locally, from `model_training/`:

```bash
cd model_training
pip install -r ../backend/requirements.txt   # or your own venv setup
python3 train_model.py
```

This reruns training against the already-processed `cohort_dataset.csv`
(no need to re-fetch raw OULAD data or rerun `prepare_data.py`), and produces
a fresh `dropout_risk_model.joblib` in `model_training/`.

Copy that file into `backend/app/ml/dropout_risk_model.joblib` so the Flask
API (`/api/predict`, `/api/flagged`) has a model to load.
