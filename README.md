# Dropout Risk Prediction — RESK401 Group 13

**Using Random Forest to Trigger Targeted Interventions for First-Year Undergraduates**

Identifies which university **modules** are persistently difficult across
cohorts — using senior cohorts' historical, multi-year performance data —
so targeted early interventions can be triggered for first-years entering
those flagged modules. This is a module-level early-warning system, not
an individual per-student dropout risk score.

## Structure

```
model_training/    — data prep + Random Forest training (OULAD-based)
                      see model_training/README.md for full methodology,
                      results, and stated limitations
backend/            — Flask API + dashboard frontend serving the trained
                      model (SQLite via Flask-SQLAlchemy, Blueprints)
                      see backend/README.md for setup and API reference
docs/               — Dropout_Model_Build_Walkthrough.docx: a step-by-step
                      explanation of the pipeline, mapped to the exact
                      file/function responsible for each step — useful
                      prep for defending the build in an oral
```

## Quick start

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python seed_db.py
python run.py
```

Then open `http://127.0.0.1:5000/` for the dashboard.

To rebuild the model from raw OULAD data instead of using the committed
`cohort_dataset.csv` / `dropout_risk_model.joblib`:

```bash
cd model_training
mkdir -p oulad_data   # place studentInfo.csv and courses.csv from OULAD here
python3 prepare_data.py
python3 train_model.py
```

## Results

5-fold stratified cross-validation on 15 usable module-cohorts:
**Accuracy 0.80, Precision 0.80, Recall 0.90, F1 0.83.**
See `model_training/README.md` for the full methodology, the
leakage-avoidance reasoning, and limitations worth stating in the oral
(small sample size, OULAD as a structural proxy for DUT data).

## Authors

RESK401 Group 13 — Neo Mbhele and team.
