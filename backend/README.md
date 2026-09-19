# Dropout Risk API (Flask backend — RESK401 Group 13)

Serves the trained module-level "persistently difficult" Random Forest
model over a small REST API, backed by SQLite, using Flask Blueprints.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python seed_db.py     # creates instance/dropout_risk.db and loads cohort_dataset.csv
python run.py         # starts the dev server on http://127.0.0.1:5000
```

Open `http://127.0.0.1:5000/` for the dashboard — Flask serves it directly
from `app/templates/` + `app/static/`, same origin as the API, so there's
no separate frontend deploy or CORS setup needed.

**Dashboard**: overview stats (modules tracked / flagged / total cohorts),
a modules table with inline difficulty-rate bars and status badges, a
"Predict next" button per module (calls `/api/predict`, shows a toast,
refreshes flagged list), a click-through cohort-history panel with a
small trend sparkline, and a live flagged-modules list. All wired to the
API below — verified end-to-end (predict → flag → detail view) with
screenshots during this build.

## Structure

```
app/
  __init__.py       # app factory: config, db.init_app, blueprint registration, "/" route
  extensions.py     # the shared SQLAlchemy() instance
  models.py         # Cohort (historical data) and Prediction (logged scores)
  api/
    routes.py       # all /api/* endpoints
  ml/
    model_service.py       # loads dropout_risk_model.joblib, wraps predict()
    dropout_risk_model.joblib
  templates/
    index.html      # dashboard page
  static/
    css/dashboard.css
    js/dashboard.js # fetches /api/*, renders table/detail/flagged, handles predict clicks
run.py              # entrypoint
seed_db.py          # one-time load of cohort_dataset.csv into the DB
requirements.txt
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness check |
| GET | `/api/modules` | one row per module: latest known cohort + its difficulty label |
| GET | `/api/modules/<code_module>/cohorts` | full cohort-by-cohort history for one module |
| POST | `/api/predict` | score a module presentation; logs the prediction |
| GET | `/api/flagged?min_probability=0.5` | most recent prediction per module, filtered to flagged ones, sorted by risk |
| GET | `/api/predictions?limit=50` | raw prediction log, most recent first |

### `POST /api/predict`

Two ways to call it:

**A — use a module's latest known cohort as the basis** (simplest; "if
the next presentation looks like the last one"):
```json
{"code_module": "BBB", "code_presentation": "2015J"}
```

**B — supply your own feature values** (for a hypothetical scenario, or
a module with no cohort history yet):
```json
{
  "code_module": "BBB",
  "code_presentation": "2015J",
  "features": {
    "enrollment_count": 2000,
    "avg_studied_credits": 85,
    "avg_prev_attempts": 0.2,
    "avg_imd": 45,
    "disability_rate": 0.1,
    "distinction_rate": 0.08,
    "module_presentation_length": 260,
    "hist_avg_difficulty": 0.53,
    "hist_last_difficulty": 0.50,
    "hist_n_prior_cohorts": 4,
    "hist_avg_enrollment": 1950
  }
}
```

Response:
```json
{
  "code_module": "DDD",
  "code_presentation": "2015J",
  "predicted_label": 1,
  "predicted_difficult": true,
  "predicted_probability": 0.9314,
  "features_used": { ... },
  "prediction_id": 4
}
```

## Tested endpoints (this session)

All six endpoints were run against the live model + seeded DB:
`/api/health`, `/api/modules`, `/api/modules/BBB/cohorts`,
`/api/predict` (both the lookup-based and manual-feature paths),
`/api/flagged`, and 404 error handling for an unknown module —
all returned correct results.

## Next steps

- Point your existing Flask-Login/Blueprints frontend (from Alumni
  Nexus) at `/api/flagged` for an advisor-facing "modules to watch"
  view.
- Deploy: this app is stateless enough for Railway/Render as planned —
  just make sure `seed_db.py` runs once on first deploy (or bake
  `instance/dropout_risk.db` into the image).
- Add a `PUT /api/cohorts/<module>/<presentation>` endpoint once a real
  presentation's outcome is known, to append it as new ground truth and
  eventually retrain.
