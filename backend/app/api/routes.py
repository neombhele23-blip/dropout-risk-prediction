from flask import Blueprint, jsonify, request
from app.extensions import db
from app.models import Cohort, Prediction
from app.ml import model_service

api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health():
    return {"status": "ok"}


@api_bp.get("/modules")
def list_modules():
    """One row per module, showing its most recent known cohort and
    that cohort's ground-truth difficulty label — the overview an
    advisor dashboard would list first."""
    cohorts = Cohort.query.order_by(Cohort.code_module, Cohort.code_presentation).all()
    by_module = {}
    for c in cohorts:
        by_module.setdefault(c.code_module, []).append(c)

    result = []
    for module, rows in sorted(by_module.items()):
        latest = rows[-1]
        result.append({
            "code_module": module,
            "n_cohorts": len(rows),
            "latest_presentation": latest.code_presentation,
            "latest_difficulty_rate": latest.difficulty_rate,
            "latest_is_difficult": latest.is_difficult,
            "historical_avg_difficulty": latest.hist_avg_difficulty,
        })
    return jsonify(result)


@api_bp.get("/modules/<code_module>/cohorts")
def module_cohorts(code_module):
    """Full cohort-by-cohort history for one module — powers a
    per-module trend view."""
    rows = (
        Cohort.query.filter_by(code_module=code_module)
        .order_by(Cohort.code_presentation)
        .all()
    )
    if not rows:
        return jsonify({"error": f"No cohorts found for module '{code_module}'"}), 404
    return jsonify([r.to_dict() for r in rows])


@api_bp.post("/predict")
def predict():
    """Score a module presentation for persistent-difficulty risk.

    Body can be EITHER:
      {"code_module": "BBB", "code_presentation": "2015J"}
        -> looks up the module's latest known cohort in the DB and uses
           its stats as the historical/known-in-advance features
           (i.e. "if the next presentation looks like the last one").
      OR a full feature dict, for a hypothetical/manual scenario:
      {"code_module": "BBB", "code_presentation": "2015J",
       "features": {"enrollment_count": 2000, "avg_prev_attempts": 0.3, ...}}
    """
    body = request.get_json(silent=True) or {}
    code_module = body.get("code_module")
    code_presentation = body.get("code_presentation")

    if not code_module:
        return jsonify({"error": "code_module is required"}), 400

    if "features" in body:
        features = body["features"]
    else:
        latest = (
            Cohort.query.filter_by(code_module=code_module)
            .order_by(Cohort.code_presentation.desc())
            .first()
        )
        if latest is None:
            return jsonify({"error": f"No historical cohorts found for module '{code_module}'. "
                                      f"Supply 'features' explicitly for a module with no history."}), 404

        # Use the latest cohort's own stats as this-cohort's "known in
        # advance" composition guess, and roll its outcome into history.
        features = {
            "enrollment_count": latest.enrollment_count,
            "avg_studied_credits": latest.avg_studied_credits,
            "avg_prev_attempts": latest.avg_prev_attempts,
            "avg_imd": latest.avg_imd,
            "disability_rate": latest.disability_rate,
            "distinction_rate": latest.distinction_rate,
            "module_presentation_length": latest.module_presentation_length,
            "hist_avg_difficulty": (
                ((latest.hist_avg_difficulty or 0) * (latest.hist_n_prior_cohorts or 0) + latest.difficulty_rate)
                / ((latest.hist_n_prior_cohorts or 0) + 1)
            ),
            "hist_last_difficulty": latest.difficulty_rate,
            "hist_n_prior_cohorts": (latest.hist_n_prior_cohorts or 0) + 1,
            "hist_avg_enrollment": (
                ((latest.hist_avg_enrollment or 0) * (latest.hist_n_prior_cohorts or 0) + latest.enrollment_count)
                / ((latest.hist_n_prior_cohorts or 0) + 1)
            ),
        }

    label, probability = model_service.predict(features)

    record = Prediction(
        code_module=code_module,
        code_presentation=code_presentation,
        predicted_label=label,
        predicted_probability=probability,
        input_features=features,
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "code_module": code_module,
        "code_presentation": code_presentation,
        "predicted_label": label,
        "predicted_difficult": bool(label),
        "predicted_probability": round(probability, 4),
        "features_used": features,
        "prediction_id": record.id,
    })


@api_bp.get("/flagged")
def flagged():
    """Most recent prediction per module, filtered to ones flagged as
    persistently difficult — the list an intervention workflow would
    actually act on. Optional ?min_probability=0.5 to threshold."""
    min_probability = float(request.args.get("min_probability", 0.5))

    all_preds = Prediction.query.order_by(Prediction.created_at.desc()).all()
    latest_per_module = {}
    for p in all_preds:
        if p.code_module not in latest_per_module:
            latest_per_module[p.code_module] = p

    result = [
        p.to_dict() for p in latest_per_module.values()
        if p.predicted_label == 1 and p.predicted_probability >= min_probability
    ]
    result.sort(key=lambda r: r["predicted_probability"], reverse=True)
    return jsonify(result)


@api_bp.get("/predictions")
def list_predictions():
    """Full prediction log, most recent first."""
    limit = min(int(request.args.get("limit", 50)), 200)
    preds = Prediction.query.order_by(Prediction.created_at.desc()).limit(limit).all()
    return jsonify([p.to_dict() for p in preds])
