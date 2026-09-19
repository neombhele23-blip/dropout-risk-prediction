from datetime import datetime, timezone
from app.extensions import db


class Cohort(db.Model):
    """One (code_module, code_presentation) row from the OULAD-derived
    cohort_dataset.csv — the historical record the model was trained on
    and the source of a module's up-to-date history features."""

    __tablename__ = "cohorts"

    id = db.Column(db.Integer, primary_key=True)
    code_module = db.Column(db.String(10), nullable=False, index=True)
    code_presentation = db.Column(db.String(10), nullable=False)

    enrollment_count = db.Column(db.Integer)
    avg_studied_credits = db.Column(db.Float)
    avg_prev_attempts = db.Column(db.Float)
    avg_imd = db.Column(db.Float)
    disability_rate = db.Column(db.Float)
    distinction_rate = db.Column(db.Float)
    difficulty_rate = db.Column(db.Float)
    module_presentation_length = db.Column(db.Integer)

    hist_avg_difficulty = db.Column(db.Float, nullable=True)
    hist_last_difficulty = db.Column(db.Float, nullable=True)
    hist_n_prior_cohorts = db.Column(db.Integer)
    hist_avg_enrollment = db.Column(db.Float, nullable=True)

    is_difficult = db.Column(db.Integer)  # ground-truth label, 0/1 (known cohorts only)

    __table_args__ = (
        db.UniqueConstraint("code_module", "code_presentation", name="uq_module_presentation"),
    )

    def to_dict(self):
        return {
            "code_module": self.code_module,
            "code_presentation": self.code_presentation,
            "enrollment_count": self.enrollment_count,
            "avg_studied_credits": self.avg_studied_credits,
            "avg_prev_attempts": self.avg_prev_attempts,
            "avg_imd": self.avg_imd,
            "disability_rate": self.disability_rate,
            "distinction_rate": self.distinction_rate,
            "difficulty_rate": self.difficulty_rate,
            "module_presentation_length": self.module_presentation_length,
            "hist_avg_difficulty": self.hist_avg_difficulty,
            "hist_last_difficulty": self.hist_last_difficulty,
            "hist_n_prior_cohorts": self.hist_n_prior_cohorts,
            "hist_avg_enrollment": self.hist_avg_enrollment,
            "is_difficult": self.is_difficult,
        }


class Prediction(db.Model):
    """A logged model prediction for a module — either re-scoring a
    known historical cohort or scoring a new/upcoming presentation
    whose own outcome isn't known yet. This is what an advisor-facing
    dashboard would read from to trigger interventions."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    code_module = db.Column(db.String(10), nullable=False, index=True)
    code_presentation = db.Column(db.String(10), nullable=True)  # null if hypothetical/upcoming

    predicted_label = db.Column(db.Integer, nullable=False)   # 0 / 1
    predicted_probability = db.Column(db.Float, nullable=False)  # P(persistently difficult)

    input_features = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "code_module": self.code_module,
            "code_presentation": self.code_presentation,
            "predicted_label": self.predicted_label,
            "predicted_probability": self.predicted_probability,
            "input_features": self.input_features,
            "created_at": self.created_at.isoformat(),
        }
