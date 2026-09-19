"""Load cohort_dataset.csv (produced by prepare_data.py) into the DB.

Run once after a fresh `flask db` / app.create_app() creates the tables:
    python seed_db.py
"""

import pandas as pd
from app import create_app
from app.extensions import db
from app.models import Cohort


def seed():
    df = pd.read_csv("cohort_dataset.csv")
    df = df.where(pd.notnull(df), None)

    app = create_app()
    with app.app_context():
        db.create_all()

        added, skipped = 0, 0
        for _, row in df.iterrows():
            exists = Cohort.query.filter_by(
                code_module=row["code_module"],
                code_presentation=row["code_presentation"],
            ).first()
            if exists:
                skipped += 1
                continue

            cohort = Cohort(
                code_module=row["code_module"],
                code_presentation=row["code_presentation"],
                enrollment_count=row["enrollment_count"],
                avg_studied_credits=row["avg_studied_credits"],
                avg_prev_attempts=row["avg_prev_attempts"],
                avg_imd=row["avg_imd"],
                disability_rate=row["disability_rate"],
                distinction_rate=row["distinction_rate"],
                difficulty_rate=row["difficulty_rate"],
                module_presentation_length=row["module_presentation_length"],
                hist_avg_difficulty=row["hist_avg_difficulty"],
                hist_last_difficulty=row["hist_last_difficulty"],
                hist_n_prior_cohorts=row["hist_n_prior_cohorts"],
                hist_avg_enrollment=row["hist_avg_enrollment"],
                is_difficult=row["is_difficult"],
            )
            db.session.add(cohort)
            added += 1

        db.session.commit()
        print(f"Seeded {added} cohorts ({skipped} already present, skipped).")


if __name__ == "__main__":
    seed()
