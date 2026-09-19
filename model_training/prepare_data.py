"""
RESK401 Group 13 — Dropout Risk Prediction (module-level)
Step 1: Build a module-cohort level dataset from OULAD.

Approach (per project scope):
  We are NOT predicting an individual student's dropout risk.
  We are identifying which MODULES are persistently difficult across
  cohorts, using senior/prior cohorts' historical performance to flag
  risk for the NEXT cohort of first-years entering that module.

Each row in the output = one (code_module, code_presentation) cohort.
Target = whether THIS cohort's fail+withdrawal rate is "difficult"
         (above DIFFICULTY_THRESHOLD).
Features = (a) known-in-advance cohort composition stats (demographics,
              enrollment size, module length) — available before the
              presentation's outcomes are known, so not leakage.
           (b) historical difficulty features computed ONLY from that
              module's PRIOR presentations (true historical signal,
              mirrors how this would be used in production: flag a
              module before a new cohort's results come in).
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "oulad_data")
# Threshold is set dynamically to the dataset's own median difficulty_rate
# (see main()) rather than a fixed guess like 0.40 — with only 22 cohorts,
# a fixed threshold either flagged almost everything or almost nothing as
# "difficult". The median gives a balanced difficult / not-difficult split,
# which stratified k-fold needs to work at all on a sample this small.

# Presentation codes sort chronologically as strings EXCEPT B before J
# within the same year is already correct alphabetically (B < J), so
# a simple string sort of "YYYYP" works here.
def presentation_sort_key(pres):
    return pres  # e.g. "2013B" < "2013J" < "2014B" < "2014J" sorts correctly as strings


def load_raw():
    student_info = pd.read_csv(f"{DATA_DIR}/studentInfo.csv")
    courses = pd.read_csv(f"{DATA_DIR}/courses.csv")
    return student_info, courses


def imd_to_numeric(imd_band):
    """Convert IMD deprivation band string (e.g. '20-30%') to a rough
    numeric midpoint for averaging. Higher = less deprived. '?' -> NaN."""
    if pd.isna(imd_band) or imd_band == "?":
        return np.nan
    band = str(imd_band).replace("%", "")
    try:
        lo, hi = band.split("-")
        return (float(lo) + float(hi)) / 2
    except ValueError:
        return np.nan


def build_cohort_table(student_info: pd.DataFrame) -> pd.DataFrame:
    df = student_info.copy()
    df["imd_numeric"] = df["imd_band"].apply(imd_to_numeric)
    df["is_disabled"] = (df["disability"] == "Y").astype(int)
    df["is_fail_or_withdrawn"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    df["is_distinction"] = (df["final_result"] == "Distinction").astype(int)

    grouped = df.groupby(["code_module", "code_presentation"])

    cohort = grouped.agg(
        enrollment_count=("id_student", "count"),
        avg_studied_credits=("studied_credits", "mean"),
        avg_prev_attempts=("num_of_prev_attempts", "mean"),
        avg_imd=("imd_numeric", "mean"),
        disability_rate=("is_disabled", "mean"),
        distinction_rate=("is_distinction", "mean"),
        difficulty_rate=("is_fail_or_withdrawn", "mean"),  # this cohort's OWN outcome (the target source)
    ).reset_index()

    return cohort


def add_historical_features(cohort: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    cohort = cohort.merge(courses, on=["code_module", "code_presentation"], how="left")
    cohort = cohort.sort_values(["code_module", "code_presentation"], key=lambda s: s if s.name != "code_presentation" else s.map(presentation_sort_key))

    rows = []
    for module, group in cohort.groupby("code_module"):
        group = group.sort_values("code_presentation")
        prior_difficulty_rates = []
        prior_enrollments = []
        for _, row in group.iterrows():
            row = row.copy()
            if prior_difficulty_rates:
                row["hist_avg_difficulty"] = np.mean(prior_difficulty_rates)
                row["hist_n_prior_cohorts"] = len(prior_difficulty_rates)
                row["hist_last_difficulty"] = prior_difficulty_rates[-1]
                row["hist_avg_enrollment"] = np.mean(prior_enrollments)
            else:
                # First observed presentation of this module: no history yet.
                row["hist_avg_difficulty"] = np.nan
                row["hist_n_prior_cohorts"] = 0
                row["hist_last_difficulty"] = np.nan
                row["hist_avg_enrollment"] = np.nan
            rows.append(row)
            prior_difficulty_rates.append(row["difficulty_rate"])
            prior_enrollments.append(row["enrollment_count"])

    result = pd.DataFrame(rows)
    threshold = result["difficulty_rate"].median()
    result["is_difficult"] = (result["difficulty_rate"] > threshold).astype(int)
    result.attrs["difficulty_threshold"] = threshold
    return result


def main():
    student_info, courses = load_raw()
    cohort = build_cohort_table(student_info)
    full = add_historical_features(cohort, courses)

    out_path = os.path.join(os.path.dirname(__file__), "cohort_dataset.csv")
    full.to_csv(out_path, index=False)

    threshold = full.attrs.get("difficulty_threshold")
    print(f"Built {len(full)} module-cohort rows.")
    print(f"Difficulty threshold (dataset median): {threshold:.3f}")
    print(f"Cohorts with no prior history (first presentation of a module): {full['hist_n_prior_cohorts'].eq(0).sum()}")
    print(f"Difficult cohorts (difficulty_rate > {threshold:.3f}): {full['is_difficult'].sum()} / {len(full)}")
    print(f"\nSaved to {out_path}")
    print("\nPreview:")
    print(full[["code_module", "code_presentation", "enrollment_count", "difficulty_rate",
                "hist_avg_difficulty", "hist_n_prior_cohorts", "is_difficult"]].to_string(index=False))


if __name__ == "__main__":
    main()
