# Dropout Risk Prediction — Module-Level (RESK401 Group 13)

## What this predicts

**Not** an individual student's dropout risk. Following the group's
corrected project framing: this predicts whether a **module** is likely
to be **persistently difficult** for its next cohort of students, based
on that module's own performance history in prior cohorts. The intended
use is to flag high-risk modules *before* a new cohort starts, so
first-years entering them can get targeted early support.

## Data

OULAD (Open University Learning Analytics Dataset): `studentInfo.csv`
and `courses.csv`. 32,594 students across 22 module-presentation
cohorts (7 modules: AAA–GGG, 2–4 presentations each, 2013–2014).

## Pipeline

1. **`prepare_data.py`** — aggregates student-level rows into one row
   per (module, presentation) cohort:
   - Target source: `difficulty_rate` = proportion of students in that
     cohort who Failed or Withdrew.
   - `is_difficult` label = 1 if `difficulty_rate` is above the
     **dataset's own median** (0.521). A fixed threshold (e.g. 40%)
     either flagged almost every cohort or almost none — the median
     gives a workable, balanced split on a sample this small.
   - Historical features (`hist_avg_difficulty`, `hist_last_difficulty`,
     `hist_avg_enrollment`, `hist_n_prior_cohorts`) are computed **only
     from that module's presentations before the current one** —
     mirrors real use (predicting before a new cohort's outcomes exist)
     and avoids leaking the current cohort's own outcome into its own
     features.
   - Other features (`avg_studied_credits`, `avg_prev_attempts`,
     `avg_imd`, `disability_rate`, `enrollment_count`,
     `module_presentation_length`) are cohort composition stats known
     at the start of a presentation, not outcome-derived.

2. **`train_model.py`** — trains a `RandomForestClassifier`
   (`max_depth=4`, `class_weight="balanced"`, 200 trees) with
   stratified k-fold cross-validation, reports accuracy / precision /
   recall / F1 per fold, a training-fit classification report and
   confusion matrix, and feature importances. Saves the fitted model
   to `dropout_risk_model.joblib`.

## Results (this run)

- 22 total cohorts → **15 usable** after dropping each module's first
  observed presentation (no prior history to build features from).
- Class balance: 8 difficult / 7 not-difficult.
- 5-fold stratified CV: **Accuracy 0.80, Precision 0.80, Recall 0.90,
  F1 0.83** (mean across folds).
- Top features by importance: `avg_prev_attempts`,
  `hist_avg_difficulty`, `hist_last_difficulty`, `hist_avg_enrollment`
  — i.e. a module's own difficulty history is doing most of the work,
  which is the expected and defensible result for this framing.

## Limitations to state in the report / oral

- **Sample size**: only 15 usable module-cohort rows. 5-fold CV means
  ~3 samples per test fold — individual fold scores swing widely (one
  fold scored 0.33 accuracy). Treat the CV mean as indicative, not a
  precise estimate. This is the single biggest weakness to name
  proactively rather than have a panel member raise it.
- **OULAD as a structural proxy**: this is UK Open University data,
  not DUT's own. Used because it has the multi-cohort, module-level
  structure DUT-specific data likely doesn't have available — already
  flagged as a limitation in the SLR.
- **Training-fit report is not held-out performance** — it's shown to
  extract feature importances and a confusion matrix on the data the
  final model was fit on; the CV numbers above are the honest
  out-of-sample estimate.
- **Small number of modules (7)**: the model is really learning
  patterns across only 7 distinct modules' histories, so it may not
  generalize to a genuinely new module with no track record at all
  (by construction, those rows were excluded).

## Next steps

- Try `studentAssessment.csv` / `studentVle.csv` (engagement data) for
  richer cohort-level features once the current model is written up —
  not needed for the current DSRM iteration but a natural extension.
- If DUT-specific historical module data becomes accessible later,
  re-run this exact pipeline on it directly.
- Wire `dropout_risk_model.joblib` into the Flask backend: on a new
  presentation's registration open, compute its known-in-advance
  features + the module's historical features, call
  `model.predict_proba()`, and surface flagged modules to advisors.

## Files

- `prepare_data.py` — run first, builds `cohort_dataset.csv`
- `train_model.py` — run second, trains model, prints metrics, saves
  `dropout_risk_model.joblib`
- `cohort_dataset.csv` — the 22-row module-cohort dataset
- `dropout_risk_model.joblib` — trained model + feature column list
