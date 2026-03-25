# auto-ml — salary prediction

Small tabular ML project: predict **software engineering salary** (`salary_usd`) from role and stack features. Training and inference live in [`train.py`](train.py).

## Data

Place CSVs in the repo root (same directory as `train.py`):

| File | Purpose |
|------|---------|
| [`train.csv`](train.csv) | Labeled examples for training and cross-validation |
| [`test.csv`](test.csv) | Rows to score (must include the same feature columns as train) |

**Columns** (both files are expected to include):

- `experience` — numeric (years)
- `country` — categorical
- `education` — categorical
- `languages` — comma-separated list (e.g. `Python, Go`)
- `frameworks` — comma-separated list
- `company_size` — categorical (e.g. bin like `51-200`)
- `salary_usd` — target on train; present on test for schema compatibility (predictions are written separately)

## Requirements

Python 3.10+ recommended. Install:

```bash
pip install pandas numpy scipy scikit-learn
```

## Run

From the repo root:

```bash
python3 train.py
```

**Behavior:**

1. Loads `train.csv` and `test.csv`.
2. Runs **5-fold cross-validation** on the training set. The model is fit on `log1p(salary_usd)`; **MAE and RMSE** are reported on the original **USD** scale (out-of-fold predictions).
3. Retrains on **all** training data, predicts `test.csv`, writes **[`predictions.csv`](predictions.csv)** with a single column `salary_usd` (integer, non-negative).

**Console output:**

- **stderr:** progress (folds), CV MAE/RMSE, path to `predictions.csv`, min/median/max of predicted salaries.
- **stdout:** a short machine-friendly block, e.g. `cv_mae_usd`, `cv_rmse_usd`, `predictions_file`, `test_predictions`.

## What `train.py` does (high level)

- **Feature engineering:** splits `languages` / `frameworks` into tokens; adds `num_languages` / `num_frameworks`; one-hot encodes `country`, `education`, `company_size`; multi-label binary features for the top 60 languages and top 60 frameworks (by frequency on the fold or on full train).
- **Model:** scikit-learn `HistGradientBoostingRegressor` on a dense design matrix, with log target and `expm1` at inference.

## Experimenting

See [`program.md`](program.md) for an autonomous loop: tune **`train.py`**, measure **CV MAE / RMSE (USD)**, **`git push` the experiment branch after every run** (improved, worse, or crash), and keep a single append-only **`results.tsv` on `main`** that aggregates logs from all experiment branches (see that file for column schema, including **`branch`**).

## License

If you add a license, place it here. Until then, treat as private / local practice.
