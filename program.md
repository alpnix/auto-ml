# auto-ml

This is my copy of Karpathy's experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `auto-ml/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b auto-ml/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — data layout, dependencies, how to run `train.py`, what it outputs.
   - `train.py` — the file you modify: features, sklearn model, CV, and `predictions.csv` export.
4. **Verify data exists**: Check that `test.csv` and `train.csv` exist in the repository. If not, tell the human to upload them (see `README.md` for expected columns).
5. **Initialize results.tsv** (optional): Create `results.tsv` with header `commit	cv_mae_usd	cv_rmse_usd	status	description`. The baseline row is recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs **`train.py`** end-to-end on CPU (no fixed wall-clock budget; runtime depends on data size and model settings). From the repo root:

```bash
python3 train.py
```

(Use `uv run train.py` only if you manage the environment with `uv` and dependencies match [`README.md`](README.md).)

**What `train.py` does:** loads `train.csv` / `test.csv`, engineers features (counts, one-hot categoricals, multi-label language/framework tokens), fits a **`HistGradientBoostingRegressor`** on **`log1p(salary_usd)`**, runs **K-fold CV** (5 folds by default), reports **MAE and RMSE in USD** on out-of-fold predictions, retrains on all training rows, writes **`predictions.csv`** for the test set (predicted `salary_usd`).

**What you CAN do:**

- Edit **`train.py`** — feature caps (`MAX_LANGUAGE_FEATURES`, `MAX_FRAMEWORK_FEATURES`), encoders, fold count, model class or hyperparameters (`_make_model`), target transform, output path, etc.

**What you CANNOT do** (without changing the project scope):

- Change the required CSV column names expected by `train.py` without updating the loader and feature builder.
- Assume a separate harness like `prepare.py` or a `val_bpb` metric; this repo scores **tabular salary regression** only.

**The goal:** improve **out-of-fold CV MAE (USD)** — lower is better. **RMSE** is secondary but should not regress badly. Final predictions must still land in **`predictions.csv`** with one row per test row.

**Simplicity criterion:** Same as before — prefer a small, clear win over a large refactor for a tiny gain. If you delete code and match or beat CV MAE, keep it.

**First run:** Always establish a baseline by running the script as-is (or from a clean checkout) before iterating.

## Output format

After a successful run, **stdout** ends with:

```
---
cv_mae_usd:       <float>
cv_rmse_usd:      <float>
predictions_file: predictions.csv
test_predictions: <int>
```

Progress and human-readable metrics (fold completion, CV MAE/RMSE, prediction min/median/max) go to **stderr**.

To scrape metrics from a log:

```bash
grep "^cv_mae_usd:\|^cv_rmse_usd:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	cv_mae_usd	cv_rmse_usd	status	description
```

1. git commit hash (short, 7 chars)
2. out-of-fold CV MAE in USD (e.g. 13268.957) — use `0.000000` for crashes
3. out-of-fold CV RMSE in USD — use `0.000000` for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	cv_mae_usd	cv_rmse_usd	status	description
a1b2c3d	13268.957	16166.266	keep	baseline train.py
b2c3d4e	12800.000	15800.000	keep	CatBoost trial (if swapped in)
c3d4e5f	13500.000	16500.000	discard	too deep trees, overfit
d4e5f6g	0.000000	0.000000	crash	typo in column name
```

## The experiment loop

The experiment can run on a dedicated branch (e.g. `auto-ml/<tag>`).

LOOP FOREVER:

1. Note the current branch/commit.
2. Change `train.py` with one experimental idea.
3. `git commit`
4. Run: `python3 train.py > run.log 2>&1` (redirect everything — do NOT use `tee` or flood the context with full logs.)
5. Read metrics: `grep "^cv_mae_usd:\|^cv_rmse_usd:" run.log`
6. If grep is empty, the run crashed — inspect `tail -n 50 run.log`, fix or discard.
7. Append a row to `results.tsv` (leave `results.tsv` untracked if you prefer not to commit it).
8. If **CV MAE improved** (lower USD error), keep the commit as the new best.
9. If MAE is equal or worse (or RMSE blew up), `git reset` to the previous best.

**Timeout:** If a run stalls far beyond normal (e.g. hung process), treat as failure; revert or skip.

**Crashes:** Fix trivial bugs and re-run; abandon ideas that are structurally broken and log `crash`.

**Autonomy:** Once the loop starts, keep iterating without asking the human for permission to continue until stopped — same spirit as the original experiment, but tuned to **salary CV metrics** and **`train.py`**.