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
5. **`results.tsv` on `main`:** Ensure branch `main` has a committed `results.tsv` whose header is `commit	branch	cv_mae_usd	cv_rmse_usd	status	description` (see **Logging results**). If it is missing, add it on `main` and push so every experiment branch can append into the same central file.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs **`train.py`** end-to-end on CPU (no fixed wall-clock budget; runtime depends on data size and model settings). From the repo root:

```bash
python3 train.py
```

(Use `uv run train.py` only if you manage the environment with `uv` and dependencies match [`README.md`](README.md).)

**What `train.py` does:** loads `train.csv` / `test.csv`, engineers features (counts, one-hot categoricals, multi-label language/framework tokens), fits a **`HistGradientBoostingRegressor`** on **`log1p(salary_usd)`**, runs **K-fold CV** (5 folds by default), reports **MAE and RMSE in USD** on out-of-fold predictions, retrains on all training rows, writes **`predictions.csv`** for the test set (predicted `salary_usd`).

**What `NOTES.md` does:** provides you notes from your supervisor regarding **what the next changes should look like** and how your progress is going so far. You should make sure to read this file each time before making a choice for what to edit in `train.py` next. 


**What you CAN do:**

- Edit **`train.py`** — feature caps (`MAX_LANGUAGE_FEATURES`, `MAX_FRAMEWORK_FEATURES`), encoders, fold count, model class or hyperparameters (`_make_model`), target transform, output path, etc.

**What you CANNOT do** (without changing the project scope):

- Change the required CSV column names expected by `train.py` without updating the loader and feature builder.
- Assume a separate harness like `prepare.py` or a `val_bpb` metric; this repo scores **tabular salary regression** only.

**The goal:** improve **out-of-fold CV MAE (USD)** — lower is better. **RMSE** is secondary but should not regress badly. Final predictions must still land in **`predictions.csv`** with one row per test row.

**Simplicity criterion:** Same as before — prefer a small, clear win over a large refactor for a tiny gain. If you delete code and match or beat CV MAE, keep it.

**First run:** Always establish a baseline by running the script as-is (or from a clean checkout) before iterating.

**Git / push policy (required):**

- After **every** experiment — whether CV **improved**, **worsened**, or the run **crashed** — **commit** the relevant state and **`git push`** your current experiment branch to `origin` **immediately**. Do not batch pushes until you “find a winner”; the remote should reflect every step.
- **`results.tsv` on `main` is the single cross-branch history.** After logging a run, **also** append that row to `results.tsv` on **`main`** and **push `main`** (see **The experiment loop**). That way work on `auto-ml/<tag-a>`, `auto-ml/<tag-b>`, etc. all accumulate in one file on the default branch without losing the narrative.

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

When an experiment finishes, record it in **`results.tsv`** (tab-separated, NOT comma-separated — commas break in descriptions).

**Canonical copy:** maintain one append-only **`results.tsv` on branch `main`** that aggregates every run from every experiment branch. Include a **`branch`** column so you can tell which line came from `auto-ml/mar5` vs `auto-ml/mar6`, etc.

The TSV has a header row and **6** columns:

```
commit	branch	cv_mae_usd	cv_rmse_usd	status	description
```

1. **commit** — git commit hash (short, 7 chars) for the **experiment branch** state you just pushed (use `0000000` if no commit applies, e.g. pre-commit crash-only note — still prefer committing something meaningful when possible)
2. **branch** — experiment branch name (e.g. `auto-ml/mar5`), not `main`
3. **cv_mae_usd** — out-of-fold CV MAE in USD (e.g. 13268.957) — use `0.000000` for crashes
4. **cv_rmse_usd** — out-of-fold CV RMSE in USD — use `0.000000` for crashes
5. **status** — `keep`, `discard`, or `crash` (your judgment on whether the change is worth keeping in the codebase **later**; independent of push policy — you still **push** either way)
6. **description** — short text: what this experiment tried

Example:

```
commit	branch	cv_mae_usd	cv_rmse_usd	status	description
a1b2c3d	auto-ml/mar5	13268.957	16166.266	keep	baseline train.py
b2c3d4e	auto-ml/mar5	12800.000	15800.000	keep	deeper trees
c3d4e5f	auto-ml/mar6	13500.000	16500.000	discard	alt encoder trial
d4e5f6g	auto-ml/mar6	0.000000	0.000000	crash	typo in column name
```

Before appending to `main`, `git pull` on `main` to reduce concurrent push conflicts. If the same `commit` already appears in `results.tsv`, do not duplicate the row.

## The experiment loop

Work on a dedicated experiment branch (e.g. `auto-ml/<tag>`). **`main` holds the shared `results.tsv` timeline** across all such branches.

LOOP FOREVER:

1. Note the current experiment **branch** name (e.g. `auto-ml/<tag>`).
2. Change `train.py` with one experimental idea.
3. Run: `python3 train.py > run.log 2>&1` (redirect everything — do NOT use `tee` or flood the context with full logs.)
4. Read metrics: `grep "^cv_mae_usd:\|^cv_rmse_usd:" run.log`
5. If grep is empty, the run crashed — inspect `tail -n 50 run.log`. Still **record** a `results.tsv` row with `crash` / zeros as in **Logging results**, **commit** the current tree (`train.py` and anything else relevant), and continue — **do not** skip the push steps below just because the score worsened or the run failed.
6. **`git add`** / **`git commit`** the experiment branch state (at minimum `train.py` when it changed; include `run.log` only if you explicitly want it in the repo).
7. **`git push origin HEAD`** immediately — **every** outcome (better MAE, worse MAE, or crash). Never defer pushes until the score improves.
8. **Update `main`’s `results.tsv`:** `git checkout main && git pull origin main`. Append **one** new line to `results.tsv` with `commit` (**short hash of the commit you just pushed on the experiment branch**), **`branch`**, `cv_mae_usd`, `cv_rmse_usd`, `status`, `description` (see **Logging results**). `git add results.tsv && git commit -m "results: <branch> <short-hash>" && git push origin main`.
9. Return to your experiment branch: `git checkout auto-ml/<tag>` (merge `main` into it if you want `results.tsv` locally — optional).

**Branch vs best code:** You may use `status: discard` for runs you do not want to build on, but **the commit stays pushed** on the experiment branch for history. To restore “best known `train.py`”, use `git revert`, a new branch from a good commit, or manual edits — **do not** use `git reset --hard` + force-push to erase history unless the human explicitly wants that.

**Timeout:** If a run stalls far beyond normal (e.g. hung process), treat as failure; log `crash`, push whatever commit records that state, still append to `main`’s `results.tsv`.

**Crashes:** Fix trivial bugs and re-run when useful; otherwise log `crash`, push, and append to `main`.

**Autonomy:** Once the loop starts, keep iterating without asking the human for permission to continue until stopped — same spirit as the original experiment, but tuned to **salary CV metrics**, **`train.py`**, **push-every-step**, and a **single `results.tsv` on `main`**.