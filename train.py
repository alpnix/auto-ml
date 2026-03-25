"""
Train a salary regressor on train.csv, evaluate with K-fold CV, write predictions for test.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder


ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
PREDICTIONS_CSV = ROOT / "predictions.csv"

# Cap multi-label vocabulary to keep memory reasonable for dense HGBR input
MAX_LANGUAGE_FEATURES = 100
MAX_FRAMEWORK_FEATURES = 100
N_SPLITS = 5
RANDOM_STATE = 42


def _parse_comma_list(value) -> list[str]:
    if pd.isna(value):
        return []
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _add_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_langs"] = out["languages"].map(_parse_comma_list)
    out["_fws"] = out["frameworks"].map(_parse_comma_list)
    out["num_languages"] = out["_langs"].map(len).astype(np.int32)
    out["num_frameworks"] = out["_fws"].map(len).astype(np.int32)
    return out


def _top_k_labels(series_of_lists: pd.Series, k: int) -> list[str]:
    from collections import Counter

    c: Counter[str] = Counter()
    for lst in series_of_lists:
        c.update(lst)
    return [lab for lab, _ in c.most_common(k)]


class MultiLabelEncoder:
    """Fit top-k vocabulary on train rows; transform to binary columns (unknown labels ignored)."""

    def __init__(self, max_features: int):
        self.max_features = max_features
        self._mlb: MultiLabelBinarizer | None = None
        self._classes: list[str] | None = None

    def fit(self, series_of_lists: pd.Series) -> MultiLabelEncoder:
        self._classes = sorted(_top_k_labels(series_of_lists, self.max_features))
        self._mlb = MultiLabelBinarizer(classes=self._classes, sparse_output=True)
        self._mlb.fit(series_of_lists.tolist())
        return self

    def transform(self, series_of_lists: pd.Series) -> sparse.csr_matrix:
        assert self._mlb is not None and self._classes is not None
        return self._mlb.transform(series_of_lists.tolist())


class SalaryFeatureBuilder:
    """Dense design matrix: numeric + ordinal (native cat) + multi-label skills."""

    # column layout: [exp, exp^2, n_lang, n_fw] + [country_ord, edu_ord, size_ord] + [lang...] + [fw...]
    N_NUM = 4
    CAT_COLS = ["country", "education", "company_size"]
    N_CAT = 3  # indices N_NUM .. N_NUM+N_CAT-1 are native categoricals

    def __init__(self):
        self._ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._enc_lang = MultiLabelEncoder(MAX_LANGUAGE_FEATURES)
        self._enc_fw = MultiLabelEncoder(MAX_FRAMEWORK_FEATURES)

    def cat_feature_indices(self) -> list[int]:
        return list(range(self.N_NUM, self.N_NUM + self.N_CAT))

    def fit(self, df: pd.DataFrame) -> SalaryFeatureBuilder:
        d = _add_list_columns(df)
        self._ord.fit(d[self.CAT_COLS])
        self._enc_lang.fit(d["_langs"])
        self._enc_fw.fit(d["_fws"])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        d = _add_list_columns(df)
        exp = d["experience"].to_numpy(dtype=np.float64)
        n_lang = d["num_languages"].to_numpy(dtype=np.float64)
        n_fw = d["num_frameworks"].to_numpy(dtype=np.float64)
        X_num = np.column_stack([exp, exp ** 2, n_lang, n_fw])
        X_cat = self._ord.transform(d[self.CAT_COLS])
        X_lang = self._enc_lang.transform(d["_langs"]).toarray()
        X_fw = self._enc_fw.transform(d["_fws"]).toarray()
        return np.hstack([X_num, X_cat, X_lang, X_fw])


def _load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    required = {
        "experience",
        "country",
        "education",
        "languages",
        "frameworks",
        "company_size",
        "salary_usd",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {sorted(missing)}")
    return df


def _load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    required = {
        "experience",
        "country",
        "education",
        "languages",
        "frameworks",
        "company_size",
        "salary_usd",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test.csv missing columns: {sorted(missing)}")
    return df


def _make_model(cat_features: list[int] | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=7,
        min_samples_leaf=20,
        l2_regularization=0.05,
        categorical_features=cat_features,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=40,
        random_state=RANDOM_STATE,
    )


def run_cv(train_df: pd.DataFrame) -> tuple[float, float]:
    y = train_df["salary_usd"].to_numpy(dtype=np.float64)
    y_log = np.log1p(y)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(train_df), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df), start=1):
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]
        fb = SalaryFeatureBuilder()
        fb.fit(tr)
        X_tr = fb.transform(tr)
        X_va = fb.transform(va)
        model = _make_model(fb.cat_feature_indices())
        model.fit(X_tr, y_log[tr_idx])
        oof[va_idx] = model.predict(X_va)
        print(f"  fold {fold}/{N_SPLITS} done", file=sys.stderr)

    pred_salary = np.clip(np.expm1(oof), 0.0, None)
    mae = mean_absolute_error(y, pred_salary)
    rmse = float(np.sqrt(mean_squared_error(y, pred_salary)))
    return mae, rmse


def main() -> None:
    if not TRAIN_CSV.is_file():
        raise SystemExit(f"Missing {TRAIN_CSV}")
    if not TEST_CSV.is_file():
        raise SystemExit(f"Missing {TEST_CSV}")

    train_df = _load_train()
    test_df = _load_test()

    print("Salary regression — train.py", file=sys.stderr)
    print(f"  train rows: {len(train_df)}", file=sys.stderr)
    print(f"  test rows:  {len(test_df)}", file=sys.stderr)

    print("K-fold CV (log1p target, metrics on USD scale)...", file=sys.stderr)
    mae, rmse = run_cv(train_df)
    print(f"  CV MAE (USD):  {mae:,.2f}", file=sys.stderr)
    print(f"  CV RMSE (USD): {rmse:,.2f}", file=sys.stderr)

    print("Fitting on full train, predicting test...", file=sys.stderr)
    fb = SalaryFeatureBuilder()
    fb.fit(train_df)
    X_train = fb.transform(train_df)
    X_test = fb.transform(test_df)
    y_log_full = np.log1p(train_df["salary_usd"].to_numpy(dtype=np.float64))

    final = _make_model(fb.cat_feature_indices())
    final.fit(X_train, y_log_full)
    test_pred_log = final.predict(X_test)
    test_pred = np.clip(np.expm1(test_pred_log), 0.0, None).astype(np.int64)

    out = pd.DataFrame({"salary_usd": test_pred})
    out.to_csv(PREDICTIONS_CSV, index=False)

    print(f"  wrote {PREDICTIONS_CSV} ({len(out)} rows)", file=sys.stderr)
    print(
        f"  pred salary_usd min/median/max: "
        f"{test_pred.min():,} / {np.median(test_pred):,.0f} / {test_pred.max():,}",
        file=sys.stderr,
    )

    print("---")
    print(f"cv_mae_usd:       {mae:.6f}")
    print(f"cv_rmse_usd:      {rmse:.6f}")
    print(f"predictions_file: {PREDICTIONS_CSV.name}")
    print(f"test_predictions: {len(out)}")


if __name__ == "__main__":
    main()
