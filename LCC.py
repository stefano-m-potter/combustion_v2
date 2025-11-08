#!/usr/bin/env python
# coding: utf-8

import os
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.utils.validation import check_is_fitted
from scipy.stats import randint, uniform
import joblib

# ----------------- PATHS -----------------
INPUT_CSV = "/explore/nobackup/people/spotter5/new_combustion/2025-10-03_CombustionModelPredictors.csv"
OUT_DIR   = "/explore/nobackup/people/spotter5/new_combustion/LCC"
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------- SEARCH CONFIG -----------------
RANDOM_STATE   = 42
N_JOBS         = -1
INNER_FOLDS    = 5        # inner CV folds for hyperparameter tuning
N_ITER_SEARCH  = 40       # RandomizedSearch iterations per LOOCV split
SCORER         = make_scorer(mean_squared_error, greater_is_better=False)  # neg MSE

# Reasonable, broad-ish RF search spaces
RF_PARAM_DIST = {
    "n_estimators": randint(200, 1000),
    "max_depth":    randint(3, 40),
    "max_features": uniform(0.2, 0.8),  # fraction of features (0.2..1.0)
    "min_samples_split": randint(2, 20),
    "min_samples_leaf":  randint(1, 10),
    "bootstrap":   [True, False]
}

print(f"Reading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# ----------------- BASIC CLEANUP -----------------
df.columns = [c.strip() for c in df.columns]

rename_map = {}
if 'ID' in df.columns: rename_map['ID'] = 'id'
if 'Id' in df.columns: rename_map['Id'] = 'id'
if 'project_name' in df.columns and 'project.name' not in df.columns:
    rename_map['project_name'] = 'project.name'
if 'Date' in df.columns and 'date' not in df.columns:
    rename_map['Date'] = 'date'
if 'latitude' in df.columns and 'lat' not in df.columns:
    rename_map['latitude'] = 'lat'
if 'longitude' in df.columns and 'lon' not in df.columns:
    rename_map['longitude'] = 'lon'
if 'fireYr' in df.columns and 'burn_year' not in df.columns:
    rename_map['fireYr'] = 'burn_year'
df = df.rename(columns=rename_map)

# Schema snapshot
schema = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str),
    "n_null": df.isna().sum(),
    "n_unique": [df[c].nunique(dropna=True) for c in df.columns]
})
schema.to_csv(os.path.join(OUT_DIR, "schema_summary.csv"), index=False)

# ----------------- CATEGORICAL: LandCover -> one-hot -----------------
if 'LandCover' in df.columns:
    df = pd.get_dummies(df, columns=['LandCover'], prefix='LC', drop_first=True, dummy_na=False)

# ----------------- EXCLUDED PREDICTOR COLUMNS -----------------
EXCLUDE_PRED_COLS = {
    'id', 'project.name', 'lat', 'lon', 'burn_year', 'date', 'project',
    # allow for variants if they slipped through
    'ID', 'Id', 'project_name', 'latitude', 'longitude', 'fireYr', 'Date', 'landcover_name'
}

# ----------------- TARGET PICKER -----------------
def pick_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

COL_ABOVE = pick_col(['combusted_above', 'above.carbon.combusted'])
COL_BELOW = pick_col(['combusted_below'])
COL_DEPTH = pick_col(['burn_depth'])

# If you later want all 3, switch back to:
# targets = [(c, "units") for c in [COL_ABOVE, COL_BELOW, COL_DEPTH] if c]
# ALL_TARGET_COLS = [c for c in [COL_ABOVE, COL_BELOW, COL_DEPTH] if c]

targets = [(c, "units") for c in [COL_DEPTH] if c]
if not targets:
    raise ValueError("None of the expected target columns were found in the dataset.")

ALL_TARGET_COLS = [c for c in [COL_DEPTH] if c]

# ------------- GLOBAL METRICS STORAGE FOR VIOLIN PLOT -------------
GLOBAL_METRICS = []   # will collect per-target LOOCV metrics (including R²)

# ----------------- Helper: build X, y -----------------
def build_xy(df_in: pd.DataFrame, target_col: str):
    drop_cols = [c for c in EXCLUDE_PRED_COLS if c in df_in.columns]
    work = df_in.drop(columns=drop_cols, errors='ignore').copy()
    work = work.dropna(subset=[target_col])
    y = work[target_col].astype(float).copy()
    X = work.drop(columns=ALL_TARGET_COLS, errors='ignore')
    # keep only numeric predictors
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
    return X, y

# ----------------- Nested LOOCV with inner tuning -----------------
def run_target_nested_loocv(target_col: str, units_label: str = "units"):
    X, y = build_xy(df, target_col)
    if X.shape[1] == 0 or len(y) < 3:
        print(f"[ERROR] Not enough predictors or samples for '{target_col}'.")
        return

    print(f"\nTarget: {target_col} | X: {X.shape} | y: {y.shape}")
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    loo = LeaveOneOut()
    y_pred = np.zeros_like(y, dtype=float)

    split_records = []  # per-split metadata: best params, inner best score

    # Iterate LOOCV splits
    for i, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        # inner CV tuner
        inner = KFold(n_splits=min(INNER_FOLDS, len(ytr)), shuffle=True, random_state=RANDOM_STATE)
        base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS)

        tuner = RandomizedSearchCV(
            estimator=base,
            param_distributions=RF_PARAM_DIST,
            n_iter=N_ITER_SEARCH,
            scoring=SCORER,
            cv=inner,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=0,
            refit=True,  # refit on full train with best params
        )
        tuner.fit(Xtr, ytr)

        best_est = tuner.best_estimator_
        # predict held-out
        y_pred[test_idx] = best_est.predict(Xte)

        split_records.append({
            "split": i,
            "test_index": int(test_idx[0]),
            "best_params": tuner.best_params_,
            "inner_cv_neg_mse": float(tuner.best_score_),  # neg MSE
            "inner_cv_rmse": float(np.sqrt(-tuner.best_score_))
        })

        if i % 25 == 0 or i == len(y):
            print(f"  LOOCV progress: {i}/{len(y)}")

    # Metrics
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2   = r2_score(y, y_pred)
    print(f"[{target_col}] LOOCV tuned RMSE: {rmse:.4f} {units_label} | R²: {r2:.4f}")

    # Save per-split results
    preds_df = pd.DataFrame({
        "index": np.arange(len(y)),
        "y_obs": y.values,
        "y_pred": y_pred
    })
    splits_df = pd.DataFrame(split_records)
    splits_df["inner_cv_rmse"] = splits_df["inner_cv_rmse"].astype(float)

    out_prefix = target_col.replace('.', '_')
    preds_df.to_csv(os.path.join(OUT_DIR, f"{out_prefix}_loocv_predictions.csv"), index=False)
    splits_df.to_csv(os.path.join(OUT_DIR, f"{out_prefix}_loocv_split_tuning.csv"), index=False)

    # Save LOOCV metrics (per-target)
    metrics_df = pd.DataFrame({
        "target": [target_col],
        "n": [len(y)],
        "n_predictors": [X.shape[1]],
        "loocv_rmse": [rmse],
        "loocv_r2": [r2]
    })
    metrics_path = os.path.join(OUT_DIR, f"{out_prefix}_loocv_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # ---- Add to global metrics for violin plot later ----
    GLOBAL_METRICS.append({
        "target": target_col,
        "n": len(y),
        "n_predictors": X.shape[1],
        "loocv_rmse": rmse,
        "loocv_r2": r2
    })

    # Plot 1:1
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=preds_df["y_obs"], y=preds_df["y_pred"], s=18, edgecolor=None)
    lo = float(np.nanmin([preds_df["y_obs"].min(), preds_df["y_pred"].min()]))
    hi = float(np.nanmax([preds_df["y_obs"].max(), preds_df["y_pred"].max()]))
    plt.plot([lo, hi], [lo, hi], 'k--', lw=2, label='1:1 Line')
    plt.xlabel(f"Observed {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"{target_col}: Nested-LOOCV Obs vs Pred (RF)\nRMSE={rmse:.3f} {units_label}, R²={r2:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"{out_prefix}_loocv_obs_pred.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # ----------------- Choose consensus hyperparameters -----------------
    # Strategy: group identical param dicts, compute median inner-CV RMSE per group, pick best (lowest)
    # Normalize dicts into JSONable tuples so they can be grouped
    def normalize_params(d):
        # For float max_features from uniform, keep as float; bootstrap is bool; ints as ints
        return tuple(sorted(d.items(), key=lambda x: x[0]))

    grp = defaultdict(list)
    for _, row in splits_df.iterrows():
        grp[normalize_params(row["best_params"])].append(float(row["inner_cv_rmse"]))

    summary_rows = []
    for params_key, rmses in grp.items():
        summary_rows.append({
            "params_key": params_key,
            "median_inner_rmse": float(np.median(rmses)),
            "count": len(rmses)
        })
    params_summary = pd.DataFrame(summary_rows).sort_values(
        ["median_inner_rmse", "count"], ascending=[True, False]
    ).reset_index(drop=True)

    params_summary.to_csv(os.path.join(OUT_DIR, f"{out_prefix}_param_summary.csv"), index=False)

    # Best overall param set
    best_key = params_summary.loc[0, "params_key"]
    best_params = dict(best_key)

    # ----------------- Fit final model on ALL data with best params -----------------
    final_model = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=N_JOBS, **best_params
    )
    final_model.fit(X, y)

    # Save model + metadata
    model_path = os.path.join(MODEL_DIR, f"rf_final_{out_prefix}.joblib")
    joblib.dump(final_model, model_path)

    meta = {
        "target": target_col,
        "n_samples": int(len(y)),
        "n_predictors": int(X.shape[1]),
        "units": units_label,
        "loocv_rmse": float(rmse),
        "loocv_r2": float(r2),
        "final_params": best_params,
        "search_config": {
            "inner_folds": INNER_FOLDS,
            "n_iter_search": N_ITER_SEARCH,
            "random_state": RANDOM_STATE
        },
        "files": {
            "predictions_csv": os.path.relpath(os.path.join(OUT_DIR, f"{out_prefix}_loocv_predictions.csv"), OUT_DIR),
            "split_tuning_csv": os.path.relpath(os.path.join(OUT_DIR, f"{out_prefix}_loocv_split_tuning.csv"), OUT_DIR),
            "param_summary_csv": os.path.relpath(os.path.join(OUT_DIR, f"{out_prefix}_param_summary.csv"), OUT_DIR),
            "plot_png": os.path.relpath(plot_path, OUT_DIR),
            "model_joblib": os.path.relpath(model_path, OUT_DIR)
        }
    }
    with open(os.path.join(OUT_DIR, f"{out_prefix}_final_model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved tuned LOOCV products for [{target_col}]")
    print(f"  Model  → {model_path}")
    print(f"  Meta   → {os.path.join(OUT_DIR, f'{out_prefix}_final_model_metadata.json')}")
    print(f"  Plots/CSVs in {OUT_DIR}")

# ----------------- RUN FOR EACH TARGET -----------------
for tcol, units in targets:
    run_target_nested_loocv(tcol, units)

# ----------------- GLOBAL VIOLIN PLOT OF LOOCV R² -----------------
if GLOBAL_METRICS:
    global_df = pd.DataFrame(GLOBAL_METRICS)
    global_csv_path = os.path.join(OUT_DIR, "all_targets_loocv_metrics.csv")
    global_df.to_csv(global_csv_path, index=False)

    plt.figure(figsize=(8, 6))
    sns.violinplot(x="target", y="loocv_r2", data=global_df, inner="point", cut=0)
    plt.title("Distribution of LOOCV R² by Target")
    plt.xlabel("Target")
    plt.ylabel("LOOCV R²")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    violin_path = os.path.join(OUT_DIR, "all_targets_loocv_r2_violin.png")
    plt.savefig(violin_path, dpi=150)
    plt.close()

    print(f"\nSaved global LOOCV metrics CSV → {global_csv_path}")
    print(f"Saved LOOCV R² violin plot   → {violin_path}")
else:
    print("\nNo global metrics collected; skipping violin plot and global CSV.")

print("\nDone.")
