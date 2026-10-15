"""LightGBM projection models: median + 10th/90th percentile PPR points, plus
receptions and passing TDs so projections can be rescored for any league settings."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import FEATURES, sleeper_projections

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

# name -> (target column, objective, extra params)
HEADS = {
    "fp_mean": ("fp", "regression", {}),
    "fp_q10": ("fp", "quantile", {"alpha": 0.1}),
    "fp_q50": ("fp", "quantile", {"alpha": 0.5}),
    "fp_q90": ("fp", "quantile", {"alpha": 0.9}),
    "receptions": ("receptions", "regression", {}),
    "passing_tds": ("passing_tds", "regression", {}),
}


def _train_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Need at least one prior game so the form features mean something.
    return df[df.fp.notna() & (df.career_games > 0)]


def fit(df: pd.DataFrame, min_season: int = 0) -> dict:
    tr = _train_rows(df)
    tr = tr[tr.season >= min_season]
    models = {}
    for name, (target, obj, extra) in HEADS.items():
        m = lgb.LGBMRegressor(objective=obj, **PARAMS, **extra)
        m.fit(tr[FEATURES], tr[target].fillna(0))
        models[name] = m
    return models


def predict(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, m in models.items():
        out[f"pred_{name}"] = m.predict(df[FEATURES])
    # Quantile heads are trained independently; enforce q10 <= q50 <= q90.
    q = np.sort(out[["pred_fp_q10", "pred_fp_q50", "pred_fp_q90"]].values, axis=1)
    out["pred_fp_q10"], out["pred_fp_q50"], out["pred_fp_q90"] = q[:, 0], q[:, 1], q[:, 2]
    out["pred_receptions"] = out.pred_receptions.clip(lower=0)
    out["pred_passing_tds"] = out.pred_passing_tds.clip(lower=0)
    return out


def save(models: dict, meta: dict):
    joblib.dump(models, MODEL_DIR / "models.joblib")
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load() -> tuple[dict, dict]:
    return joblib.load(MODEL_DIR / "models.joblib"), json.loads((MODEL_DIR / "meta.json").read_text())


# ---------- backtest ----------

def backtest(df: pd.DataFrame, test_season: int, min_season: int = 0) -> dict:
    """Train on seasons before test_season, score every week of test_season."""
    train = df[df.season < test_season]
    test = _train_rows(df[df.season == test_season])
    preds = predict(fit(train, min_season), test)

    both = preds[preds.sleeper_proj.notna()]

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def start_sit(d, col, n_pairs=200_000, seed=0):
        """% of same-position, same-week player pairs where `col` ranks the higher scorer first.
        Restricted to fantasy-relevant players (Sleeper projection >= 5)."""
        d = d[d.sleeper_proj >= 5].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        keys = d.groupby(["week", "position"]).indices
        groups = [ix for ix in keys.values() if len(ix) > 1]
        sizes = np.array([len(g) for g in groups], dtype=float)
        pick = rng.choice(len(groups), n_pairs, p=sizes / sizes.sum())
        a = np.array([rng.choice(groups[i]) for i in pick])
        b = np.array([rng.choice(groups[i]) for i in pick])
        keep = (a != b) & (d.fp.values[a] != d.fp.values[b])
        a, b = a[keep], b[keep]
        return float(np.mean((d[col].values[a] > d[col].values[b]) == (d.fp.values[a] > d.fp.values[b])))

    cover = float(np.mean((preds.fp >= preds.pred_fp_q10) & (preds.fp <= preds.pred_fp_q90)))

    # Interval scale that makes the q10–q90 band cover 80% of outcomes.
    def coverage(k):
        lo = preds.pred_fp_q50 - k * (preds.pred_fp_q50 - preds.pred_fp_q10)
        hi = preds.pred_fp_q50 + k * (preds.pred_fp_q90 - preds.pred_fp_q50)
        return float(np.mean((preds.fp >= lo) & (preds.fp <= hi)))
    interval_scale = min(np.arange(1.0, 1.6, 0.01), key=lambda k: abs(coverage(k) - 0.80))
    res = {
        "test_season": test_season,
        "rows": len(preds),
        "mae_model": mae(preds.fp, preds.pred_fp_mean),
        "mae_baseline_recent_avg": mae(preds.fp, preds.fp_ewm.fillna(preds.fp_ewm.mean())),
        "q10_q90_coverage": cover,  # should be ~0.80
        "interval_scale": float(interval_scale),
        "coverage_after_scaling": coverage(interval_scale),
        "head_to_head_vs_sleeper": {
            "rows": len(both),
            "mae_model": mae(both.fp, both.pred_fp_mean),
            "mae_sleeper": mae(both.fp, both.sleeper_proj),
            "mae_blend_50_50": mae(both.fp, (both.pred_fp_mean + both.sleeper_proj) / 2),
            "mae_model_median": mae(both.fp, both.pred_fp_q50),
            "rmse_model": rmse(both.fp, both.pred_fp_mean),
            "rmse_sleeper": rmse(both.fp, both.sleeper_proj),
            "start_sit_correct_model": start_sit(both, "pred_fp_mean"),
            "start_sit_correct_sleeper": start_sit(both, "sleeper_proj"),
        },
        "by_position": {
            pos: {"mae_model": mae(g.fp, g.pred_fp_mean), "mae_sleeper": mae(g.fp, g.sleeper_proj)}
            for pos, g in both.groupby("position")
        },
    }
    return res
