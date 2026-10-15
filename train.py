"""Retrain the projection model. Run weekly (e.g. Tuesday after games):  uv run python train.py"""

import json
from datetime import date

import numpy as np

from analytics import data, model
from sleeper_client import _state


def main():
    st = _state()
    season, week = int(st["season"]), int(st["week"])
    df = data.build(season, week, refresh=True)

    results = [model.backtest(df, y) for y in (season - 2, season - 1)]
    models = model.fit(df)
    meta = {
        "trained_on": str(date.today()),
        "season": season,
        "through_week": week - 1,
        "interval_scale": float(np.mean([r["interval_scale"] for r in results])),
        "backtests": results,
    }
    model.save(models, meta)
    for r in results:
        h = r["head_to_head_vs_sleeper"]
        print(f"{r['test_season']}: start/sit {h['start_sit_correct_model']:.1%} (model) vs "
              f"{h['start_sit_correct_sleeper']:.1%} (Sleeper) | RMSE {h['rmse_model']:.2f} vs "
              f"{h['rmse_sleeper']:.2f} | median MAE {h['mae_model_median']:.2f} vs {h['mae_sleeper']:.2f} | "
              f"80% band covers {r['coverage_after_scaling']:.1%}")
    print(json.dumps({k: v for k, v in meta.items() if k != "backtests"}, indent=1))


if __name__ == "__main__":
    main()
