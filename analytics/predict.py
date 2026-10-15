"""Projections, lineup optimization, and Monte Carlo sampling on top of the trained model."""

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from . import data, model

Z90 = 1.2816  # z-score of the 90th percentile
OUT_STATUSES = {"Out", "IR", "PUP", "Suspended", "NA", "Doubtful"}
SLOT_ELIGIBLE = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "K": {"K"}, "DEF": {"DEF"},
    "FLEX": {"RB", "WR", "TE"}, "REC_FLEX": {"WR", "TE"}, "WRRB_FLEX": {"WR", "RB"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}
# Fill the most restrictive slots first so the greedy fill is optimal in practice.
SLOT_ORDER = ["QB", "K", "DEF", "TE", "RB", "WR", "REC_FLEX", "WRRB_FLEX", "FLEX", "SUPER_FLEX"]
PRED_COLS = ["pred_fp_mean", "pred_fp_q10", "pred_fp_q50", "pred_fp_q90"]

_cache: dict = {}


def _memo(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


# ---------- raw model projections ----------

def model_weeks(season: int, current_week: int, last_week: int = 18) -> pd.DataFrame:
    """Model output (PPR) for every skill player for current_week..last_week."""
    def run():
        df = data.build(season, current_week, max_age=6 * 3600)
        rows = [df[(df.season == season) & (df.week == current_week)]]
        future = list(range(current_week + 1, last_week + 1))
        if future:
            rows.append(data.future_rows(df, season, current_week, future))
        rows = pd.concat(rows, ignore_index=True)
        models, meta = model.load()
        p = model.predict(models, rows)
        k = meta.get("interval_scale", 1.0)  # calibrated so q10–q90 covers 80%
        p["pred_fp_q10"] = p.pred_fp_q50 - k * (p.pred_fp_q50 - p.pred_fp_q10)
        p["pred_fp_q90"] = p.pred_fp_q50 + k * (p.pred_fp_q90 - p.pred_fp_q50)
        return p[["sleeper_id", "week", "position", "team", "opponent_team", "implied_total",
                  "sleeper_proj", *PRED_COLS, "pred_receptions", "pred_passing_tds"]
                 ].dropna(subset=["sleeper_id"])
    return _memo(("model_weeks", season, current_week, last_week), 3 * 3600, run)


def _sleeper_week(season: int, week: int) -> pd.DataFrame:
    """Sleeper projections incl. K/DEF, used for positions/players the model doesn't cover."""
    def run():
        params = [("season_type", "regular")] + [("position[]", p) for p in
                                                 ["QB", "RB", "WR", "TE", "K", "DEF"]]
        raw = httpx.get(f"https://api.sleeper.app/projections/nfl/{season}/{week}",
                        params=params, timeout=60).json()
        rows = []
        for r in raw:
            st = r.get("stats") or {}
            if st.get("pts_ppr") is None:
                continue
            rows.append({"sleeper_id": str(r["player_id"]), "week": week,
                         "position": (r.get("player") or {}).get("position"),
                         "opponent_team": r.get("opponent"), "sleeper_proj": st["pts_ppr"],
                         "pred_receptions": st.get("rec") or 0, "pred_passing_tds": st.get("pass_td") or 0})
        return pd.DataFrame(rows)
    return _memo(("sleeper_week", season, week), 3 * 3600, run)


# ---------- league-scored projections ----------

def projections(season: int, current_week: int, weeks: list[int], scoring: dict,
                players: dict) -> pd.DataFrame:
    """Per player per week projections rescored for the league. Index: (sleeper_id, week)."""
    m = model_weeks(season, current_week, max(weeks))
    m = m[m.week.isin(weeks)].assign(source="model")
    ratios = (m.assign(lo=m.pred_fp_q10 / m.pred_fp_q50.clip(lower=1),
                       hi=m.pred_fp_q90 / m.pred_fp_q50.clip(lower=1))
              .groupby("position")[["lo", "hi"]].median())

    fbs = []
    for w in weeks:
        fb = _sleeper_week(season, w)
        if fb.empty:
            continue
        fb = fb[~fb.sleeper_id.isin(m[m.week == w].sleeper_id)].copy()
        fb["pred_fp_mean"] = fb.sleeper_proj
        fb["pred_fp_q50"] = fb.sleeper_proj * 0.95  # fantasy scores are right-skewed
        fb["pred_fp_q10"] = fb.pred_fp_q50 * fb.position.map(ratios.lo).fillna(0.35)
        fb["pred_fp_q90"] = fb.pred_fp_q50 * fb.position.map(ratios.hi).fillna(1.8)
        fb["source"] = "sleeper"
        fbs.append(fb)
    p = pd.concat([m, *fbs], ignore_index=True)

    # Model predicts PPR with 4-pt passing TDs; adjust for this league's scoring.
    skill = p.position.isin(["QB", "RB", "WR", "TE"])
    adj = ((scoring.get("rec", 0) - 1) * p.pred_receptions
           + np.where(p.position == "TE", scoring.get("bonus_rec_te", 0) * p.pred_receptions, 0)
           + (scoring.get("pass_td", 4) - 4) * p.pred_passing_tds).where(skill, 0).fillna(0)
    for c in PRED_COLS:
        p[c] = p[c] + adj

    info = lambda i, k: (players.get(i) or {}).get(k)
    p["name"] = p.sleeper_id.map(lambda i: info(i, "full_name") or i)
    p["position"] = p.position.fillna(p.sleeper_id.map(lambda i: info(i, "position")))
    p["team"] = p.sleeper_id.map(lambda i: info(i, "team"))
    p["injury"] = p.sleeper_id.map(lambda i: info(i, "injury_status"))
    # Current injury designations only apply to this week; long-term ones (IR/PUP) apply to all.
    out_now = p.injury.isin(OUT_STATUSES) & (p.week == current_week)
    out_long = p.injury.isin({"IR", "PUP", "Suspended", "NA"})
    p.loc[out_now | out_long, PRED_COLS] = 0.0
    return p.drop_duplicates(["sleeper_id", "week"]).set_index(["sleeper_id", "week"]).sort_index()


def week_table(proj: pd.DataFrame, week: int) -> pd.DataFrame:
    return proj.xs(week, level="week") if week in proj.index.get_level_values("week") else proj.iloc[:0].droplevel("week")


# ---------- lineups ----------

def optimal_lineup(roster_ids: list[str], wk: pd.DataFrame, slots: list[str],
                   key: str = "pred_fp_mean", locked: dict | None = None) -> list[tuple]:
    """Greedy optimal lineup for one week. `locked` maps slot index -> player already locked in.
    Returns [(slot, player_id or None)] for the league's starting slots."""
    locked = locked or {}
    pool = [(pid, wk.at[pid, "position"], float(wk.at[pid, key]))
            for pid in roster_ids if pid in wk.index and pid not in locked.values()]
    pool.sort(key=lambda x: -x[2])
    starting = [s for s in slots if s in SLOT_ELIGIBLE]
    used, filled = set(), dict(locked)
    for slot_type in SLOT_ORDER:
        for i, s in enumerate(starting):
            if s != slot_type or i in filled:
                continue
            for pid, pos, _ in pool:
                if pid not in used and pos in SLOT_ELIGIBLE[s]:
                    filled[i] = pid
                    used.add(pid)
                    break
    return [(s, filled.get(i)) for i, s in enumerate(starting)]


# ---------- simulation ----------

def sample(wk: pd.DataFrame, ids: list[str], n: int, rng) -> np.ndarray:
    """n x len(ids) simulated scores from a two-piece normal fit to q10/q50/q90."""
    ids = [i for i in ids if i in wk.index]
    if not ids:
        return np.zeros((n, 0))
    q10, q50, q90 = (wk.loc[ids, c].to_numpy(float) for c in ["pred_fp_q10", "pred_fp_q50", "pred_fp_q90"])
    lo_sd = np.maximum(q50 - q10, 0.0) / Z90
    hi_sd = np.maximum(q90 - q50, 0.0) / Z90
    z = rng.standard_normal((n, len(ids)))
    return np.maximum(q50 + np.where(z < 0, z * lo_sd, z * hi_sd), -2.0)


# ---------- game status ----------

def finished_teams(season: int, week: int) -> set[str]:
    """NFL teams whose game this week has finished (kickoff + 3.5h has passed), in Sleeper abbrs."""
    def run():
        sc = data._retry(data.nfl.load_schedules, [season])
        return sc[(sc.week == week) & (sc.game_type == "REG")][["home_team", "away_team", "gameday", "gametime"]]
    sc = _memo(("sched", season, week), 3600, run)
    now = datetime.now(ZoneInfo("America/New_York"))
    done = set()
    for r in sc.itertuples():
        kick = datetime.fromisoformat(f"{r.gameday}T{r.gametime or '13:00'}").replace(
            tzinfo=ZoneInfo("America/New_York"))
        if now > kick + timedelta(hours=3, minutes=30):
            done |= {to_sleeper_team(r.home_team), to_sleeper_team(r.away_team)}
    return done


def to_sleeper_team(t: str) -> str:
    return {"LA": "LAR"}.get(t, t)
