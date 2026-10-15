"""Builds the player-week feature table from nflverse data.

Every feature for week W uses only information available before week W kicks off
(prior-game rolling stats, pregame Vegas lines, opponent defense history), so the
same table is used for training, backtesting, and predicting the upcoming week.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import nflreadpy as nfl
from nflreadpy import config as nfl_config
import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / ".cache"
CACHE.mkdir(exist_ok=True)
nfl_config.update_config(cache_mode="filesystem", cache_dir=CACHE / "nflverse", cache_duration=12 * 3600, verbose=False)
FIRST_SEASON = 2016
POSITIONS = ["QB", "RB", "WR", "TE"]

# Per-game stats that get rolled into "recent form" features.
FORM_STATS = [
    "fp", "xfp", "fp_over_xfp", "targets", "receptions", "carries", "attempts",
    "passing_yards", "passing_tds", "rushing_yards", "receiving_yards", "receiving_air_yards",
    "target_share", "air_yards_share", "wopr", "offense_pct", "passing_epa", "rushing_epa",
    "receiving_epa", "rz_opps",
]


SLEEPER_PROJ_FIRST_SEASON = 2018


def sleeper_projections(season: int, week: int, max_age: int | None = None) -> pd.DataFrame:
    """Sleeper's own weekly PPR projections (all skill positions). Past weeks are cached forever."""
    path = CACHE / f"sleeper_proj_{season}_{week}.json"
    if path.exists() and (max_age is None or time.time() - path.stat().st_mtime < max_age):
        data = json.loads(path.read_text())
    else:
        params = [("season_type", "regular")] + [("position[]", p) for p in POSITIONS]
        data = httpx.get(f"https://api.sleeper.app/projections/nfl/{season}/{week}",
                         params=params, timeout=60).json()
        path.write_text(json.dumps(data))
    rows = [{"sleeper_id": str(r["player_id"]), "sleeper_proj": (r.get("stats") or {}).get("pts_ppr")}
            for r in data if (r.get("stats") or {}).get("pts_ppr") is not None]
    return pd.DataFrame(rows, columns=["sleeper_id", "sleeper_proj"]).assign(season=season, week=week)


def _all_sleeper_projections(weeks: list[tuple[int, int]], current_season: int) -> pd.DataFrame:
    def one(sw):
        s, w = sw
        return sleeper_projections(s, w, max_age=3 * 3600 if s == current_season else None)
    with ThreadPoolExecutor(8) as ex:
        return pd.concat(list(ex.map(one, weeks)), ignore_index=True)


def _retry(fn, *args, **kwargs):
    """nflverse files are served from GitHub releases, which occasionally 500s."""
    for attempt in range(4):
        try:
            return fn(*args, **kwargs).to_pandas()
        except ConnectionError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def _seasons(current: int) -> list[int]:
    return list(range(FIRST_SEASON, current + 1))


def _player_weeks(seasons: list[int]) -> pd.DataFrame:
    ps = _retry(nfl.load_player_stats, seasons)
    ps = ps[(ps.season_type == "REG") & ps.position.isin(POSITIONS)].copy()
    ps = ps.rename(columns={"fantasy_points_ppr": "fp"})

    opp = _retry(nfl.load_ff_opportunity, seasons=seasons)
    opp = opp[["season", "week", "player_id", "total_fantasy_points_exp",
               "rec_touchdown_exp", "rush_touchdown_exp"]].copy()
    opp["season"] = opp["season"].astype(int)
    opp["week"] = opp["week"].astype(int)
    opp = opp.rename(columns={"total_fantasy_points_exp": "xfp"})
    # Expected TDs are a good proxy for red-zone / goal-line opportunity.
    opp["rz_opps"] = opp.rec_touchdown_exp.fillna(0) + opp.rush_touchdown_exp.fillna(0)
    ps = ps.merge(opp[["season", "week", "player_id", "xfp", "rz_opps"]],
                  on=["season", "week", "player_id"], how="left")
    ps["fp_over_xfp"] = ps.fp - ps.xfp

    ids = _retry(nfl.load_ff_playerids)[["gsis_id", "pfr_id", "sleeper_id", "birthdate", "draft_year"]]
    ids = ids.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    ids["sleeper_id"] = ids.sleeper_id.map(lambda x: str(int(x)) if pd.notna(x) else None)
    ps = ps.merge(ids, left_on="player_id", right_on="gsis_id", how="left")

    snaps = _retry(nfl.load_snap_counts, seasons)
    snaps = snaps[snaps.game_type == "REG"][["season", "week", "pfr_player_id", "offense_pct"]]
    ps = ps.merge(snaps, left_on=["season", "week", "pfr_id"],
                  right_on=["season", "week", "pfr_player_id"], how="left")
    return ps


def _team_games(seasons: list[int]) -> pd.DataFrame:
    """One row per team per game with pregame Vegas context."""
    sc = _retry(nfl.load_schedules, seasons)
    sc = sc[sc.game_type == "REG"]
    rows = []
    for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
        d = pd.DataFrame({
            "season": sc.season, "week": sc.week, "game_id": sc.game_id,
            "team": sc[f"{side}_team"], "opponent_team": sc[f"{other}_team"],
            "is_home": int(side == "home"),
            # spread_line is home-minus-away; positive means the home team is favored.
            "team_spread": sign * sc.spread_line,
            "total_line": sc.total_line,
            "dome": sc.roof.isin(["dome", "closed"]).astype(int),
            "wind": sc.wind, "temp": sc.temp, "gameday": sc.gameday,
            "rest": sc[f"{side}_rest"],
        })
        rows.append(d)
    tg = pd.concat(rows, ignore_index=True)
    tg["implied_total"] = tg.total_line / 2 + tg.team_spread / 2
    return tg


def _defense_vs_position(pw: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Rolling fantasy points each defense allowed to each position, prior games only."""
    allowed = (pw.groupby(["season", "week", "opponent_team", "position"]).fp.sum()
               .rename("fp_allowed").reset_index()
               .rename(columns={"opponent_team": "defense"}))
    # Normalize vs league average for that position so it reads as "+3 pts easier than avg".
    allowed["fp_allowed_rel"] = allowed.fp_allowed - allowed.groupby(
        ["season", "position"]).fp_allowed.transform("mean")

    # Expand to every (defense, position, scheduled game) so upcoming weeks get a value too.
    games = tg[["season", "week", "team"]].rename(columns={"team": "defense"})
    grid = games.merge(pd.DataFrame({"position": POSITIONS}), how="cross")
    d = grid.merge(allowed, on=["season", "week", "defense", "position"], how="left")
    d = d.sort_values(["defense", "position", "season", "week"])
    g = d.groupby(["defense", "position"]).fp_allowed_rel
    d["def_allowed_ewm"] = g.transform(lambda s: s.shift(1).ewm(halflife=6, ignore_na=True).mean())
    return d[["season", "week", "defense", "position", "def_allowed_ewm"]]


REPORT_CODE = {"Questionable": 1, "Doubtful": 2, "Out": 3}
PRACTICE_CODE = {"Full Participation in Practice": 0, "Limited Participation in Practice": 1,
                 "Did Not Participate In Practice": 2}


def _injuries(seasons: list[int]) -> pd.DataFrame:
    inj = _retry(nfl.load_injuries, seasons)
    inj = inj[inj.game_type == "REG"]
    inj = pd.DataFrame({
        "season": inj.season, "week": inj.week, "player_id": inj.gsis_id,
        "inj_report": inj.report_status.map(REPORT_CODE),
        "inj_practice": inj.practice_status.map(PRACTICE_CODE),
    })
    return inj.groupby(["season", "week", "player_id"], as_index=False).max()


def _add_injury_features(df: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(inj, on=["season", "week", "player_id"], how="left")
    df[["inj_report", "inj_practice"]] = df[["inj_report", "inj_practice"]].fillna(0)

    # Opportunity vacated by teammates ruled Out/Doubtful. Those teammates have no stat row
    # for the week, so look up their most recent form from any earlier row.
    out = inj[inj.inj_report >= 2].copy()
    out["key"] = (out.season * 100 + out.week).astype("int64")
    latest = df[["player_id", "season", "week", "team", "targets_ewm", "carries_ewm", "attempts_ewm"]].copy()
    latest["key"] = (latest.season * 100 + latest.week + 1).astype("int64")
    out = pd.merge_asof(out.sort_values("key"),
                        latest.drop(columns=["season", "week"]).sort_values("key"),
                        on="key", by="player_id", direction="backward")
    out = out[out.team.notna()]
    agg = out.groupby(["season", "week", "team"]).agg(
        team_out_targets=("targets_ewm", "sum"),
        team_out_carries=("carries_ewm", "sum"),
        team_qb_out=("attempts_ewm", lambda a: float((a.fillna(0) > 15).any())),
    ).reset_index()
    df = df.merge(agg, on=["season", "week", "team"], how="left")
    df[["team_out_targets", "team_out_carries", "team_qb_out"]] = df[
        ["team_out_targets", "team_out_carries", "team_qb_out"]].fillna(0)
    return df


def _add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    g = df.groupby("player_id")
    for c in FORM_STATS:
        prev = g[c].shift(1)
        df[f"{c}_ewm"] = prev.groupby(df.player_id).transform(
            lambda s: s.ewm(halflife=3, ignore_na=True).mean())
        df[f"{c}_last"] = prev
    # Season-to-date and last-season averages.
    gs = df.groupby(["player_id", "season"])
    df["fp_std_season"] = gs.fp.transform(lambda s: s.shift(1).expanding().mean())
    df["games_this_season"] = gs.cumcount()
    df["fp_volatility"] = g.fp.transform(lambda s: s.shift(1).rolling(8, min_periods=3).std())
    season_avg = df.groupby(["player_id", "season"]).fp.mean().rename("fp_prev_season").reset_index()
    season_avg["season"] += 1
    df = df.merge(season_avg, on=["player_id", "season"], how="left")
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    df["career_games"] = df.groupby("player_id").cumcount()
    return df


def build(current_season: int, upcoming_week: int | None = None, refresh: bool = False,
          max_age: float | None = None) -> pd.DataFrame:
    """Feature table for all player-weeks, plus rows for `upcoming_week` (fp is NaN there)."""
    path = CACHE / f"features_{current_season}_{upcoming_week}.parquet"
    fresh = path.exists() and (max_age is None or time.time() - path.stat().st_mtime < max_age)
    if fresh and not refresh:
        return pd.read_parquet(path)

    seasons = _seasons(current_season)
    pw = _player_weeks(seasons)
    tg = _team_games(seasons)

    if upcoming_week:
        # Players who appeared this season and are still on a team → candidate rows for the upcoming week.
        recent = pw[pw.season == current_season].sort_values("week").groupby("player_id").tail(1)
        up = recent[["player_id", "player_display_name", "position", "team", "gsis_id", "pfr_id",
                     "sleeper_id", "birthdate", "draft_year"]].copy()
        up["season"], up["week"], up["fp"] = current_season, upcoming_week, np.nan
        pw = pd.concat([pw, up], ignore_index=True)

    pw = pw.drop(columns=["opponent_team", "game_id"], errors="ignore")
    pw = pw.merge(tg, on=["season", "week", "team"], how="inner")  # drops bye weeks for upcoming rows
    pw = _add_form_features(pw)
    pw = _add_injury_features(pw, _injuries(seasons))

    dvp = _defense_vs_position(pw[pw.fp.notna()], tg)
    pw = pw.merge(dvp, left_on=["season", "week", "opponent_team", "position"],
                  right_on=["season", "week", "defense", "position"], how="left")

    weeks = sorted({(int(a), int(b)) for a, b in pw[["season", "week"]].itertuples(index=False)
                    if a >= SLEEPER_PROJ_FIRST_SEASON})
    sp = _all_sleeper_projections(weeks, current_season).drop_duplicates(["sleeper_id", "season", "week"])
    pw = pw.merge(sp, on=["sleeper_id", "season", "week"], how="left")

    pw["age"] = (pd.to_datetime(pw.gameday) - pd.to_datetime(pw.birthdate, errors="coerce")).dt.days / 365.25
    pw["years_exp"] = pw.season - pw.draft_year
    pw["pos_code"] = pw.position.map({p: i for i, p in enumerate(POSITIONS)})
    pw.to_parquet(path)
    return pw


FEATURES = (
    [f"{c}_ewm" for c in FORM_STATS] + [f"{c}_last" for c in FORM_STATS] + [
        "fp_std_season", "games_this_season", "fp_prev_season", "career_games", "fp_volatility",
        "implied_total", "team_spread", "total_line", "is_home", "dome", "wind", "temp", "rest",
        "def_allowed_ewm", "sleeper_proj", "inj_report", "inj_practice",
        "team_out_targets", "team_out_carries", "team_qb_out", "age", "years_exp", "pos_code", "week",
    ]
)


GAME_LEVEL = ["opponent_team", "game_id", "is_home", "team_spread", "total_line", "dome", "wind", "temp",
              "gameday", "rest", "implied_total", "def_allowed_ewm", "sleeper_proj", "inj_report",
              "inj_practice", "team_out_targets", "team_out_carries", "team_qb_out", "week"]


def future_rows(df: pd.DataFrame, season: int, upcoming_week: int, weeks: list[int]) -> pd.DataFrame:
    """Feature rows for later weeks: each player's current form paired with each future matchup.
    Vegas lines are only published ~2 weeks out, so far-off weeks fall back to the team's
    season-average implied total."""
    base = df[(df.season == season) & (df.week == upcoming_week)]
    player_cols = [c for c in base.columns if c not in GAME_LEVEL]
    base = base[player_cols]

    tg_all = _team_games(_seasons(season))
    dvp = _defense_vs_position(df[df.fp.notna()], tg_all)
    tg = tg_all[(tg_all.season == season) & tg_all.week.isin(weeks)]
    avg_implied = tg_all[tg_all.season == season].groupby("team").implied_total.mean()

    fut = base.drop(columns=["season"]).merge(tg, on="team", how="inner")
    fut["implied_total"] = fut.implied_total.fillna(fut.team.map(avg_implied))
    fut = fut.merge(dvp, left_on=["season", "week", "opponent_team", "position"],
                    right_on=["season", "week", "defense", "position"], how="left", suffixes=("", "_d"))
    sp = _all_sleeper_projections([(season, w) for w in weeks], season).drop_duplicates(
        ["sleeper_id", "season", "week"])
    fut = fut.merge(sp, on=["sleeper_id", "season", "week"], how="left")
    for c in ["inj_report", "inj_practice", "team_out_targets", "team_out_carries", "team_qb_out"]:
        fut[c] = 0.0
    return fut
