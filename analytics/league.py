"""League-level analytics: lineups, matchup sims, playoff odds, ROS trade analysis, buy-low/sell-high."""

import numpy as np
import pandas as pd

from sleeper_client import SLEEPER, _get, _players, _state, _user_id

from . import data, predict as P

LAST_WEEK = 18


# ---------- shared context ----------

class League:
    def __init__(self, league_id: str):
        self.id = league_id
        self.lg = _get(f"{SLEEPER}/league/{league_id}")
        self.scoring = self.lg.get("scoring_settings", {})
        self.slots = self.lg.get("roster_positions", [])
        s = self.lg.get("settings", {})
        self.playoff_start = s.get("playoff_week_start") or 15
        self.playoff_teams = s.get("playoff_teams") or 6
        self.rosters = _get(f"{SLEEPER}/league/{league_id}/rosters")
        users = _get(f"{SLEEPER}/league/{league_id}/users")
        self.names = {u["user_id"]: u["display_name"] for u in users}
        self.players = _players()
        st = _state()
        self.season, self.week = int(st["season"]), int(st["week"])

    def owner(self, r) -> str:
        return self.names.get(r["owner_id"], str(r["roster_id"]))

    def roster_of(self, username: str = "") -> dict:
        uid = _user_id(username)
        for r in self.rosters:
            if r.get("owner_id") == uid or uid in (r.get("co_owners") or []):
                return r
        raise ValueError(f"No roster for '{username or 'you'}' in this league")

    def roster_by_id(self, rid: int) -> dict:
        return next(r for r in self.rosters if r["roster_id"] == rid)

    def proj(self, weeks: list[int]) -> pd.DataFrame:
        return P.projections(self.season, self.week, weeks, self.scoring, self.players)

    def name(self, pid: str) -> str:
        p = self.players.get(pid) or {}
        return p.get("full_name") or pid


def _row(lg: League, wk: pd.DataFrame, pid: str | None, slot: str | None = None) -> dict:
    if pid is None:
        return {"slot": slot, "player": "EMPTY"}
    d = {"slot": slot, "player": lg.name(pid), "player_id": pid}
    if pid in wk.index:
        r = wk.loc[pid]
        d.update(pos=r.position, opp=r.opponent_team, injury=r.injury,
                 proj=round(float(r.pred_fp_mean), 1), floor=round(float(r.pred_fp_q10), 1),
                 ceiling=round(float(r.pred_fp_q90), 1),
                 sleeper_ppr_proj=None if pd.isna(r.sleeper_proj) else round(float(r.sleeper_proj), 1))
    else:
        inj = (lg.players.get(pid) or {}).get("injury_status")
        d.update(proj=0.0, injury=inj, note=f"not playing ({inj})" if inj else "bye week")
    return {k: v for k, v in d.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}


# ---------- lineup ----------

def lineup(league_id: str, username: str = "", week: int = 0) -> dict:
    lg = League(league_id)
    week = week or lg.week
    wk = P.week_table(lg.proj([week]), week)
    r = lg.roster_of(username)
    roster = r.get("players") or []
    current = r.get("starters") or []
    best = P.optimal_lineup(roster, wk, lg.slots)
    best_ids = {p for _, p in best if p}
    cur_total = sum(float(wk.at[p, "pred_fp_mean"]) for p in current if p in wk.index)
    best_total = sum(float(wk.at[p, "pred_fp_mean"]) for p in best_ids)
    bench = [p for p in roster if p not in best_ids]
    return {
        "week": week,
        "optimal_lineup": [_row(lg, wk, p, s) for s, p in best],
        "bench": sorted([_row(lg, wk, p) for p in bench], key=lambda d: -d.get("proj", 0)),
        "currently_set_projection": round(cur_total, 1),
        "optimal_projection": round(best_total, 1),
        "changes": {"start": [lg.name(p) for p in best_ids - set(current)],
                    "sit": [lg.name(p) for p in set(current) - best_ids if p and p != "0"]},
        "note": "proj = model mean (league scoring); floor/ceiling = 10th/90th percentile outcomes.",
    }


# ---------- matchup simulation ----------

def _locked_actuals(lg: League, week: int) -> tuple[set, dict]:
    done = P.finished_teams(lg.season, week)
    locked = {pid for pid, p in lg.players.items() if p.get("team") in done}
    actual = {}
    for m in _get(f"{SLEEPER}/league/{lg.id}/matchups/{week}"):
        actual.update(m.get("players_points") or {})
    return locked, actual


def _team_samples(wk, starters, n, rng, locked, actual):
    fixed = sum(actual.get(p, 0.0) for p in starters if p in locked)
    live = [p for p in starters if p and p not in locked]
    return fixed + P.sample(wk, live, n, rng).sum(axis=1)


def matchup(league_id: str, username: str = "", week: int = 0, sims: int = 20000) -> dict:
    lg = League(league_id)
    week = week or lg.week
    wk = P.week_table(lg.proj([week]), week)
    me = lg.roster_of(username)
    ms = _get(f"{SLEEPER}/league/{league_id}/matchups/{week}")
    mine = next(m for m in ms if m["roster_id"] == me["roster_id"])
    opp_m = next((m for m in ms if m["matchup_id"] == mine["matchup_id"] and m["roster_id"] != me["roster_id"]), None)
    if not opp_m:
        return {"week": week, "note": "No opponent this week (bye or playoffs not set)."}
    opp = lg.roster_by_id(opp_m["roster_id"])
    locked, actual = _locked_actuals(lg, week)
    rng = np.random.default_rng(7)

    my_set = [p for p in (mine.get("starters") or me.get("starters") or []) if p and p != "0"]
    opp_set = [p for p in (opp_m.get("starters") or opp.get("starters") or []) if p and p != "0"]
    # Opponent assumed to play their optimal lineup (players already played stay as set).
    starting_slots = [s for s in lg.slots if s in P.SLOT_ELIGIBLE]
    opp_locked = {i: p for i, p in enumerate(opp_set) if p in locked}
    opp_best = [p for _, p in P.optimal_lineup(opp.get("players") or [], wk, lg.slots, locked=opp_locked) if p]
    opp_s = _team_samples(wk, opp_best, sims, rng, locked, actual)

    def win_prob(starters):
        my_s = _team_samples(wk, starters, sims, np.random.default_rng(11), locked, actual)
        return float(np.mean(my_s > opp_s)), my_s

    p_set, s_set = win_prob(my_set)
    my_locked = {i: p for i, p in enumerate(my_set) if p in locked}
    my_best = [p for _, p in P.optimal_lineup(me.get("players") or [], wk, lg.slots, locked=my_locked) if p]
    p_best, s_best = win_prob(my_best)

    # Swaps: try every legal bench-for-starter swap and keep the ones that raise win probability.
    swaps = []
    bench = [p for p in me.get("players") or [] if p not in my_set and p in wk.index and p not in locked]
    for i, (slot, starter) in enumerate(zip(starting_slots, my_set)):
        if starter in locked:
            continue
        for b in bench:
            if wk.at[b, "position"] not in P.SLOT_ELIGIBLE.get(slot, set()):
                continue
            trial = my_set.copy()
            trial[i] = b
            pb, _ = win_prob(trial)
            if pb > p_set + 0.005:
                swaps.append({"start": lg.name(b), "instead_of": lg.name(starter), "slot": slot,
                              "win_prob": round(pb, 3), "gain": round(pb - p_set, 3)})
    swaps.sort(key=lambda d: -d["gain"])

    fav = float(np.median(s_set)) >= float(np.median(opp_s))
    return {
        "week": week,
        "opponent": lg.owner(opp),
        "win_prob_as_set": round(p_set, 3),
        "win_prob_optimal_lineup": round(p_best, 3),
        "my_score_range": {"floor": round(float(np.percentile(s_set, 10)), 1),
                           "median": round(float(np.median(s_set)), 1),
                           "ceiling": round(float(np.percentile(s_set, 90)), 1)},
        "opponent_score_range": {"floor": round(float(np.percentile(opp_s, 10)), 1),
                                 "median": round(float(np.median(opp_s)), 1),
                                 "ceiling": round(float(np.percentile(opp_s, 90)), 1)},
        "best_swaps_by_win_probability": swaps[:6],
        "strategy": ("You're favored: prefer high-floor players." if fav else
                     "You're the underdog: prefer high-ceiling players; variance helps you."),
        "my_starters": [_row(lg, wk, p) for p in my_set],
        "opponent_starters": [_row(lg, wk, p) for p in opp_best],
        "games_already_final": sorted(P.finished_teams(lg.season, week)),
    }


# ---------- playoff odds ----------

def playoff_odds(league_id: str, sims: int = 5000) -> dict:
    lg = League(league_id)
    last_reg = lg.playoff_start - 1
    weeks = list(range(lg.week, last_reg + 1))
    if not weeks:
        return {"note": "Regular season is over."}
    proj = lg.proj(weeks)
    rng = np.random.default_rng(3)
    ids = [r["roster_id"] for r in lg.rosters]
    idx = {rid: i for i, rid in enumerate(ids)}
    wins = np.zeros((sims, len(ids)))
    pts = np.zeros((sims, len(ids)))
    for r in lg.rosters:
        s = r.get("settings", {})
        wins[:, idx[r["roster_id"]]] = s.get("wins", 0) + 0.5 * s.get("ties", 0)
        pts[:, idx[r["roster_id"]]] = s.get("fpts", 0) + s.get("fpts_decimal", 0) / 100

    locked, actual = _locked_actuals(lg, lg.week)
    # Rosters change over a season (trades, injuries, waivers), so each simulated season draws a
    # persistent strength multiplier per team; growing with distance from today.
    drift = rng.normal(0, 0.05, (sims, len(ids)))
    for w in weeks:
        wk = P.week_table(proj, w)
        scores = np.zeros((sims, len(ids)))
        for r in lg.rosters:
            if w == lg.week:
                lk = {i: p for i, p in enumerate(r.get("starters") or []) if p in locked}
                best = [p for _, p in P.optimal_lineup(r.get("players") or [], wk, lg.slots, locked=lk) if p]
                scores[:, idx[r["roster_id"]]] = _team_samples(wk, best, sims, rng, locked, actual)
            else:
                best = [p for _, p in P.optimal_lineup(r.get("players") or [], wk, lg.slots) if p]
                scores[:, idx[r["roster_id"]]] = P.sample(wk, best, sims, rng).sum(axis=1)
        scores *= 1 + drift * np.sqrt(w - lg.week)
        pts += scores
        ms = _get(f"{SLEEPER}/league/{league_id}/matchups/{w}")
        pairs: dict = {}
        for m in ms:
            if m.get("matchup_id"):
                pairs.setdefault(m["matchup_id"], []).append(m["roster_id"])
        if not pairs:  # schedule not published: random pairings
            perm = rng.permutation(ids)
            pairs = {i: list(perm[2 * i:2 * i + 2]) for i in range(len(ids) // 2)}
        for a, b in (p for p in pairs.values() if len(p) == 2):
            ia, ib = idx[a], idx[b]
            wins[:, ia] += scores[:, ia] > scores[:, ib]
            wins[:, ib] += scores[:, ib] > scores[:, ia]

    # Rank by wins, then points for.
    order = np.argsort(-(wins * 1e5 + pts), axis=1)
    seed = np.empty_like(order)
    rows = np.arange(sims)[:, None]
    seed[rows, order] = np.arange(len(ids))[None, :]
    out = []
    for r in lg.rosters:
        i = idx[r["roster_id"]]
        s = r.get("settings", {})
        out.append({
            "team": lg.owner(r),
            "record": f"{s.get('wins', 0)}-{s.get('losses', 0)}-{s.get('ties', 0)}",
            "projected_wins": round(float(wins[:, i].mean()), 1),
            "playoff_odds": round(float((seed[:, i] < lg.playoff_teams).mean()), 3),
            "top_seed_odds": round(float((seed[:, i] == 0).mean()), 3),
        })
    out.sort(key=lambda d: -d["playoff_odds"])
    return {"weeks_simulated": weeks, "playoff_teams": lg.playoff_teams, "sims": sims, "teams": out}


# ---------- rest-of-season values & trades ----------

def _ros_weeks(lg: League, through_playoffs: bool) -> list[int]:
    end = LAST_WEEK if through_playoffs else lg.playoff_start - 1
    return list(range(lg.week, min(end, LAST_WEEK) + 1))


def ros_projections(league_id: str, player_ids: list[str], include_playoffs: bool = True) -> list[dict]:
    lg = League(league_id)
    weeks = _ros_weeks(lg, include_playoffs)
    proj = lg.proj(weeks)
    out = []
    for pid in player_ids:
        rows = proj.loc[pid] if pid in proj.index.get_level_values(0) else None
        tot = float(rows.pred_fp_mean.sum()) if rows is not None else 0.0
        out.append({"player": lg.name(pid), "player_id": pid,
                    "ros_points": round(tot, 1), "games": 0 if rows is None else int((rows.pred_fp_mean > 0).sum()),
                    "per_game": round(tot / max(1, len(rows)) if rows is not None else 0, 1),
                    "playoff_weeks_points": round(float(rows[rows.index >= lg.playoff_start].pred_fp_mean.sum()), 1)
                    if rows is not None else 0})
    return out


def _lineup_points(lg, proj, roster, weeks) -> float:
    tot = 0.0
    for w in weeks:
        wk = P.week_table(proj, w)
        tot += sum(float(wk.at[p, "pred_fp_mean"]) for _, p in P.optimal_lineup(roster, wk, lg.slots) if p)
    return tot


def trade_impact(league_id: str, give_player_ids: list[str], get_player_ids: list[str],
                 username: str = "", include_playoffs: bool = True) -> dict:
    """How many starting-lineup points each side gains over the rest of the season."""
    lg = League(league_id)
    weeks = _ros_weeks(lg, include_playoffs)
    proj = lg.proj(weeks)
    me = lg.roster_of(username)
    owners = {p: r for r in lg.rosters for p in r.get("players") or []}
    partner = owners.get(get_player_ids[0]) if get_player_ids else None

    def impact(r, out_ids, in_ids):
        before = r.get("players") or []
        after = [p for p in before if p not in out_ids] + list(in_ids)
        b, a = _lineup_points(lg, proj, before, weeks), _lineup_points(lg, proj, after, weeks)
        return {"lineup_points_before": round(b, 1), "lineup_points_after": round(a, 1), "change": round(a - b, 1)}

    res = {
        "weeks": f"{weeks[0]}-{weeks[-1]}",
        "you_give": ros_projections(league_id, give_player_ids, include_playoffs),
        "you_get": ros_projections(league_id, get_player_ids, include_playoffs),
        "your_lineup_impact": impact(me, give_player_ids, get_player_ids),
    }
    if partner and partner["roster_id"] != me["roster_id"]:
        res["partner"] = lg.owner(partner)
        res["partner_lineup_impact"] = impact(partner, get_player_ids, give_player_ids)
    res["note"] = ("Lineup impact counts only points that would actually be started, so depth you "
                   "can't start is worth ~0 and filling a weak slot is worth a lot.")
    return res


# ---------- buy low / sell high ----------

def buy_sell(league_id: str, username: str = "", min_games: int = 2, limit: int = 12) -> dict:
    """Players scoring far from their expected fantasy points (opportunity-based) this season."""
    lg = League(league_id)
    df = data.build(lg.season, lg.week, max_age=6 * 3600)
    cur = df[(df.season == lg.season) & df.fp.notna() & df.xfp.notna()]
    g = cur.groupby("sleeper_id").agg(games=("fp", "size"), fp_pg=("fp", "mean"), xfp_pg=("xfp", "mean"),
                                       pos=("position", "last"))
    g = g[g.games >= min_games]
    g["luck_pg"] = g.fp_pg - g.xfp_pg
    me = lg.roster_of(username)
    mine = set(me.get("players") or [])
    owner = {p: lg.owner(r) for r in lg.rosters for p in r.get("players") or []}

    def fmt(df_):
        return [{"player": lg.name(pid), "pos": r.pos, "owner": owner.get(pid, "free agent"),
                 "games": int(r.games), "ppr_pts_per_game": round(r.fp_pg, 1),
                 "expected_pts_per_game": round(r.xfp_pg, 1), "over_under_expected": round(r.luck_pg, 1)}
                for pid, r in df_.iterrows()]

    relevant = g[g.xfp_pg >= 8]
    buy = relevant[~relevant.index.isin(mine) & relevant.index.isin(owner)].sort_values("luck_pg").head(limit)
    sell = g[g.index.isin(mine) & (g.fp_pg >= 8) & (g.luck_pg >= 1.5)].sort_values("luck_pg", ascending=False).head(limit)
    return {
        "buy_low_targets": fmt(buy),
        "sell_high_candidates_on_your_roster": fmt(sell),
        "explanation": ("Expected points (xFP) measure the volume and quality of a player's opportunities. "
                        "Big negative gaps tend to bounce back (unlucky TDs/efficiency); big positive gaps "
                        "tend to regress."),
    }


# ---------- waivers ----------

def waiver_targets(league_id: str, username: str = "", position: str = "", limit: int = 10,
                   include_playoffs: bool = True) -> dict:
    """Free agents ranked by how many rest-of-season starting-lineup points they add to YOUR team."""
    lg = League(league_id)
    weeks = _ros_weeks(lg, include_playoffs)
    proj = lg.proj(weeks)
    rostered = {p for r in lg.rosters for p in r.get("players") or []}
    ros = proj.groupby(level=0).pred_fp_mean.sum()
    pos = proj.groupby(level=0).position.first()
    startable = set().union(*(P.SLOT_ELIGIBLE[s] for s in lg.slots if s in P.SLOT_ELIGIBLE))
    wanted = {position.upper()} & startable if position else startable
    fa = ros[~ros.index.isin(rostered) & pos.reindex(ros.index).isin(wanted)]
    candidates = (fa.to_frame("ros").assign(pos=pos).sort_values("ros", ascending=False)
                  .groupby("pos").head(15).index)

    me = lg.roster_of(username)
    roster = me.get("players") or []
    base = _lineup_points(lg, proj, roster, weeks)
    wk_now = P.week_table(proj, lg.week)
    bench_now = [p for p in roster if p not in {x for _, x in P.optimal_lineup(roster, wk_now, lg.slots)}]
    # Cheapest drop: the bench player whose removal costs the least ROS lineup points.
    drop_cost = {p: base - _lineup_points(lg, proj, [x for x in roster if x != p], weeks) for p in bench_now}
    drop = min(drop_cost, key=drop_cost.get) if drop_cost else None

    out = []
    for pid in candidates:
        gain = _lineup_points(lg, proj, roster + [pid], weeks) - base
        trend = (P.week_table(proj, lg.week).loc[pid].pred_fp_mean if pid in wk_now.index else 0.0)
        out.append({"player": lg.name(pid), "player_id": pid, "pos": pos.get(pid),
                    "team": (lg.players.get(pid) or {}).get("team"),
                    "ros_points": round(float(fa[pid]), 1), "this_week_proj": round(float(trend), 1),
                    "lineup_points_added": round(gain, 1)})
    out.sort(key=lambda d: -d["lineup_points_added"])
    s = lg.lg.get("settings", {})
    return {
        "weeks": f"{weeks[0]}-{weeks[-1]}",
        "targets": out[:limit],
        "suggested_drop": None if drop is None else {"player": lg.name(drop), "lineup_points_lost": round(drop_cost[drop], 1)},
        "faab_remaining": None if s.get("waiver_type") != 2 else
            (s.get("waiver_budget", 100) - (me.get("settings") or {}).get("waiver_budget_used", 0)),
    }
