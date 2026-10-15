"""Sleeper fantasy football MCP server.

Data sources:
  - Sleeper public API (read-only): leagues, rosters, matchups, transactions, players, trending
  - Sleeper projections endpoint (unofficial): weekly projections
  - FantasyCalc API: trade values / overall + positional rankings

Set SLEEPER_USERNAME in the environment so tools can default to your account.
"""

import threading

from mcp.server.mcpserver import MCPServer

from analytics import league as L
from analytics import model as M

from sleeper_client import (DEFAULT_USERNAME, SLEEPER, _get, _league_format, _players, _state,
                            _summ, _user_id, _values, _cached)

mcp = MCPServer("sleeper")


# ---------- tools ----------

@mcp.tool()
def get_nfl_state() -> dict:
    """Current NFL season, week, and season type."""
    return _state()


@mcp.tool()
def get_my_leagues(username: str = "", season: str = "") -> list[dict]:
    """List the user's Sleeper NFL leagues for a season (defaults to current season and SLEEPER_USERNAME)."""
    season = season or _state()["league_season"]
    leagues = _get(f"{SLEEPER}/user/{_user_id(username)}/leagues/nfl/{season}")
    return [{"league_id": l["league_id"], "name": l["name"], "status": l["status"],
             "teams": l["total_rosters"], "type": {0: "redraft", 1: "keeper", 2: "dynasty"}.get(
                 l.get("settings", {}).get("type"), "unknown")} for l in leagues]


@mcp.tool()
def get_league(league_id: str) -> dict:
    """League settings: roster slots, scoring, waiver/trade rules, and league members."""
    lg = _get(f"{SLEEPER}/league/{league_id}")
    users = _get(f"{SLEEPER}/league/{league_id}/users")
    s = lg.get("settings", {})
    return {
        "name": lg["name"], "season": lg["season"], "status": lg["status"],
        "roster_positions": lg.get("roster_positions"),
        "scoring_settings": lg.get("scoring_settings"),
        "waiver_type": {0: "rolling", 1: "reverse standings", 2: "FAAB"}.get(s.get("waiver_type")),
        "waiver_budget": s.get("waiver_budget"),
        "trade_deadline_week": s.get("trade_deadline"),
        "playoff_start_week": s.get("playoff_week_start"),
        "members": [{"user_id": u["user_id"], "display_name": u["display_name"],
                     "team_name": (u.get("metadata") or {}).get("team_name")} for u in users],
    }


@mcp.tool()
def get_rosters(league_id: str, only_username: str = "") -> list[dict]:
    """All rosters in a league with player names, starters, record, and points.
    Pass only_username to return just that user's roster."""
    players = _players()
    rosters = _get(f"{SLEEPER}/league/{league_id}/rosters")
    users = {u["user_id"]: u["display_name"] for u in _get(f"{SLEEPER}/league/{league_id}/users")}
    out = []
    for r in rosters:
        owner = users.get(r["owner_id"], r["owner_id"])
        if only_username and owner.lower() != only_username.lower():
            continue
        s = r.get("settings", {})
        starters = set(r.get("starters") or [])
        out.append({
            "roster_id": r["roster_id"], "owner": owner,
            "record": f"{s.get('wins', 0)}-{s.get('losses', 0)}-{s.get('ties', 0)}",
            "points_for": s.get("fpts", 0) + s.get("fpts_decimal", 0) / 100,
            "waiver_budget_used": s.get("waiver_budget_used"),
            "starters": [_summ(p, players) for p in r.get("starters") or []],
            "bench": [_summ(p, players) for p in r.get("players") or [] if p not in starters],
            "taxi": [_summ(p, players) for p in r.get("taxi") or []],
            "ir": [_summ(p, players) for p in r.get("reserve") or []],
        })
    return out


@mcp.tool()
def get_matchups(league_id: str, week: int = 0) -> list[dict]:
    """Head-to-head matchups and scores for a week (defaults to current week)."""
    week = week or _state()["week"]
    players = _players()
    rosters = {r["roster_id"]: r["owner_id"] for r in _get(f"{SLEEPER}/league/{league_id}/rosters")}
    users = {u["user_id"]: u["display_name"] for u in _get(f"{SLEEPER}/league/{league_id}/users")}
    games: dict[int, list] = {}
    for m in _get(f"{SLEEPER}/league/{league_id}/matchups/{week}"):
        games.setdefault(m.get("matchup_id") or 0, []).append({
            "owner": users.get(rosters.get(m["roster_id"]), m["roster_id"]),
            "points": m.get("points"),
            "starters": [{**_summ(p, players), "pts": (m.get("players_points") or {}).get(p)}
                         for p in m.get("starters") or []],
        })
    return [{"matchup_id": k, "teams": v} for k, v in games.items()]


@mcp.tool()
def get_transactions(league_id: str, week: int = 0, kind: str = "") -> list[dict]:
    """League transactions for a week (trades, waivers, free agent moves).
    kind filters to 'trade', 'waiver', or 'free_agent'."""
    week = week or _state()["week"]
    players = _players()
    rosters = {r["roster_id"]: r["owner_id"] for r in _get(f"{SLEEPER}/league/{league_id}/rosters")}
    users = {u["user_id"]: u["display_name"] for u in _get(f"{SLEEPER}/league/{league_id}/users")}
    who = lambda rid: users.get(rosters.get(rid), rid)
    out = []
    for t in _get(f"{SLEEPER}/league/{league_id}/transactions/{week}"):
        if kind and t["type"] != kind:
            continue
        out.append({
            "type": t["type"], "status": t["status"],
            "adds": {_summ(p, players)["name"]: who(r) for p, r in (t.get("adds") or {}).items()},
            "drops": {_summ(p, players)["name"]: who(r) for p, r in (t.get("drops") or {}).items()},
            "picks": [{"season": d["season"], "round": d["round"], "to": who(d["owner_id"]),
                       "from": who(d["previous_owner_id"])} for d in t.get("draft_picks") or []],
            "faab": [{"amount": w["amount"], "to": who(w["receiver"])}
                     for w in t.get("waiver_budget") or []],
        })
    return out


@mcp.tool()
def search_players(name: str = "", position: str = "", team: str = "", limit: int = 25) -> list[dict]:
    """Search the full NFL player database by name, position (QB/RB/WR/TE/K/DEF), and/or team abbr."""
    players = _players()
    name, position, team = name.lower(), position.upper(), team.upper()
    out = []
    for pid, p in players.items():
        if name and name not in (p.get("full_name") or p.get("search_full_name") or "").lower():
            continue
        if position and p.get("position") != position:
            continue
        if team and p.get("team") != team:
            continue
        if not p.get("active") and not name:
            continue
        out.append(_summ(pid, players))
        if len(out) >= limit:
            break
    return out


@mcp.tool()
def get_player(player_id: str) -> dict:
    """Full detail for a single player (bio, depth chart, injury notes)."""
    p = _players().get(player_id)
    if not p:
        raise ValueError(f"Unknown player id {player_id}")
    keys = ["full_name", "position", "fantasy_positions", "team", "age", "years_exp", "college",
            "height", "weight", "status", "injury_status", "injury_body_part", "injury_notes",
            "depth_chart_position", "depth_chart_order", "number", "news_updated"]
    return {k: p.get(k) for k in keys}


@mcp.tool()
def get_trending(kind: str = "add", lookback_hours: int = 24, limit: int = 25) -> list[dict]:
    """Most added ('add') or dropped ('drop') players across all Sleeper leagues."""
    players = _players()
    data = _get(f"{SLEEPER}/players/nfl/trending/{kind}",
                {"lookback_hours": lookback_hours, "limit": limit})
    return [{**_summ(d["player_id"], players), "count": d["count"]} for d in data]


@mcp.tool()
def get_rankings(position: str = "", dynasty: bool = False, superflex: bool = False,
                 ppr: float = 1, num_teams: int = 12, limit: int = 50) -> list[dict]:
    """Player rankings by trade value (FantasyCalc, crowd-sourced from real trades).
    Use get_league_format first to get the right dynasty/superflex/ppr settings for a league."""
    players = _players()
    vals = _values(dynasty, superflex, ppr, num_teams)
    rows = [{**_summ(pid, players), **v} for pid, v in vals.items()]
    if position:
        rows = [r for r in rows if r.get("pos") == position.upper()]
    return sorted(rows, key=lambda r: -r["value"])[:limit]


@mcp.tool()
def get_league_format(league_id: str) -> dict:
    """The dynasty/superflex/ppr/team-count settings for a league, for use with ranking tools."""
    return _league_format(league_id)


@mcp.tool()
def evaluate_trade(league_id: str, give_player_ids: list[str], get_player_ids: list[str]) -> dict:
    """Compare the trade value of players given vs received, using the league's format."""
    f = _league_format(league_id)
    vals = _values(**f)
    players = _players()

    def side(ids):
        rows = [{**_summ(p, players), "value": vals.get(p, {}).get("value", 0)} for p in ids]
        return rows, sum(r["value"] for r in rows)

    give, gt = side(give_player_ids)
    get, rt = side(get_player_ids)
    return {"format": f, "give": give, "give_total": gt, "get": get, "get_total": rt,
            "net_value_for_you": rt - gt,
            "note": "Raw sums favor 2-for-1 sides; the single best player usually carries a premium."}


@mcp.tool()
def get_free_agents(league_id: str, position: str = "", limit: int = 25) -> list[dict]:
    """Best available unrostered players in a league, ranked by trade value for the league's format."""
    players = _players()
    rostered = {p for r in _get(f"{SLEEPER}/league/{league_id}/rosters") for p in r.get("players") or []}
    vals = _values(**_league_format(league_id))
    rows = [{**_summ(pid, players), **v} for pid, v in vals.items() if pid not in rostered]
    if position:
        rows = [r for r in rows if r.get("pos") == position.upper()]
    return sorted(rows, key=lambda r: -r["value"])[:limit]


@mcp.tool()
def get_projections(week: int = 0, position: str = "", player_ids: list[str] | None = None,
                    limit: int = 40) -> list[dict]:
    """Sleeper's weekly fantasy point projections (PPR, half-PPR, standard) for the current season.
    Filter by position or a specific list of player ids."""
    st = _state()
    week = week or st["week"]
    params = {"season_type": "regular"}
    if position:
        params["position[]"] = position.upper()
    data = _cached(f"proj_{st['season']}_{week}_{position}", 3 * 3600, lambda: _get(
        f"https://api.sleeper.app/projections/nfl/{st['season']}/{week}", params))
    players = _players()
    out = []
    for row in data:
        pid = row.get("player_id")
        if player_ids and pid not in player_ids:
            continue
        s = row.get("stats") or {}
        if not s.get("pts_ppr"):
            continue
        out.append({**_summ(pid, players), "opponent": row.get("opponent"),
                    "ppr": s.get("pts_ppr"), "half_ppr": s.get("pts_half_ppr"), "std": s.get("pts_std")})
    return sorted(out, key=lambda r: -(r["ppr"] or 0))[: len(player_ids) if player_ids else limit]


# ---------- prediction model tools ----------

@mcp.tool()
def project_lineup(league_id: str, week: int = 0, username: str = "") -> dict:
    """Model projections (mean, floor, ceiling in the league's scoring) for every player on the
    user's roster, the optimal lineup, and which start/sit changes to make."""
    return L.lineup(league_id, username, week)


@mcp.tool()
def simulate_matchup(league_id: str, week: int = 0, username: str = "") -> dict:
    """Monte Carlo simulation of the user's matchup: win probability as set and with the optimal
    lineup, score ranges, and bench swaps ranked by how much they raise win probability.
    Players whose games are already final use their actual points."""
    return L.matchup(league_id, username, week)


@mcp.tool()
def playoff_odds(league_id: str) -> dict:
    """Simulates the rest of the regular season for every team: projected wins, playoff odds,
    and top-seed odds."""
    return L.playoff_odds(league_id)


@mcp.tool()
def ros_projections(league_id: str, player_ids: list[str], include_playoffs: bool = True) -> list[dict]:
    """Rest-of-season projected points (league scoring) for players, including fantasy playoff weeks."""
    return L.ros_projections(league_id, player_ids, include_playoffs)


@mcp.tool()
def trade_impact(league_id: str, give_player_ids: list[str], get_player_ids: list[str],
                 username: str = "", include_playoffs: bool = True) -> dict:
    """Rest-of-season trade analysis: change in each team's projected STARTING lineup points.
    Better than raw value for redraft; pair with evaluate_trade (market value) for dynasty."""
    return L.trade_impact(league_id, give_player_ids, get_player_ids, username, include_playoffs)


@mcp.tool()
def buy_low_sell_high(league_id: str, username: str = "") -> dict:
    """Players outscoring or underscoring their expected fantasy points (opportunity-based xFP):
    buy-low targets on other rosters and sell-high candidates on the user's roster."""
    return L.buy_sell(league_id, username)


@mcp.tool()
def waiver_targets(league_id: str, position: str = "", username: str = "") -> dict:
    """Free agents ranked by rest-of-season starting-lineup points they'd add to the user's team,
    plus the cheapest bench player to drop and FAAB remaining."""
    return L.waiver_targets(league_id, username, position)


@mcp.tool()
def model_info() -> dict:
    """How the projection model was trained and how it backtests against Sleeper's projections."""
    _, meta = M.load()
    return meta


def _warm():
    """Build this week's features in the background so the first tool call is fast."""
    try:
        from analytics import predict
        st = _state()
        predict.model_weeks(int(st["season"]), int(st["week"]))
    except Exception:
        pass


if __name__ == "__main__":
    threading.Thread(target=_warm, daemon=True).start()
    mcp.run()
