"""Sleeper public API helpers with a small disk cache."""

import json
import os
import time
from pathlib import Path

import httpx

SLEEPER = "https://api.sleeper.app/v1"
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
DEFAULT_USERNAME = os.environ.get("SLEEPER_USERNAME", "")

http = httpx.Client(timeout=30)


# ---------- helpers ----------

def _get(url: str, params: dict | None = None):
    r = http.get(url, params=params)
    r.raise_for_status()
    return r.json()


def _cached(name: str, max_age: int, fetch):
    """Disk cache. Sleeper asks that /players/nfl be fetched at most once a day."""
    path = CACHE_DIR / f"{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < max_age:
        return json.loads(path.read_text())
    data = fetch()
    path.write_text(json.dumps(data))
    return data


def _players() -> dict:
    return _cached("players", 86400, lambda: _get(f"{SLEEPER}/players/nfl"))


def _state() -> dict:
    return _cached("state", 3600, lambda: _get(f"{SLEEPER}/state/nfl"))


def _values(dynasty: bool, superflex: bool, ppr: float, num_teams: int) -> dict[str, dict]:
    """FantasyCalc trade values keyed by Sleeper player id."""
    key = f"values_{int(dynasty)}_{int(superflex)}_{ppr}_{num_teams}"
    raw = _cached(key, 6 * 3600, lambda: _get(
        "https://api.fantasycalc.com/values/current",
        {"isDynasty": str(dynasty).lower(), "numQbs": 2 if superflex else 1,
         "numTeams": num_teams, "ppr": ppr},
    ))
    out = {}
    for v in raw:
        sid = v["player"].get("sleeperId")
        if sid:
            out[sid] = {"value": v["value"], "overall_rank": v["overallRank"],
                        "position_rank": v["positionRank"], "trend_30d": v.get("trend30Day")}
    return out


def _summ(pid: str, players: dict | None = None) -> dict:
    p = (players or _players()).get(pid)
    if not p:
        return {"player_id": pid, "name": pid}  # team defenses use the team abbr as id
    return {
        "player_id": pid,
        "name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
        "pos": p.get("position"),
        "team": p.get("team"),
        "age": p.get("age"),
        "injury": p.get("injury_status"),
        "status": p.get("status"),
    }


def _user_id(username: str) -> str:
    username = username or DEFAULT_USERNAME
    if not username:
        raise ValueError("No username given and SLEEPER_USERNAME is not set")
    u = _get(f"{SLEEPER}/user/{username}")
    if not u:
        raise ValueError(f"Sleeper user '{username}' not found")
    return u["user_id"]


def _league_format(league_id: str) -> dict:
    lg = _get(f"{SLEEPER}/league/{league_id}")
    sc = lg.get("scoring_settings", {})
    return {
        "dynasty": lg.get("settings", {}).get("type") == 2,
        "superflex": "SUPER_FLEX" in lg.get("roster_positions", []),
        "ppr": sc.get("rec", 0),
        "num_teams": lg.get("total_rosters", 12),
    }
