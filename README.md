# Sleeper MCP

MCP server that gives Claude live data on your Sleeper leagues plus a fantasy projection model and simulations.

## Data sources
- **Sleeper API:** leagues, rosters, matchups, transactions, players, trending, projections
- **FantasyCalc:** trade-market values and rankings
- **nflverse** (via `nflreadpy`): play-by-play-derived stats since 2016, expected fantasy points (ffopportunity), snap counts, injury reports, schedules with Vegas spreads and totals

## Tools
Data: get_nfl_state, get_my_leagues, get_league, get_league_format, get_rosters, get_matchups, get_transactions,
search_players, get_player, get_trending, get_rankings, get_free_agents, evaluate_trade, get_projections

Model: project_lineup, simulate_matchup, playoff_odds, ros_projections, trade_impact, buy_low_sell_high,
waiver_targets, model_info

## Model
`analytics/` builds player-week features (recent usage, xFP, target/air-yard share, snap %, Vegas implied total,
opponent points allowed by position, injury reports, Sleeper's projection), then trains LightGBM heads for the mean
plus 10th/50th/90th percentiles, receptions, and passing TDs (the last two rescore to any league's scoring).
Simulations sample each player from a two-piece normal fit to those percentiles.

Retrain weekly (Tuesday after Monday Night Football):

    uv run python train.py

The script prints the backtest against Sleeper's projections. Features refresh on their own every 6 hours; retraining
only updates the model weights.

## Backtest

Each test season is held out of training. Compared with Sleeper's own weekly projections on the same player-weeks:

| Held-out season | Player-weeks | Start/sit calls correct (model vs. Sleeper) | RMSE (model vs. Sleeper) | MAE, median prediction (model vs. Sleeper) | 80% band coverage |
|---|---|---|---|---|---|
| 2024 | 5,226 | 65.0% vs. 65.1% | 6.15 vs. 6.18 | 4.37 vs. 4.45 | 80.6% |
| 2025 | 5,408 | 65.4% vs. 65.4% | 5.99 vs. 6.11 | 4.22 vs. 4.42 | 79.7% |

In short: about as good as Sleeper for start/sit decisions, slightly better on large misses, and its floor/ceiling ranges are
calibrated, which is what the simulations rely on. Mean-prediction MAE is about equal to Sleeper's.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

    uv sync
    uv run python train.py      # first run downloads nflverse data and trains the model

## Register
Claude Code:

    claude mcp add sleeper -s user -e SLEEPER_USERNAME=<your_sleeper_username> -- uv run --directory /path/to/sleeper-mcp python server.py

Claude desktop app: `SLEEPER_USERNAME=<your_sleeper_username> ./add-to-claude-app.sh` (it waits for the app to quit, since the app overwrites its config while running).

## License

MIT. Data comes from the Sleeper API, FantasyCalc, and nflverse under their own terms; none of it is redistributed here.
