"""
odds_fetcher.py — Live betting lines from The Odds API.

The Odds API (https://the-odds-api.com) aggregates odds from major US sportsbooks.
Free tier: 500 requests/month — enough for the entire tournament.

Setup
-----
1. Sign up for a free key at https://the-odds-api.com
2. Set ODDS_API_KEY in your .env file (see .env.example)

Response caching
----------------
Responses are cached in .odds_cache.json for 5 minutes to avoid hitting
the rate limit when running the model repeatedly.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from betting import GameLine
from data import Team
from scraper import fuzzy_match_teams, normalize_team_name

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_ncaab"
CACHE_FILE = ".odds_cache.json"
CACHE_TTL_SECONDS = 300  # 5 minutes

# Preferred bookmakers in priority order (used for consensus line)
PREFERRED_BOOKS = [
    "draftkings", "fanduel", "betmgm", "caesars", "pointsbet",
    "williamhill_us", "unibet_us", "bovada", "betonlineag",
]


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Load ODDS_API_KEY from environment or .env file."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        # Try loading from .env manually (avoid requiring python-dotenv)
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ODDS_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not key:
        raise EnvironmentError(
            "ODDS_API_KEY not set.\n"
            "  1. Sign up for a free key at https://the-odds-api.com\n"
            "  2. Add ODDS_API_KEY=your_key_here to .env"
        )
    return key


def fetch_raw_odds(
    markets: str = "h2h,spreads,totals",
    regions: str = "us",
    odds_format: str = "american",
    use_cache: bool = True,
) -> list[dict]:
    """
    Fetch raw NCAA basketball odds from The Odds API.
    Returns the full list of game objects from the API response.
    Uses a local file cache to avoid redundant requests.
    """
    # Check cache
    if use_cache and _cache_valid():
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        print(f"[odds] Using cached odds ({_cache_age_str()}).")
        return cached

    api_key = _get_api_key()
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }

    resp = requests.get(url, params=params, timeout=15)

    # Show remaining quota
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"[odds] API quota: {used} used, {remaining} remaining this month.")

    resp.raise_for_status()
    data = resp.json()

    # Save to cache
    with open(CACHE_FILE, "w") as f:
        json.dump({"data": data, "fetched_at": time.time()}, f)

    return data


def _cache_valid() -> bool:
    if not os.path.exists(CACHE_FILE):
        return False
    try:
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        age = time.time() - cached.get("fetched_at", 0)
        return age < CACHE_TTL_SECONDS
    except Exception:
        return False


def _cache_age_str() -> str:
    try:
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        age = int(time.time() - cached.get("fetched_at", 0))
        return f"{age}s old"
    except Exception:
        return "age unknown"


# ---------------------------------------------------------------------------
# Parse API response → GameLine objects
# ---------------------------------------------------------------------------

def parse_odds_to_game_lines(
    raw_games: list[dict] | dict,
    teams_by_name: dict[str, Team],
    verbose: bool = True,
) -> list[GameLine]:
    """
    Convert raw API response to GameLine objects.

    Handles fuzzy matching of API team names (e.g. "Duke Blue Devils") to
    our internal names (e.g. "Duke") using normalize_team_name + fuzzy match.

    Parameters
    ----------
    raw_games    : API response (list of game dicts, or the cache wrapper dict)
    teams_by_name: dict mapping internal team name → Team object
    """
    # Unwrap cache format if needed
    if isinstance(raw_games, dict) and "data" in raw_games:
        raw_games = raw_games["data"]

    internal_names = list(teams_by_name.keys())
    lines: list[GameLine] = []
    unmatched_pairs: list[tuple] = []

    for game in raw_games:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        commence = game.get("commence_time", "")

        # Fuzzy match to our team names
        home_internal = _match_api_name(home_raw, internal_names)
        away_internal = _match_api_name(away_raw, internal_names)

        if home_internal is None or away_internal is None:
            if verbose:
                unmatched_pairs.append((home_raw, away_raw))
            continue

        team_a = teams_by_name[home_internal]
        team_b = teams_by_name[away_internal]

        # Parse round from commence time
        round_name = _infer_round_name(commence)

        # Aggregate consensus line across bookmakers
        ml_a, ml_b = _consensus_moneyline(game, home_raw, away_raw)
        spread_a, spread_juice_a, spread_juice_b = _consensus_spread(game, home_raw)
        total, over_juice, under_juice = _consensus_total(game)

        lines.append(
            GameLine(
                team_a=team_a,
                team_b=team_b,
                round_name=round_name,
                ml_a=ml_a,
                ml_b=ml_b,
                spread_a=spread_a,
                spread_juice_a=spread_juice_a if spread_juice_a else -110,
                spread_juice_b=spread_juice_b if spread_juice_b else -110,
                total=total,
                total_over_juice=over_juice if over_juice else -110,
                total_under_juice=under_juice if under_juice else -110,
            )
        )

    if verbose and unmatched_pairs:
        print(f"[odds] {len(unmatched_pairs)} games had unmatched team names (not in bracket):")
        for a, b in unmatched_pairs[:5]:
            print(f"       '{a}' vs '{b}'")

    return lines


def _match_api_name(api_name: str, internal_names: list[str]) -> Optional[str]:
    """Match an API team name to an internal team name."""
    norm = normalize_team_name(api_name)
    # Try exact
    if api_name in internal_names:
        return api_name
    if norm in internal_names:
        return norm
    # Fuzzy
    return fuzzy_match_teams(norm, internal_names, threshold=0.55)


def _consensus_moneyline(
    game: dict, home_name: str, away_name: str
) -> tuple[Optional[int], Optional[int]]:
    """
    Extract consensus moneyline odds by averaging across preferred bookmakers.
    Returns (ml_home, ml_away) in American odds format.
    """
    home_prices = []
    away_prices = []

    for bm in game.get("bookmakers", []):
        if bm.get("key") not in PREFERRED_BOOKS:
            continue
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == home_name:
                    home_prices.append(outcome.get("price"))
                elif outcome.get("name") == away_name:
                    away_prices.append(outcome.get("price"))

    # If no preferred books found, try all bookmakers
    if not home_prices or not away_prices:
        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    if name == home_name and not home_prices:
                        home_prices.append(outcome.get("price"))
                    elif name == away_name and not away_prices:
                        away_prices.append(outcome.get("price"))

    ml_home = _median_odds(home_prices)
    ml_away = _median_odds(away_prices)
    return ml_home, ml_away


def _consensus_spread(
    game: dict, home_name: str
) -> tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Extract consensus spread for the home team.
    Returns (spread, juice_home, juice_away).
    """
    spreads = []
    juices_a = []
    juices_b = []

    for bm in game.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "spreads":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == home_name:
                    pt = outcome.get("point")
                    price = outcome.get("price")
                    if pt is not None:
                        spreads.append(float(pt))
                    if price is not None:
                        juices_a.append(int(price))
                else:
                    price = outcome.get("price")
                    if price is not None:
                        juices_b.append(int(price))

    if not spreads:
        return None, None, None

    import statistics
    spread = round(statistics.median(spreads) * 2) / 2
    juice_a = _median_odds(juices_a)
    juice_b = _median_odds(juices_b)
    return spread, juice_a, juice_b


def _consensus_total(
    game: dict,
) -> tuple[Optional[float], Optional[int], Optional[int]]:
    """Extract consensus over/under total."""
    totals = []
    over_juices = []
    under_juices = []

    for bm in game.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []):
                pt = outcome.get("point")
                price = outcome.get("price")
                if pt is not None:
                    totals.append(float(pt))
                if outcome.get("name") == "Over" and price:
                    over_juices.append(int(price))
                elif outcome.get("name") == "Under" and price:
                    under_juices.append(int(price))

    if not totals:
        return None, None, None

    import statistics
    total = round(statistics.median(totals) * 2) / 2
    return total, _median_odds(over_juices), _median_odds(under_juices)


def _median_odds(prices: list) -> Optional[int]:
    """Compute the median of a list of American odds prices."""
    valid = [p for p in prices if p is not None]
    if not valid:
        return None
    import statistics
    return int(statistics.median(valid))


# ---------------------------------------------------------------------------
# Tournament round inference from game time
# ---------------------------------------------------------------------------

# Approximate dates for each round (2026 tournament)
_ROUND_DATES_2026 = [
    ("First Four", ("2026-03-17", "2026-03-18")),
    ("Round of 64", ("2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22")),
    ("Round of 32", ("2026-03-23", "2026-03-24", "2026-03-25")),
    ("Sweet 16", ("2026-03-27", "2026-03-28")),
    ("Elite Eight", ("2026-03-29", "2026-03-30")),
    ("Final Four", ("2026-04-04",)),
    ("Championship", ("2026-04-06",)),
]


def _infer_round_name(commence_time: str) -> str:
    """Infer the tournament round from a game's start time."""
    if not commence_time:
        return "Tournament"
    try:
        # Parse ISO format: "2026-03-21T19:10:00Z"
        date_part = commence_time[:10]
        for round_name, dates in _ROUND_DATES_2026:
            if date_part in dates:
                return round_name
    except Exception:
        pass
    return "Tournament"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_game_lines(
    teams_by_name: dict[str, Team],
    use_cache: bool = True,
    verbose: bool = True,
) -> list[GameLine]:
    """
    Fully automated: fetch live odds and convert to GameLine objects.

    Parameters
    ----------
    teams_by_name : dict of internal_name → Team (from scraper)
    use_cache     : use local cache to avoid redundant API calls
    """
    if verbose:
        print("[odds] Fetching live betting lines from The Odds API...")

    raw = fetch_raw_odds(use_cache=use_cache)
    lines = parse_odds_to_game_lines(raw, teams_by_name, verbose=verbose)

    if verbose:
        print(f"[odds] Parsed {len(lines)} tournament games with odds.")

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = fetch_raw_odds(use_cache=False)
    print(f"Got {len(raw)} games from The Odds API.")
    for g in raw[:3]:
        print(f"  {g['home_team']} vs {g['away_team']}  ({g['commence_time']})")
