"""
conf_tournament.py — Conference tournament schedule fetching and analysis.

Conference tournaments are played on neutral courts (just like March Madness),
so the model's efficiency-based win probabilities apply directly with no
home-court adjustment needed.

Schedule source: ESPN scoreboard API (public, no key required)
Odds source:     The Odds API (same as --auto mode)

ESPN conference group IDs
--------------------------
Major conferences relevant for betting:
  ACC=2, Big East=10, Big Ten=4, Big 12=8, SEC=23,
  Pac-12=21, Mountain West=29, American=62, A-10=3, WCC=35
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
from tabulate import tabulate

from data import Team
from model import MarchMadnessModel
from betting import GameLine, analyze_lines, print_value_bets, estimate_game_total
from scraper import fuzzy_match_teams, normalize_team_name

# ---------------------------------------------------------------------------
# ESPN conference group IDs (used to filter schedule by conference)
# ---------------------------------------------------------------------------

CONF_GROUP_IDS: dict[str, int] = {
    "ACC": 2,
    "A-10": 3,
    "Big Ten": 4,
    "Big 12": 8,
    "Big East": 10,
    "Pac-12": 21,
    "SEC": 23,
    "Mountain West": 29,
    "WCC": 35,
    "American": 62,
    "MVC": 18,       # Missouri Valley
    "MAC": 12,
    "Sun Belt": 37,
    "CUSA": 11,
    "Horizon": 45,
    "Ivy": 22,
    "CAA": 97,
    "OVC": 27,
    "SWAC": 31,
    "Big South": 42,
    "MEAC": 24,
    "NEC": 25,
    "Patriot": 28,
    "SoCon": 29,
    "WAC": 16,
}

# Conference tournament typical week ranges (month-day)
# Used to label games as "conference tournament" vs regular season
CONF_TOURNEY_WINDOWS = {
    # (start_month_day, end_month_day) — approximate
    "default": ((3, 4), (3, 16)),  # first two weeks of March
}


# ---------------------------------------------------------------------------
# ESPN schedule fetcher
# ---------------------------------------------------------------------------

ESPN_SCOREBOARD_URL = (
    "http://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)


def _espn_get(url: str, params: dict = None, timeout: int = 12) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_conf_tournament_schedule(
    conf_name: Optional[str] = None,
    days_ahead: int = 7,
    include_today: bool = True,
) -> list[dict]:
    """
    Fetch upcoming conference tournament games from the ESPN scoreboard API.

    Parameters
    ----------
    conf_name   : Filter to a specific conference (e.g. "ACC", "SEC").
                  None = return all conferences.
    days_ahead  : How many days forward to look for games.
    include_today: Include today's games.

    Returns
    -------
    List of game dicts, each with:
      { team_a, team_b, conf, date, game_id, notes, status }
    """
    start = date.today() if include_today else date.today() + timedelta(days=1)
    games: list[dict] = []
    seen_ids: set[str] = set()

    group_id = CONF_GROUP_IDS.get(conf_name) if conf_name else None

    for i in range(days_ahead + 1):
        day = start + timedelta(days=i)
        date_str = day.strftime("%Y%m%d")

        params: dict = {"dates": date_str, "limit": 200}
        if group_id:
            params["groups"] = group_id

        try:
            data = _espn_get(ESPN_SCOREBOARD_URL, params=params)
        except Exception as e:
            print(f"[conf] ESPN fetch failed for {date_str}: {e}")
            continue

        for event in data.get("events", []):
            game_id = event.get("id", "")
            if game_id in seen_ids:
                continue
            seen_ids.add(game_id)

            # Extract notes / tournament label
            notes = " | ".join(
                n.get("headline", "") for n in event.get("notes", [])
            )
            season_type = event.get("season", {}).get("type", 0)
            season_slug = event.get("season", {}).get("slug", "")

            # Only include games that look like conference tournament games:
            # - Notes mention "tournament" / "championship"
            # - OR it's in the typical conference tournament date window
            # - OR caller forced a specific conf (so include all their games)
            is_conf_tourney = (
                _looks_like_conf_tourney(notes, season_type, season_slug, day)
                or conf_name is not None
            )
            if not is_conf_tourney:
                continue

            competition = (event.get("competitions") or [{}])[0]
            competitors = competition.get("competitors", [])
            if len(competitors) < 2:
                continue

            # Home team is index 0 in ESPN
            home = competitors[0]
            away = competitors[1]
            home_name = (
                home.get("team", {}).get("shortDisplayName")
                or home.get("team", {}).get("displayName", "Unknown")
            )
            away_name = (
                away.get("team", {}).get("shortDisplayName")
                or away.get("team", {}).get("displayName", "Unknown")
            )

            # Conference
            conf = (
                home.get("team", {}).get("conferenceId", "")
                or competition.get("conferenceCompetition", False)
            )

            status = event.get("status", {}).get("type", {}).get("description", "")

            games.append(
                {
                    "team_a": home_name,
                    "team_b": away_name,
                    "conf": str(conf),
                    "date": day.isoformat(),
                    "game_id": game_id,
                    "notes": notes,
                    "status": status,
                    "name": event.get("name", f"{home_name} vs {away_name}"),
                }
            )

    return games


def _looks_like_conf_tourney(
    notes: str, season_type: int, season_slug: str, game_date: date
) -> bool:
    """Heuristic: is this game a conference tournament game?"""
    notes_lower = notes.lower()
    slug_lower = season_slug.lower()

    # ESPN marks conference tournaments in notes
    conf_keywords = [
        "tournament", "championship", "conf. tournament",
        "conference tournament", "conf champ",
    ]
    for kw in conf_keywords:
        if kw in notes_lower or kw in slug_lower:
            return True

    # Conference tournaments run early March
    window = CONF_TOURNEY_WINDOWS["default"]
    start_md = window[0]
    end_md = window[1]
    start_date = date(game_date.year, start_md[0], start_md[1])
    end_date = date(game_date.year, end_md[0], end_md[1])
    if start_date <= game_date <= end_date:
        return True

    return False


# ---------------------------------------------------------------------------
# Build Team objects for conf tournament teams
# ---------------------------------------------------------------------------

def build_conf_teams(
    game_list: list[dict],
    stats_df: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, Team]:
    """
    Given a list of games and a Barttorvik stats DataFrame (all D-I teams),
    build a name→Team dict for every team in the schedule.

    Since conference tournament teams don't have seeds or regions, we use
    seed=0 and region="Conf" as placeholders — the model only uses efficiency
    metrics so these don't affect predictions.
    """
    bart_names = list(stats_df["team"].values)
    teams_by_name: dict[str, Team] = {}
    unmatched: list[str] = []

    all_team_names: set[str] = set()
    for g in game_list:
        all_team_names.add(g["team_a"])
        all_team_names.add(g["team_b"])

    for game_name in sorted(all_team_names):
        if game_name in teams_by_name:
            continue

        # Match to Barttorvik name
        matched = game_name if game_name in bart_names else fuzzy_match_teams(
            game_name, bart_names, threshold=0.55
        )

        if matched is None:
            if verbose:
                print(f"[conf] WARNING: No stats found for '{game_name}'")
            unmatched.append(game_name)
            continue

        row = stats_df[stats_df["team"] == matched].iloc[0]

        # Fill defaults for any missing columns
        def _f(col, default):
            v = row.get(col, default)
            try:
                return float(v) if v is not None else float(default)
            except (ValueError, TypeError):
                return float(default)

        teams_by_name[game_name] = Team(
            team=game_name,
            seed=0,          # not seeded in conf tournament
            region="Conf",
            adj_o=_f("adj_o", 110.0),
            adj_d=_f("adj_d", 110.0),
            adj_tempo=_f("adj_tempo", 70.0),
            barthag=_f("barthag", 0.5),
            wab=_f("wab", 0.0),
            efg_pct=_f("efg_pct", 0.51),
            to_pct=_f("to_pct", 0.18),
            orb_pct=_f("orb_pct", 0.30),
            ft_rate=_f("ft_rate", 0.34),
            efg_d_pct=_f("efg_d_pct", 0.51),
            to_d_pct=_f("to_d_pct", 0.18),
            drb_pct=_f("drb_pct", 0.72),
            ft_rate_d=_f("ft_rate_d", 0.34),
            wins=int(_f("wins", 20)),
            losses=int(_f("losses", 12)),
            sos=_f("sos", 0.0),
            conf=str(row.get("conf", "Unknown")),
        )

    if verbose and unmatched:
        print(
            f"[conf] {len(unmatched)} teams could not be matched to Barttorvik stats.\n"
            "       They will be excluded from analysis."
        )

    return teams_by_name


# ---------------------------------------------------------------------------
# Prediction table
# ---------------------------------------------------------------------------

def predict_conf_games(
    game_list: list[dict],
    teams_by_name: dict[str, Team],
    model: MarchMadnessModel,
) -> pd.DataFrame:
    """
    Run model predictions for every game in the schedule.

    Returns a DataFrame sorted by the higher win probability (most decisive
    matchups last, closest matchups first).
    """
    rows = []
    for g in game_list:
        name_a = g["team_a"]
        name_b = g["team_b"]
        if name_a not in teams_by_name or name_b not in teams_by_name:
            continue

        ta = teams_by_name[name_a]
        tb = teams_by_name[name_b]
        prob_a = model.predict(ta, tb)
        proj_total = estimate_game_total(ta, tb)

        rows.append(
            {
                "date": g["date"],
                "matchup": f"{name_a} vs {name_b}",
                "fav": name_a if prob_a >= 0.5 else name_b,
                "fav_prob": f"{max(prob_a, 1-prob_a)*100:.1f}%",
                "dog": name_b if prob_a >= 0.5 else name_a,
                "dog_prob": f"{min(prob_a, 1-prob_a)*100:.1f}%",
                "adj_em_diff": f"{ta.adj_em - tb.adj_em:+.1f}",
                "proj_total": proj_total,
                "notes": g.get("notes", "")[:60],
                "_sort_key": abs(prob_a - 0.5),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live odds for conf tournament games
# ---------------------------------------------------------------------------

def fetch_conf_odds(
    game_list: list[dict],
    teams_by_name: dict[str, Team],
    verbose: bool = True,
) -> list[GameLine]:
    """
    Fetch live odds for conference tournament games from The Odds API.
    The Odds API returns all NCAAB games, so conf tournament games are included.
    """
    try:
        from odds_fetcher import fetch_raw_odds, parse_odds_to_game_lines
        raw = fetch_raw_odds(use_cache=True)
        lines = parse_odds_to_game_lines(raw, teams_by_name, verbose=verbose)
        return lines
    except EnvironmentError as e:
        if verbose:
            print(f"[odds] {e}")
        return []
    except Exception as e:
        if verbose:
            print(f"[odds] Failed to fetch odds: {e}")
        return []


# ---------------------------------------------------------------------------
# Full conference tournament analysis pipeline
# ---------------------------------------------------------------------------

def run_conf_analysis(
    model: MarchMadnessModel,
    conf_name: Optional[str] = None,
    days_ahead: int = 7,
    min_edge: float = 0.02,
    fetch_odds: bool = True,
    verbose: bool = True,
) -> None:
    """
    Full pipeline:
      1. Fetch all D-I team stats from Barttorvik
      2. Fetch upcoming conf tournament schedule from ESPN
      3. Build Team objects for every team in the schedule
      4. Predict every matchup
      5. Fetch live odds and find value bets
    """
    from scraper import fetch_barttorvik_ratings

    label = conf_name or "All Conferences"
    print(f"\n{'='*70}")
    print(f"  CONFERENCE TOURNAMENT ANALYSIS — {label.upper()}")
    print(f"{'='*70}")

    # 1. Stats
    year = date.today().year
    if verbose:
        print(f"\n[conf] Fetching Barttorvik stats for {year}...")
    stats_df = fetch_barttorvik_ratings(year)
    if verbose:
        print(f"[conf] Stats loaded for {len(stats_df)} teams.")

    # 2. Schedule
    if verbose:
        print(f"[conf] Fetching ESPN schedule ({days_ahead}-day window)...")
    games = fetch_conf_tournament_schedule(
        conf_name=conf_name, days_ahead=days_ahead
    )
    if not games:
        print(
            f"\n  No upcoming conference tournament games found"
            f"{f' for {conf_name}' if conf_name else ''}.\n"
            "  Either no games are scheduled in the next "
            f"{days_ahead} days, or the ESPN API did not return results.\n"
            "  Try --days 14 to look further ahead."
        )
        return

    if verbose:
        print(f"[conf] Found {len(games)} upcoming games.\n")

    # 3. Build team objects
    teams_by_name = build_conf_teams(games, stats_df, verbose=verbose)

    # 4. Predictions
    pred_df = predict_conf_games(games, teams_by_name, model)
    if not pred_df.empty:
        print(f"\n--- MATCHUP PREDICTIONS ---")
        display = pred_df[["date", "fav", "fav_prob", "dog", "dog_prob", "adj_em_diff", "proj_total"]]
        display.columns = ["Date", "Favorite", "Prob", "Underdog", "Prob", "AdjEM Δ", "Proj Total"]
        print(tabulate(display, headers="keys", tablefmt="rounded_outline", showindex=False))

    # 5. Odds + value bets
    if fetch_odds:
        if verbose:
            print("\n[conf] Fetching live odds...")
        lines = fetch_conf_odds(games, teams_by_name, verbose=verbose)
        if lines:
            value_df = analyze_lines(lines, model, min_edge=min_edge)
            print_value_bets(value_df)
        else:
            print("\n  No odds data available for these games.")
    else:
        # Build synthetic lines from model predictions for display
        print("\n  (Odds fetching disabled — run without --no-odds to see value bets)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from model import get_or_train_model

    conf = sys.argv[1] if len(sys.argv) > 1 else None
    model = get_or_train_model()
    run_conf_analysis(model, conf_name=conf)
