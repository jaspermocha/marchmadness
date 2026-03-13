"""
scraper.py — Automated data fetching for the March Madness betting model.

Sources
-------
1. Barttorvik (T-Rank) — adjusted efficiency ratings, four factors, tempo
   Primary:  https://barttorvik.com/trank.php  (HTML table)
   Fallback: https://barttorvik.com/{year}_team_results.json (per-game JSON)

2. NCAA bracket seedings — from data.ncaa.com scoreboard and ESPN API
   Primary:  https://data.ncaa.com/casablanca/scoreboard/basketball-men/d1/{date}/scoreboard.json
   Fallback: http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/tournaments/22

Usage
-----
    from scraper import fetch_tournament_teams
    teams = fetch_tournament_teams(year=2026)
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from typing import Optional
from functools import lru_cache

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from data import Team

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json,*/*",
    }
)


def _get(url: str, timeout: int = 15, retries: int = 3, **kwargs) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}")


# ---------------------------------------------------------------------------
# Barttorvik scraper
# ---------------------------------------------------------------------------

# Maps Barttorvik HTML column headers → our internal names
_BART_COL_MAP = {
    # The HTML table headers vary; we try multiple known names
    "AdjOE": "adj_o",
    "Adj OE": "adj_o",
    "OAdj Eff": "adj_o",
    "AdjDE": "adj_d",
    "Adj DE": "adj_d",
    "DAdj Eff": "adj_d",
    "Barthag": "barthag",
    "Adj T.": "adj_tempo",
    "Adj. T.": "adj_tempo",
    "AdjT": "adj_tempo",
    "Tempo": "adj_tempo",
    "WAB": "wab",
    "EFG%": "efg_pct",
    "eFG%": "efg_pct",
    "Eff. FG%": "efg_pct",
    "EFGD%": "efg_d_pct",
    "eFGD%": "efg_d_pct",
    "Opp. eFG%": "efg_d_pct",
    "TOR": "to_pct",
    "TO%": "to_pct",
    "Turnover %": "to_pct",
    "TORD": "to_d_pct",
    "Opp. TO%": "to_d_pct",
    "ORB": "orb_pct",
    "Off. Reb%": "orb_pct",
    "DRB": "drb_pct",
    "Def. Reb%": "drb_pct",
    "FTR": "ft_rate",
    "FT Rate": "ft_rate",
    "FTRD": "ft_rate_d",
    "Opp. FT Rate": "ft_rate_d",
}


def fetch_barttorvik_ratings(year: int = 2026) -> pd.DataFrame:
    """
    Scrape T-Rank efficiency ratings from barttorvik.com.

    Returns a DataFrame with columns:
      team, conf, adj_o, adj_d, adj_tempo, barthag, wab,
      efg_pct, efg_d_pct, to_pct, to_d_pct, orb_pct, drb_pct,
      ft_rate, ft_rate_d, wins, losses, sos
    """
    url = f"https://barttorvik.com/trank.php?year={year}&conyes=1&csv=1"
    try:
        df = _fetch_barttorvik_csv(url, year)
        if df is not None and len(df) > 50:
            return df
    except Exception:
        pass

    # Fallback: scrape the HTML table
    url_html = f"https://barttorvik.com/trank.php?year={year}"
    return _fetch_barttorvik_html(url_html, year)


def _fetch_barttorvik_csv(url: str, year: int) -> Optional[pd.DataFrame]:
    """Try to get Barttorvik data as CSV."""
    resp = _get(url)
    content = resp.text.strip()
    if not content or "<html" in content[:200].lower():
        return None

    from io import StringIO
    df = pd.read_csv(StringIO(content))
    df.columns = df.columns.str.strip()
    return _normalize_barttorvik_df(df)


def _fetch_barttorvik_html(url: str, year: int) -> pd.DataFrame:
    """Parse the main trank.php HTML table."""
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "lxml")

    # Find the main data table
    table = soup.find("table", id="dataTable") or soup.find("table")
    if table is None:
        raise RuntimeError("Could not find data table on barttorvik.com")

    # Extract headers
    headers = []
    header_row = table.find("thead")
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
    else:
        first_row = table.find("tr")
        headers = [td.get_text(strip=True) for td in first_row.find_all(["th", "td"])]

    # Extract rows
    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells and len(cells) >= 5:
            rows.append(cells)

    if not rows:
        raise RuntimeError("No data rows found in Barttorvik table")

    # Build DataFrame — column count may not match header count perfectly
    max_cols = max(len(r) for r in rows)
    while len(headers) < max_cols:
        headers.append(f"col_{len(headers)}")

    df = pd.DataFrame(rows, columns=headers[: max_cols])
    return _normalize_barttorvik_df(df)


def _normalize_barttorvik_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a raw Barttorvik DataFrame into our model's column names.
    Handles varying column naming conventions across years.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Rename using the map
    rename = {}
    for col in df.columns:
        if col in _BART_COL_MAP:
            rename[col] = _BART_COL_MAP[col]
        elif col.upper() in {k.upper(): v for k, v in _BART_COL_MAP.items()}:
            rename[col] = {k.upper(): v for k, v in _BART_COL_MAP.items()}[col.upper()]
    df = df.rename(columns=rename)

    # Identify team name column
    for candidate in ["Team", "team", "TEAM", "School"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "team"})
            break

    # Identify conference column
    for candidate in ["Conf", "conf", "Conference", "CONF"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "conf"})
            break

    if "conf" not in df.columns:
        df["conf"] = "Unknown"

    # Parse wins/losses from record column (e.g. "28-5")
    if "wins" not in df.columns or "losses" not in df.columns:
        for rec_col in ["Rec", "Record", "W-L", "rec"]:
            if rec_col in df.columns:
                df[["wins", "losses"]] = (
                    df[rec_col]
                    .str.extract(r"(\d+)-(\d+)")
                    .astype(float)
                )
                break

    if "wins" not in df.columns:
        for w_col in ["W", "Wins", "wins"]:
            if w_col in df.columns:
                df["wins"] = pd.to_numeric(df[w_col], errors="coerce")
                break

    if "losses" not in df.columns:
        for l_col in ["L", "Losses", "losses"]:
            if l_col in df.columns:
                df["losses"] = pd.to_numeric(df[l_col], errors="coerce")
                break

    # Compute strength of schedule proxy from existing columns if missing
    if "sos" not in df.columns:
        for s_col in ["SOS", "Sos", "Strength"]:
            if s_col in df.columns:
                df["sos"] = pd.to_numeric(df[s_col], errors="coerce")
                break

    if "sos" not in df.columns:
        df["sos"] = 0.0

    # Convert all numeric columns
    numeric_cols = [
        "adj_o", "adj_d", "adj_tempo", "barthag", "wab",
        "efg_pct", "efg_d_pct", "to_pct", "to_d_pct",
        "orb_pct", "drb_pct", "ft_rate", "ft_rate_d",
        "wins", "losses", "sos",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert percentage columns if they look like whole-number percentages (e.g. 52.3 → 0.523)
    pct_cols = ["efg_pct", "efg_d_pct", "to_pct", "to_d_pct", "orb_pct", "drb_pct", "ft_rate", "ft_rate_d"]
    for col in pct_cols:
        if col in df.columns:
            median_val = df[col].median()
            if median_val is not None and median_val > 1.0:
                df[col] = df[col] / 100.0

    # Fill missing four-factor columns with plausible defaults
    defaults = {
        "efg_pct": 0.51, "efg_d_pct": 0.51,
        "to_pct": 0.18, "to_d_pct": 0.18,
        "orb_pct": 0.30, "drb_pct": 0.72,
        "ft_rate": 0.34, "ft_rate_d": 0.34,
        "wab": 0.0, "sos": 0.0,
        "wins": 20, "losses": 12,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # Drop rows with no team name or no efficiency data
    required = ["team", "adj_o", "adj_d"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column after normalization: {col}")

    df = df.dropna(subset=["team", "adj_o", "adj_d"])
    df = df[df["team"].str.strip() != ""]

    # Fill barthag from adj_em if missing
    if "barthag" not in df.columns or df["barthag"].isna().all():
        df["barthag"] = (df["adj_o"] - df["adj_d"]).apply(
            lambda x: float(1 / (1 + np.exp(-x / 20)))
        )
    df["barthag"] = df["barthag"].fillna(0.5)

    if "adj_tempo" not in df.columns:
        df["adj_tempo"] = 70.0
    df["adj_tempo"] = df["adj_tempo"].fillna(70.0)

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# NCAA bracket scraper
# ---------------------------------------------------------------------------

# March Madness first-round dates to check (by year)
_FIRST_ROUND_DATES = {
    2026: ["2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22"],
    2025: ["2025-03-20", "2025-03-21", "2025-03-22", "2025-03-23"],
    2024: ["2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24"],
}

_REGIONS_CANONICAL = {"East", "West", "South", "Midwest"}

# Maps ESPN region codes / names to canonical names
_REGION_NORM = {
    "east": "East", "west": "West", "south": "South", "midwest": "Midwest",
    "e": "East", "w": "West", "s": "South", "mw": "Midwest",
    "midwst": "Midwest",
}


def fetch_ncaa_bracket(year: int = 2026) -> dict[str, dict[int, str]]:
    """
    Fetch the NCAA tournament bracket seedings from data.ncaa.com.

    Returns: {region: {seed: team_name}} dict, e.g.:
        {"East": {1: "Duke", 16: "Wagner", ...}, ...}
    """
    bracket: dict[str, dict[int, str]] = {}

    # Try NCAA scoreboard for each first-round date
    dates = _FIRST_ROUND_DATES.get(year, [])
    if not dates:
        # Generate dates around mid-March
        base = date(year, 3, 19)
        dates = [(base + timedelta(days=i)).isoformat() for i in range(6)]

    for day in dates:
        url = f"https://data.ncaa.com/casablanca/scoreboard/basketball-men/d1/{day}/scoreboard.json"
        try:
            resp = _get(url, timeout=10)
            data = resp.json()
            bracket = _parse_ncaa_scoreboard(data, bracket)
        except Exception:
            continue

    if _bracket_complete(bracket):
        return bracket

    # Fallback: ESPN tournament endpoint
    try:
        bracket = _fetch_espn_bracket(year, bracket)
    except Exception:
        pass

    return bracket


def _parse_ncaa_scoreboard(data: dict, bracket: dict) -> dict:
    """Parse games from NCAA scoreboard JSON and extract seedings."""
    games = data.get("games", [])
    for game_wrapper in games:
        game = game_wrapper.get("game", game_wrapper)

        bracket_region = (
            game.get("bracketRegion")
            or game.get("gameState", {}).get("bracketRegion", "")
        )
        bracket_round = (
            game.get("bracketRound")
            or game.get("gameState", {}).get("bracketRound", "")
        )

        # Only care about main bracket rounds (not NIT etc.)
        if not bracket_region or not bracket_round:
            continue

        region = _norm_region(bracket_region)
        if not region:
            continue

        if region not in bracket:
            bracket[region] = {}

        # Each game has two teams; extract seed + name
        for team_key in ["home", "away"]:
            team_data = game.get(team_key, {})
            if not team_data:
                continue

            seed = team_data.get("seed") or team_data.get("bracketSeed")
            name = (
                team_data.get("names", {}).get("short")
                or team_data.get("names", {}).get("seo")
                or team_data.get("shortName")
                or team_data.get("name", {}).get("short")
            )
            if seed and name:
                try:
                    bracket[region][int(seed)] = str(name)
                except (ValueError, TypeError):
                    pass

    return bracket


def _fetch_espn_bracket(year: int, bracket: dict) -> dict:
    """Try ESPN's tournament endpoints for bracket data."""
    urls = [
        f"http://site.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/tournaments/22/seasons/{year}",
        f"http://site.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/tournaments/22",
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?seasontype=3&dates={year}",
    ]
    for url in urls:
        try:
            resp = _get(url, timeout=10)
            data = resp.json()
            bracket = _parse_espn_bracket(data, bracket)
            if _bracket_complete(bracket):
                return bracket
        except Exception:
            continue
    return bracket


def _parse_espn_bracket(data: dict, bracket: dict) -> dict:
    """Recursively search ESPN response for seeding info."""
    # ESPN bracket responses vary greatly; try to find competitor entries
    def _walk(obj):
        if isinstance(obj, dict):
            seed = obj.get("seed")
            region = obj.get("region", "")
            team = (
                obj.get("team", {}).get("shortDisplayName")
                or obj.get("team", {}).get("displayName")
                or obj.get("team", {}).get("abbreviation")
            )
            if seed and team and region:
                r = _norm_region(str(region))
                if r:
                    bracket.setdefault(r, {})[int(seed)] = str(team)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)
    return bracket


def _norm_region(raw: str) -> Optional[str]:
    """Normalize a region string to canonical form."""
    clean = raw.strip().lower()
    if clean in _REGION_NORM:
        return _REGION_NORM[clean]
    for canonical in _REGIONS_CANONICAL:
        if canonical.lower() in clean:
            return canonical
    return None


def _bracket_complete(bracket: dict) -> bool:
    """Check if we have all 4 regions with all 16 seeds."""
    if len(bracket) < 4:
        return False
    for region in _REGIONS_CANONICAL:
        if region not in bracket:
            return False
        if len(bracket[region]) < 16:
            return False
    return True


# ---------------------------------------------------------------------------
# Team name normalizer
# ---------------------------------------------------------------------------

def normalize_team_name(name: str) -> str:
    """
    Normalize a college basketball team name by removing common suffixes
    and standardizing format for fuzzy matching.
    """
    name = name.strip()
    # Remove state abbreviations like "St." → "State" for matching
    suffixes = [
        " Blue Devils", " Jayhawks", " Wildcats", " Bulldogs", " Tigers",
        " Tar Heels", " Hoyas", " Ramblers", " Cardinal", " Crimson Tide",
        " Longhorns", " Volunteers", " Boilermakers", " Fighting Illini",
        " Cyclones", " Bears", " Horned Frogs", " Mustangs",
    ]
    result = name
    for s in suffixes:
        if result.endswith(s):
            result = result[: -len(s)]
            break
    result = re.sub(r"\s+", " ", result).strip()
    return result


def fuzzy_match_teams(
    bart_name: str, candidates: list[str], threshold: float = 0.6
) -> Optional[str]:
    """
    Find the best fuzzy match for a Barttorvik team name among candidates.
    Returns the best match or None if no match exceeds the threshold.
    """
    from difflib import SequenceMatcher

    bart_norm = normalize_team_name(bart_name).lower()
    best_score = 0.0
    best_match = None

    for candidate in candidates:
        cand_norm = normalize_team_name(candidate).lower()

        # Exact match
        if bart_norm == cand_norm:
            return candidate

        # Partial containment
        if bart_norm in cand_norm or cand_norm in bart_norm:
            score = 0.9
        else:
            score = SequenceMatcher(None, bart_norm, cand_norm).ratio()

        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match if best_score >= threshold else None


# ---------------------------------------------------------------------------
# Main: combine Barttorvik stats + NCAA bracket
# ---------------------------------------------------------------------------

def fetch_tournament_teams(year: int = 2026, verbose: bool = True) -> list[Team]:
    """
    Fully automated: fetch efficiency stats and bracket seedings,
    merge them, and return a list of Team objects ready for the model.

    Steps:
      1. Fetch T-Rank ratings from Barttorvik
      2. Fetch bracket seedings from data.ncaa.com / ESPN
      3. Fuzzy-match team names between the two sources
      4. Build Team objects with complete data
    """
    if verbose:
        print(f"[scraper] Fetching Barttorvik T-Rank ratings for {year}...")

    stats_df = fetch_barttorvik_ratings(year)
    if verbose:
        print(f"[scraper] Got stats for {len(stats_df)} teams.")

    if verbose:
        print(f"[scraper] Fetching NCAA tournament bracket for {year}...")

    bracket = fetch_ncaa_bracket(year)
    complete = _bracket_complete(bracket)

    if verbose:
        total_seeded = sum(len(v) for v in bracket.values())
        print(
            f"[scraper] Bracket: {len(bracket)} regions, {total_seeded} seeds found "
            f"({'complete' if complete else 'INCOMPLETE – bracket may not be set yet'})."
        )

    if not bracket:
        raise RuntimeError(
            "Could not fetch bracket data. The tournament bracket may not be "
            "announced yet, or the data sources are unavailable. "
            "Run `python main.py --setup` to use sample data instead."
        )

    # Build lookup: bart_name → stats row
    bart_names = list(stats_df["team"].values)

    teams: list[Team] = []
    unmatched: list[str] = []

    for region, seed_map in bracket.items():
        for seed, bracket_name in seed_map.items():
            # Find matching row in Barttorvik stats
            matched_name = None

            # Try exact match first
            if bracket_name in bart_names:
                matched_name = bracket_name
            else:
                matched_name = fuzzy_match_teams(bracket_name, bart_names)

            if matched_name is None:
                if verbose:
                    print(
                        f"[scraper] WARNING: No Barttorvik match for "
                        f"{region} #{seed} '{bracket_name}'"
                    )
                unmatched.append(bracket_name)
                continue

            row = stats_df[stats_df["team"] == matched_name].iloc[0]

            teams.append(
                Team(
                    team=bracket_name,
                    seed=seed,
                    region=region,
                    adj_o=float(row["adj_o"]),
                    adj_d=float(row["adj_d"]),
                    adj_tempo=float(row.get("adj_tempo", 70.0)),
                    barthag=float(row.get("barthag", 0.5)),
                    wab=float(row.get("wab", 0.0)),
                    efg_pct=float(row.get("efg_pct", 0.51)),
                    to_pct=float(row.get("to_pct", 0.18)),
                    orb_pct=float(row.get("orb_pct", 0.30)),
                    ft_rate=float(row.get("ft_rate", 0.34)),
                    efg_d_pct=float(row.get("efg_d_pct", 0.51)),
                    to_d_pct=float(row.get("to_d_pct", 0.18)),
                    drb_pct=float(row.get("drb_pct", 0.72)),
                    ft_rate_d=float(row.get("ft_rate_d", 0.34)),
                    wins=int(row.get("wins", 20)),
                    losses=int(row.get("losses", 12)),
                    sos=float(row.get("sos", 0.0)),
                    conf=str(row.get("conf", "Unknown")),
                )
            )

    if verbose and unmatched:
        print(
            f"[scraper] {len(unmatched)} teams unmatched. "
            "Check team name mapping or update manually."
        )

    return teams


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    teams = fetch_tournament_teams(year=year)
    print(f"\nFetched {len(teams)} tournament teams:")
    for t in sorted(teams, key=lambda x: (x.region, x.seed)):
        print(f"  {t.region:8s} #{t.seed:2d}  {t.team:30s}  AdjEM={t.adj_em:+.1f}")
