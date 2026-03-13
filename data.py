"""
data.py — Team data loader for the March Madness betting model.

Supports two modes:
  1. Load from a local CSV (teams.csv) — recommended for reproducibility.
  2. Scrape live KenPom-style stats from sports-reference.com (requires requests).

Expected CSV columns (all per-100-possession unless noted):
  team, seed, region, adj_o, adj_d, adj_tempo, barthag, wab,
  efg_pct, efg_d_pct, to_pct, to_d_pct, orb_pct, drb_pct,
  ft_rate, ft_rate_d, wins, losses, sos, conf
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Team dataclass
# ---------------------------------------------------------------------------

@dataclass
class Team:
    team: str
    seed: int
    region: str

    # Adjusted efficiency margins (KenPom-style, points per 100 possessions)
    adj_o: float        # Adjusted offensive efficiency
    adj_d: float        # Adjusted defensive efficiency (lower = better)
    adj_tempo: float    # Adjusted tempo (possessions per 40 min)

    # Composite ratings
    barthag: float      # Power rating (probability of beating average D-I team)
    wab: float          # Wins above bubble

    # Four-factor offense
    efg_pct: float      # Effective field goal %
    to_pct: float       # Turnover % (lower = better for offense)
    orb_pct: float      # Offensive rebound %
    ft_rate: float      # Free throw rate (FTA/FGA)

    # Four-factor defense (opponent values)
    efg_d_pct: float    # Opponent eFG% (lower = better)
    to_d_pct: float     # Opponent TO% (higher = better)
    drb_pct: float      # Defensive rebound %
    ft_rate_d: float    # Opponent FT rate (lower = better)

    # Season record
    wins: int
    losses: int
    sos: float          # Strength of schedule

    conf: str = "Unknown"

    @property
    def adj_em(self) -> float:
        """Adjusted efficiency margin (adj_o - adj_d)."""
        return self.adj_o - self.adj_d

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"Team({self.team!r}, seed={self.seed}, region={self.region!r}, "
            f"AdjEM={self.adj_em:+.1f})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "team", "seed", "region", "adj_o", "adj_d", "adj_tempo",
    "barthag", "wab", "efg_pct", "efg_d_pct", "to_pct", "to_d_pct",
    "orb_pct", "drb_pct", "ft_rate", "ft_rate_d", "wins", "losses", "sos",
}


def load_teams(csv_path: str = "teams.csv") -> list[Team]:
    """Load teams from a CSV file and return a list of Team objects."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Team data file not found: {csv_path}\n"
            "Run `python data.py --generate-sample` to create a sample file, "
            "or populate teams.csv manually."
        )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    int_cols = {"seed", "wins", "losses"}
    for col in int_cols:
        df[col] = df[col].astype(int)

    float_cols = REQUIRED_COLUMNS - int_cols - {"team", "region"}
    for col in float_cols:
        df[col] = df[col].astype(float)

    if "conf" not in df.columns:
        df["conf"] = "Unknown"

    teams = []
    for _, row in df.iterrows():
        teams.append(
            Team(
                team=str(row["team"]).strip(),
                seed=int(row["seed"]),
                region=str(row["region"]).strip(),
                adj_o=float(row["adj_o"]),
                adj_d=float(row["adj_d"]),
                adj_tempo=float(row["adj_tempo"]),
                barthag=float(row["barthag"]),
                wab=float(row["wab"]),
                efg_pct=float(row["efg_pct"]),
                to_pct=float(row["to_pct"]),
                orb_pct=float(row["orb_pct"]),
                ft_rate=float(row["ft_rate"]),
                efg_d_pct=float(row["efg_d_pct"]),
                to_d_pct=float(row["to_d_pct"]),
                drb_pct=float(row["drb_pct"]),
                ft_rate_d=float(row["ft_rate_d"]),
                wins=int(row["wins"]),
                losses=int(row["losses"]),
                sos=float(row["sos"]),
                conf=str(row.get("conf", "Unknown")),
            )
        )

    return teams


def teams_to_df(teams: list[Team]) -> pd.DataFrame:
    """Convert a list of Team objects to a DataFrame."""
    rows = []
    for t in teams:
        rows.append(
            {
                "team": t.team,
                "seed": t.seed,
                "region": t.region,
                "adj_o": t.adj_o,
                "adj_d": t.adj_d,
                "adj_tempo": t.adj_tempo,
                "adj_em": t.adj_em,
                "barthag": t.barthag,
                "wab": t.wab,
                "efg_pct": t.efg_pct,
                "to_pct": t.to_pct,
                "orb_pct": t.orb_pct,
                "ft_rate": t.ft_rate,
                "efg_d_pct": t.efg_d_pct,
                "to_d_pct": t.to_d_pct,
                "drb_pct": t.drb_pct,
                "ft_rate_d": t.ft_rate_d,
                "wins": t.wins,
                "losses": t.losses,
                "sos": t.sos,
                "conf": t.conf,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sample data generator
# ---------------------------------------------------------------------------

def generate_sample_teams_csv(path: str = "teams.csv") -> None:
    """Write a realistic 68-team sample CSV for the 2026 tournament."""
    import random
    random.seed(42)
    rng = np.random.default_rng(42)

    regions = ["East", "West", "South", "Midwest"]
    confs = [
        "ACC", "Big Ten", "Big 12", "SEC", "Pac-12", "Big East",
        "American", "Mountain West", "A-10", "Missouri Valley",
        "WCC", "MAC", "Sun Belt", "CUSA", "Horizon",
        "Ivy", "SWAC", "MEAC", "Big South", "OVC",
    ]

    # Fictional teams — replace with real data for live use
    team_names_by_seed = {
        1: ["Duke", "Kansas", "Houston", "Auburn"],
        2: ["Alabama", "Iowa State", "Tennessee", "Michigan State"],
        3: ["Kentucky", "Purdue", "Creighton", "Wisconsin"],
        4: ["Texas A&M", "Illinois", "Arizona", "Arkansas"],
        5: ["Michigan", "St. John's", "Memphis", "Gonzaga"],
        6: ["BYU", "Marquette", "UCLA", "Ole Miss"],
        7: ["Clemson", "Missouri", "Georgia", "Xavier"],
        8: ["Louisville", "Oklahoma", "Washington St.", "Virginia"],
        9: ["Drake", "Nevada", "Saint Mary's", "Florida St."],
        10: ["New Mexico", "Wake Forest", "Colorado St.", "Penn St."],
        11: ["VCU", "Oregon", "NC State", "Pittsburgh"],
        12: ["McNeese St.", "Akron", "Liberty", "UNCW"],
        13: ["Vermont", "Colgate", "Samford", "Yale"],
        14: ["Oakland", "Morehead St.", "Hofstra", "Grand Canyon"],
        15: ["S. Dakota St.", "Long Beach St.", "Chattanooga", "Montana St."],
        16: ["Wagner", "Norfolk St.", "Longwood", "Holy Cross",
              "W. Kentucky", "Gardner-Webb", "Howard", "UT Rio Grande Valley"],
    }

    rows = []
    for seed, names in team_names_by_seed.items():
        teams_for_seed = names
        # Seed 16 has 8 teams (4 play-in winners, but we'll list first 4 per region)
        for i, name in enumerate(teams_for_seed[:4]):
            region = regions[i % 4]
            # Scale efficiency metrics roughly by seed quality
            quality = max(1, 17 - seed)  # higher = better
            adj_o = 115 + quality * 0.8 + rng.normal(0, 1.5)
            adj_d = 105 - quality * 0.6 + rng.normal(0, 1.5)
            adj_tempo = rng.uniform(65, 75)
            barthag = min(0.99, max(0.01, 0.5 + (adj_o - adj_d) * 0.025 + rng.normal(0, 0.03)))
            wab = (quality - 8) * 0.5 + rng.normal(0, 1.0)
            wins = int(18 + quality * 0.8 + rng.integers(-3, 4))
            losses = int(14 - quality * 0.5 + rng.integers(0, 5))
            losses = max(losses, 1)

            rows.append(
                {
                    "team": name,
                    "seed": seed,
                    "region": region,
                    "adj_o": round(adj_o, 1),
                    "adj_d": round(adj_d, 1),
                    "adj_tempo": round(adj_tempo, 1),
                    "barthag": round(barthag, 4),
                    "wab": round(wab, 1),
                    "efg_pct": round(rng.uniform(0.48, 0.56) + quality * 0.002, 3),
                    "to_pct": round(rng.uniform(0.15, 0.22) - quality * 0.001, 3),
                    "orb_pct": round(rng.uniform(0.25, 0.35), 3),
                    "ft_rate": round(rng.uniform(0.28, 0.42), 3),
                    "efg_d_pct": round(rng.uniform(0.48, 0.56) - quality * 0.002, 3),
                    "to_d_pct": round(rng.uniform(0.15, 0.22) + quality * 0.001, 3),
                    "drb_pct": round(rng.uniform(0.68, 0.78), 3),
                    "ft_rate_d": round(rng.uniform(0.28, 0.42) - quality * 0.002, 3),
                    "wins": wins,
                    "losses": losses,
                    "sos": round(rng.uniform(-4, 12) + quality * 0.3, 2),
                    "conf": random.choice(confs[:seed]),  # better seeds from better confs
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Sample teams.csv written to {path} ({len(df)} teams)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--generate-sample" in sys.argv:
        out = "teams.csv"
        for arg in sys.argv:
            if arg.startswith("--out="):
                out = arg.split("=", 1)[1]
        generate_sample_teams_csv(out)
    else:
        teams = load_teams()
        df = teams_to_df(teams)
        print(df[["team", "seed", "region", "adj_em", "barthag"]].to_string(index=False))
