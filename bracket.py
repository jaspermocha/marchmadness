"""
bracket.py — March Madness bracket structure and tournament simulator.

The 68-team tournament is modeled as:
  • 4 First Four (play-in) games  → 64 remaining
  • Round of 64  (32 games)
  • Round of 32  (16 games)
  • Sweet 16      (8 games)
  • Elite Eight   (4 games)
  • Final Four    (2 games)
  • Championship  (1 game)

Each region has seeds 1-16. After the Elite Eight, the Final Four matchups
are determined by bracket assignment (e.g. East vs West, South vs Midwest).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from data import Team
from model import MarchMadnessModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIONS = ["East", "West", "South", "Midwest"]
ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite Eight",
    5: "Final Four",
    6: "Championship",
}

# Standard bracket matchups by seed within a region (higher seed = worse)
# Seed pairings: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
SEED_MATCHUPS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

# Final Four bracket: East winner vs West winner, South vs Midwest
SEMIFINAL_PAIRS = [("East", "West"), ("South", "Midwest")]


# ---------------------------------------------------------------------------
# Bracket
# ---------------------------------------------------------------------------

@dataclass
class Matchup:
    team_a: Team
    team_b: Team
    round_num: int
    region: str
    prob_a: float = 0.0
    winner: Optional[Team] = None

    def __repr__(self) -> str:
        a = f"({self.team_a.seed}) {self.team_a.team}"
        b = f"({self.team_b.seed}) {self.team_b.team}"
        return f"{ROUND_NAMES.get(self.round_num, 'Round ?')}: {a} vs {b}"


class Bracket:
    """Represents a full 64-team tournament bracket."""

    def __init__(self, teams: list[Team]) -> None:
        self.teams = teams
        # Index by region and seed for quick lookups
        self._by_region: dict[str, dict[int, Team]] = {}
        for t in teams:
            self._by_region.setdefault(t.region, {})[t.seed] = t

        self._validate()

    def _validate(self) -> None:
        for region in REGIONS:
            if region not in self._by_region:
                raise ValueError(f"Missing region: {region}")
            seeds = set(self._by_region[region].keys())
            expected = set(range(1, 17))
            if not expected.issubset(seeds):
                missing = expected - seeds
                raise ValueError(f"Region {region!r} missing seeds: {missing}")

    def get_team(self, region: str, seed: int) -> Team:
        return self._by_region[region][seed]

    def round_of_64_matchups(self) -> list[Matchup]:
        matchups = []
        for region in REGIONS:
            for s1, s2 in SEED_MATCHUPS:
                t1 = self.get_team(region, s1)
                t2 = self.get_team(region, s2)
                matchups.append(Matchup(t1, t2, round_num=1, region=region))
        return matchups


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class TournamentSimulator:
    """
    Simulates the full March Madness tournament using Monte Carlo methods.

    Usage
    -----
    sim = TournamentSimulator(bracket, model)
    results = sim.simulate(n_sims=10_000)
    print(results.to_string())
    """

    def __init__(self, bracket: Bracket, model: MarchMadnessModel) -> None:
        self.bracket = bracket
        self.model = model

    # ------------------------------------------------------------------
    # Single simulation run
    # ------------------------------------------------------------------

    def _run_once(self, rng: np.random.Generator) -> dict[str, int]:
        """
        Simulate one full tournament. Returns a dict mapping team name to
        the round they were eliminated (0 = did not make it to Round 1).
        Teams that win the championship are marked with round 7.
        """
        # Current survivors per region
        survivors: dict[str, list[Team]] = {r: [] for r in REGIONS}
        for region in REGIONS:
            for s1, s2 in SEED_MATCHUPS:
                t1 = self.bracket.get_team(region, s1)
                t2 = self.bracket.get_team(region, s2)
                winner = self._play_game(t1, t2, rng)
                survivors[region].append(winner)

        results: dict[str, int] = {}
        round_num = 1

        # Regional rounds (2–4)
        for round_num in range(2, 5):
            next_survivors: dict[str, list[Team]] = {r: [] for r in REGIONS}
            for region in REGIONS:
                teams_in_round = survivors[region]
                for i in range(0, len(teams_in_round), 2):
                    t1, t2 = teams_in_round[i], teams_in_round[i + 1]
                    winner = self._play_game(t1, t2, rng)
                    loser = t2 if winner is t1 else t1
                    results[loser.team] = round_num - 1
                    next_survivors[region].append(winner)
            survivors = next_survivors

        # Final Four (round 5)
        ff_winners = []
        for reg_a, reg_b in SEMIFINAL_PAIRS:
            t1 = survivors[reg_a][0]
            t2 = survivors[reg_b][0]
            winner = self._play_game(t1, t2, rng)
            loser = t2 if winner is t1 else t1
            results[loser.team] = 4  # Elite Eight winner, lost in Final Four
            ff_winners.append(winner)

        # Championship (round 6)
        champ = self._play_game(ff_winners[0], ff_winners[1], rng)
        loser = ff_winners[1] if champ is ff_winners[0] else ff_winners[0]
        results[loser.team] = 5  # Final Four winner, lost in Championship
        results[champ.team] = 6  # Champion

        # Any teams not in results didn't advance past Round 1
        for t in self.bracket.teams:
            if t.team not in results:
                results[t.team] = 0

        return results

    def _play_game(
        self, team_a: Team, team_b: Team, rng: np.random.Generator
    ) -> Team:
        """Simulate a single game. Returns the winner."""
        prob_a = self.model.predict(team_a, team_b)
        return team_a if rng.random() < prob_a else team_b

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def simulate(
        self, n_sims: int = 10_000, seed: int = 0
    ) -> pd.DataFrame:
        """
        Run n_sims full tournament simulations.

        Returns a DataFrame with one row per team and columns:
          team, seed, region, win_pct (R1-R6), champion_pct,
          avg_rounds, median_rounds
        """
        rng = np.random.default_rng(seed)

        # Accumulate round counts per team
        round_counts: dict[str, list[int]] = {t.team: [] for t in self.bracket.teams}

        for _ in range(n_sims):
            res = self._run_once(rng)
            for team, rnd in res.items():
                round_counts[team].append(rnd)

        rows = []
        for t in self.bracket.teams:
            counts = np.array(round_counts[t.team])
            rows.append(
                {
                    "team": t.team,
                    "seed": t.seed,
                    "region": t.region,
                    "adj_em": round(t.adj_em, 1),
                    "r32_pct": round(np.mean(counts >= 1) * 100, 1),
                    "s16_pct": round(np.mean(counts >= 2) * 100, 1),
                    "e8_pct": round(np.mean(counts >= 3) * 100, 1),
                    "f4_pct": round(np.mean(counts >= 4) * 100, 1),
                    "f2_pct": round(np.mean(counts >= 5) * 100, 1),
                    "champ_pct": round(np.mean(counts == 6) * 100, 1),
                    "avg_rounds": round(float(np.mean(counts)), 2),
                }
            )

        df = pd.DataFrame(rows).sort_values("champ_pct", ascending=False)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Single deterministic bracket (highest-probability winner each game)
    # ------------------------------------------------------------------

    def predict_bracket(self) -> list[dict]:
        """
        Return a deterministic bracket prediction where the higher-probability
        team always wins. Useful for filling out a bracket.
        """
        survivors: dict[str, list[Team]] = {r: [] for r in REGIONS}
        for region in REGIONS:
            for s1, s2 in SEED_MATCHUPS:
                t1 = self.bracket.get_team(region, s1)
                t2 = self.bracket.get_team(region, s2)
                winner, prob = self._best_team(t1, t2)
                survivors[region].append(winner)

        predictions = []
        round_num = 1

        for round_num in range(2, 5):
            next_survivors: dict[str, list[Team]] = {r: [] for r in REGIONS}
            for region in REGIONS:
                teams_in_round = survivors[region]
                for i in range(0, len(teams_in_round), 2):
                    t1, t2 = teams_in_round[i], teams_in_round[i + 1]
                    winner, prob = self._best_team(t1, t2)
                    predictions.append(
                        {
                            "round": ROUND_NAMES[round_num],
                            "region": region,
                            "team_a": t1.team,
                            "seed_a": t1.seed,
                            "team_b": t2.team,
                            "seed_b": t2.seed,
                            "prob_winner": round(prob, 4),
                            "predicted_winner": winner.team,
                        }
                    )
                    next_survivors[region].append(winner)
            survivors = next_survivors

        ff_winners = []
        for reg_a, reg_b in SEMIFINAL_PAIRS:
            t1, t2 = survivors[reg_a][0], survivors[reg_b][0]
            winner, prob = self._best_team(t1, t2)
            predictions.append(
                {
                    "round": ROUND_NAMES[5],
                    "region": f"{reg_a}/{reg_b}",
                    "team_a": t1.team,
                    "seed_a": t1.seed,
                    "team_b": t2.team,
                    "seed_b": t2.seed,
                    "prob_winner": round(prob, 4),
                    "predicted_winner": winner.team,
                }
            )
            ff_winners.append(winner)

        champ, prob = self._best_team(ff_winners[0], ff_winners[1])
        predictions.append(
            {
                "round": ROUND_NAMES[6],
                "region": "National",
                "team_a": ff_winners[0].team,
                "seed_a": ff_winners[0].seed,
                "team_b": ff_winners[1].team,
                "seed_b": ff_winners[1].seed,
                "prob_winner": round(prob, 4),
                "predicted_winner": champ.team,
            }
        )

        return predictions

    def _best_team(self, team_a: Team, team_b: Team) -> tuple[Team, float]:
        prob_a = self.model.predict(team_a, team_b)
        if prob_a >= 0.5:
            return team_a, prob_a
        return team_b, 1.0 - prob_a


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_simulation_results(df: pd.DataFrame, top_n: int = 20) -> None:
    print(f"\n{'='*70}")
    print("  MARCH MADNESS TOURNAMENT SIMULATION RESULTS")
    print(f"{'='*70}")
    display = df.head(top_n)[
        ["team", "seed", "region", "adj_em",
         "r32_pct", "s16_pct", "e8_pct", "f4_pct", "champ_pct"]
    ].copy()
    display.columns = [
        "Team", "Seed", "Region", "AdjEM",
        "R32%", "S16%", "E8%", "F4%", "Champ%"
    ]
    print(tabulate(display, headers="keys", tablefmt="rounded_outline", showindex=False))


def print_bracket_predictions(predictions: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("  PREDICTED BRACKET (highest probability winner)")
    print(f"{'='*70}")
    df = pd.DataFrame(predictions)
    for rnd in df["round"].unique():
        rnd_df = df[df["round"] == rnd]
        print(f"\n--- {rnd} ---")
        for _, row in rnd_df.iterrows():
            print(
                f"  ({row['seed_a']}) {row['team_a']:25s} vs "
                f"({row['seed_b']}) {row['team_b']:25s}"
                f"  →  {row['predicted_winner']} ({row['prob_winner']*100:.1f}%)"
            )
