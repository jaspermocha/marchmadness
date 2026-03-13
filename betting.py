"""
betting.py — Betting value analysis for March Madness.

Supports:
  • Moneyline value (model prob vs sportsbook implied prob)
  • Spread betting value (model projected spread vs posted line)
  • Game-total over/under hints (based on tempo/efficiency)
  • Expected Value (EV) calculation
  • Kelly Criterion stake sizing

Usage
-----
Create a list of `GameLine` objects (one per game) with the posted odds,
then call `analyze_lines(lines, model)` to get a DataFrame of value bets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from data import Team
from model import MarchMadnessModel, _prob_to_spread


# ---------------------------------------------------------------------------
# Odds conversion helpers
# ---------------------------------------------------------------------------

def american_to_implied_prob(american_odds: int) -> float:
    """Convert American moneyline odds to implied probability (no vig)."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def implied_prob_to_american(prob: float) -> int:
    """Convert a probability to American odds (rough inverse)."""
    prob = np.clip(prob, 0.001, 0.999)
    if prob >= 0.5:
        return int(-round(prob / (1 - prob) * 100))
    else:
        return int(round((1 - prob) / prob * 100))


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Remove vig from a two-sided market by normalizing implied probs.
    Returns (fair_prob_a, fair_prob_b) that sum to 1.
    """
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds (e.g. 2.10) to American odds."""
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1) * 100))
    else:
        return int(-round(100 / (decimal_odds - 1)))


# ---------------------------------------------------------------------------
# Game line dataclass
# ---------------------------------------------------------------------------

@dataclass
class GameLine:
    """
    Represents the posted betting lines for a single game.

    team_a and team_b must be Team objects.
    All odds fields are optional; omit any lines you don't have.

    ml_a, ml_b : American moneyline odds for team_a / team_b
    spread_a   : Point spread for team_a (e.g. -5.5 means team_a -5.5)
    spread_juice_a : Juice on the spread for team_a (default -110)
    total      : Posted game total (over/under)
    """

    team_a: Team
    team_b: Team
    round_name: str = "Unknown Round"

    ml_a: Optional[int] = None
    ml_b: Optional[int] = None

    spread_a: Optional[float] = None         # e.g. -5.5 (team_a -5.5)
    spread_juice_a: int = -110               # standard -110 juice
    spread_juice_b: int = -110

    total: Optional[float] = None
    total_over_juice: int = -110
    total_under_juice: int = -110

    def implied_prob_a_ml(self) -> Optional[float]:
        """Implied probability for team_a from moneyline (raw, with vig)."""
        if self.ml_a is None:
            return None
        return american_to_implied_prob(self.ml_a)

    def implied_prob_b_ml(self) -> Optional[float]:
        if self.ml_b is None:
            return None
        return american_to_implied_prob(self.ml_b)

    def fair_probs_ml(self) -> Optional[tuple[float, float]]:
        """Vig-removed fair probabilities from the moneyline."""
        pa = self.implied_prob_a_ml()
        pb = self.implied_prob_b_ml()
        if pa is None or pb is None:
            return None
        return remove_vig(pa, pb)


# ---------------------------------------------------------------------------
# Expected value & Kelly
# ---------------------------------------------------------------------------

def expected_value(model_prob: float, odds: int) -> float:
    """
    Calculate expected value per unit bet.
    EV = model_prob * profit_on_win − (1 − model_prob) * 1
    """
    if odds > 0:
        profit = odds / 100.0
    else:
        profit = 100.0 / abs(odds)
    return model_prob * profit - (1.0 - model_prob)


def kelly_fraction(model_prob: float, odds: int, fraction: float = 0.25) -> float:
    """
    Kelly criterion bet size as a fraction of bankroll.
    Uses fractional Kelly (default 25%) for practical risk management.
    Returns 0 if the bet has negative EV.
    """
    if odds > 0:
        b = odds / 100.0
    else:
        b = 100.0 / abs(odds)

    q = 1.0 - model_prob
    kelly_full = (b * model_prob - q) / b
    if kelly_full <= 0:
        return 0.0
    return round(kelly_full * fraction, 4)


# ---------------------------------------------------------------------------
# Value analysis
# ---------------------------------------------------------------------------

BET_TYPES = ["moneyline_a", "moneyline_b", "spread_a", "spread_b"]


@dataclass
class BetOpportunity:
    team: str
    opponent: str
    round_name: str
    bet_type: str
    model_prob: float
    sbook_implied_prob: float
    odds: int
    ev: float
    kelly_quarter: float
    edge: float      # model_prob - sbook_implied_prob

    @property
    def is_value(self) -> bool:
        return self.ev > 0

    def __repr__(self) -> str:
        sign = "+" if self.odds > 0 else ""
        return (
            f"{self.bet_type.upper():15s} | {self.team:25s} vs {self.opponent:25s}"
            f" | odds {sign}{self.odds} | EV={self.ev*100:+.1f}% | "
            f"Kelly={self.kelly_quarter*100:.1f}% | Edge={self.edge*100:+.1f}%"
        )


def analyze_lines(
    lines: list[GameLine],
    model: MarchMadnessModel,
    min_edge: float = 0.02,
) -> pd.DataFrame:
    """
    Analyze a list of game lines and return a DataFrame of value bets.

    Parameters
    ----------
    lines     : list of GameLine objects
    model     : trained MarchMadnessModel
    min_edge  : minimum model_prob − implied_prob to flag as a value bet

    Returns
    -------
    DataFrame sorted by EV descending, filtered to bets with edge >= min_edge.
    """
    opportunities = []

    for line in lines:
        ta, tb = line.team_a, line.team_b
        prob_a = model.predict(ta, tb)
        prob_b = 1.0 - prob_a
        model_spread_a = _prob_to_spread(prob_a)  # positive = ta favored

        # --- Moneyline ---
        if line.ml_a is not None and line.ml_b is not None:
            fair_a, fair_b = line.fair_probs_ml()

            # Team A moneyline
            ev_a = expected_value(prob_a, line.ml_a)
            edge_a = prob_a - fair_a
            if edge_a >= min_edge:
                opportunities.append(
                    BetOpportunity(
                        team=ta.team, opponent=tb.team,
                        round_name=line.round_name,
                        bet_type="moneyline",
                        model_prob=round(prob_a, 4),
                        sbook_implied_prob=round(fair_a, 4),
                        odds=line.ml_a,
                        ev=round(ev_a, 4),
                        kelly_quarter=kelly_fraction(prob_a, line.ml_a),
                        edge=round(edge_a, 4),
                    )
                )

            # Team B moneyline
            ev_b = expected_value(prob_b, line.ml_b)
            edge_b = prob_b - fair_b
            if edge_b >= min_edge:
                opportunities.append(
                    BetOpportunity(
                        team=tb.team, opponent=ta.team,
                        round_name=line.round_name,
                        bet_type="moneyline",
                        model_prob=round(prob_b, 4),
                        sbook_implied_prob=round(fair_b, 4),
                        odds=line.ml_b,
                        ev=round(ev_b, 4),
                        kelly_quarter=kelly_fraction(prob_b, line.ml_b),
                        edge=round(edge_b, 4),
                    )
                )

        # --- Spread ---
        if line.spread_a is not None:
            # team_a covers if they win by more than spread_a (or lose by less)
            # We model the probability of covering as a function of AdjEM diff
            # adjusted for the spread line.
            #
            # Approach: convert the posted spread to an implied win probability
            # using the same logistic function, then compare to model.
            sbook_spread = line.spread_a   # e.g. -5.5 means ta -5.5

            # Implied prob of team_a covering from sbook spread
            # A spread of -5.5 is roughly equivalent to a ~66% win prob at 11 pts/unit
            implied_prob_cover_a = float(1 / (1 + np.exp(sbook_spread / 11.0)))
            implied_prob_cover_b = 1.0 - implied_prob_cover_a

            # Model-based cover probability: same logistic on (model_spread - posted_spread)
            # Covers more often than implied if model spread is bigger than posted
            cover_diff_a = model_spread_a - sbook_spread
            model_prob_cover_a = float(1 / (1 + np.exp(-cover_diff_a / 7.0)))
            model_prob_cover_b = 1.0 - model_prob_cover_a

            fair_implied_a = american_to_implied_prob(line.spread_juice_a)
            fair_implied_b = american_to_implied_prob(line.spread_juice_b)
            fair_cover_a, fair_cover_b = remove_vig(fair_implied_a, fair_implied_b)

            edge_spread_a = model_prob_cover_a - fair_cover_a
            if edge_spread_a >= min_edge:
                ev_s = expected_value(model_prob_cover_a, line.spread_juice_a)
                opportunities.append(
                    BetOpportunity(
                        team=ta.team, opponent=tb.team,
                        round_name=line.round_name,
                        bet_type=f"spread {sbook_spread:+.1f}",
                        model_prob=round(model_prob_cover_a, 4),
                        sbook_implied_prob=round(fair_cover_a, 4),
                        odds=line.spread_juice_a,
                        ev=round(ev_s, 4),
                        kelly_quarter=kelly_fraction(model_prob_cover_a, line.spread_juice_a),
                        edge=round(edge_spread_a, 4),
                    )
                )

            edge_spread_b = model_prob_cover_b - fair_cover_b
            if edge_spread_b >= min_edge:
                ev_s = expected_value(model_prob_cover_b, line.spread_juice_b)
                spread_b_display = -sbook_spread
                opportunities.append(
                    BetOpportunity(
                        team=tb.team, opponent=ta.team,
                        round_name=line.round_name,
                        bet_type=f"spread {spread_b_display:+.1f}",
                        model_prob=round(model_prob_cover_b, 4),
                        sbook_implied_prob=round(fair_cover_b, 4),
                        odds=line.spread_juice_b,
                        ev=round(ev_s, 4),
                        kelly_quarter=kelly_fraction(model_prob_cover_b, line.spread_juice_b),
                        edge=round(edge_spread_b, 4),
                    )
                )

    if not opportunities:
        return pd.DataFrame(
            columns=[
                "team", "opponent", "round", "bet_type", "model_prob",
                "implied_prob", "odds", "ev_pct", "kelly_pct", "edge_pct",
            ]
        )

    rows = [
        {
            "team": o.team,
            "opponent": o.opponent,
            "round": o.round_name,
            "bet_type": o.bet_type,
            "model_prob": f"{o.model_prob*100:.1f}%",
            "implied_prob": f"{o.sbook_implied_prob*100:.1f}%",
            "odds": (f"+{o.odds}" if o.odds > 0 else str(o.odds)),
            "ev_pct": f"{o.ev*100:+.1f}%",
            "kelly_pct": f"{o.kelly_quarter*100:.1f}%",
            "edge_pct": f"{o.edge*100:+.1f}%",
        }
        for o in sorted(opportunities, key=lambda x: x.ev, reverse=True)
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Game total estimation
# ---------------------------------------------------------------------------

def estimate_game_total(team_a: Team, team_b: Team) -> float:
    """
    Rough game total estimate based on pace and offensive efficiency.

    Formula: each team's expected points per possession × combined possessions.
    Points per possession ≈ adj_o / 100
    Possessions per game ≈ avg of both tempos × 40/40 (normalised)
    """
    avg_possessions = (team_a.adj_tempo + team_b.adj_tempo) / 2.0
    ppp_a = team_a.adj_o / 100.0
    ppp_b = team_b.adj_o / 100.0
    # Each team scores against opponent's defense
    pts_a = ppp_a * avg_possessions * (1 - (team_b.adj_d - 90) / 100.0)
    pts_b = ppp_b * avg_possessions * (1 - (team_a.adj_d - 90) / 100.0)
    return round(pts_a + pts_b, 1)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_value_bets(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("  VALUE BET OPPORTUNITIES")
    print(f"{'='*70}")
    if df.empty:
        print("  No value bets found at current edge threshold.")
        return
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    print(
        "\n  NOTE: Kelly % is 1/4 Kelly. Bet sizing is for informational use only.\n"
        "  Past model performance does not guarantee future results.\n"
        "  Always bet responsibly and within legal jurisdictions.\n"
    )
