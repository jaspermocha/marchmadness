"""
Dynasty Fantasy Football Trade Analyzer

Evaluates dynasty trade packages using age curves, positional value,
performance metrics, and team roster composition.

Also fetches live dynasty values and ADP from FantasyCalc.com to surface
undervalued / overvalued players relative to the market.

Usage:
    python main.py --dynasty                         # Interactive trade analyzer
    python main.py --dynasty --market                # Show undervalued players from FantasyCalc
    python main.py --dynasty --roster my_roster.csv  # Analyze with your roster
    python main.py --dynasty --trade "CeeDee Lamb, 2025 1st" "Ja'Marr Chase"
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Player:
    name: str
    position: str          # QB, RB, WR, TE
    nfl_team: str
    age: float
    years_pro: int
    dynasty_rank: int      # Overall dynasty rank (1 = best)
    positional_rank: int   # Rank within position
    recent_ppg: float      # PPR points per game (last season)
    dynasty_value: int     # KTC-style 0–10000 scale
    tier: int              # 1=elite, 2=starter, 3=flex, 4=deep
    injury_risk: str       # low / medium / high
    situation: str         # starter / handcuff / backup / redzone
    notes: str = ""

    def buy_window(self) -> tuple[int, int]:
        """Returns (years_remaining_peak, total_years_left) estimate."""
        peak_end = POSITION_AGE_CURVES[self.position]["peak_end"]
        decline_end = POSITION_AGE_CURVES[self.position]["decline_end"]
        years_in_peak = max(0, peak_end - self.age)
        years_total = max(0, decline_end - self.age)
        return int(round(years_in_peak)), int(round(years_total))

    def age_multiplier(self) -> float:
        """Value multiplier based on age vs position curve."""
        curve = POSITION_AGE_CURVES[self.position]
        age = self.age
        peak_start = curve["peak_start"]
        peak_end = curve["peak_end"]
        decline_end = curve["decline_end"]

        if age < peak_start:
            # Rising — discount for unproven upside
            dist = peak_start - age
            return max(0.60, 1.0 - dist * 0.05)
        elif age <= peak_end:
            return 1.0
        elif age <= decline_end:
            dist = age - peak_end
            range_ = decline_end - peak_end
            return max(0.15, 1.0 - (dist / range_) * 0.85)
        else:
            return 0.05

    def adjusted_value(self, league_format: str = "2QB") -> int:
        """Dynasty value adjusted for age, situation, injury risk, and league format."""
        val = self.dynasty_value
        val *= self.age_multiplier()
        # Situation adjustment
        sit_mult = {"starter": 1.0, "handcuff": 0.7, "backup": 0.5, "redzone": 0.75}
        val *= sit_mult.get(self.situation, 1.0)
        # Injury risk discount
        inj_mult = {"low": 1.0, "medium": 0.92, "high": 0.82}
        val *= inj_mult.get(self.injury_risk, 1.0)
        # 2QB leagues: QBs fill a scarce second starting slot
        if self.position == "QB" and league_format in ("2QB", "SF"):
            val *= QB_2QB_MULT
        return int(val)


@dataclass
class DraftPick:
    year: int
    round: int
    pick_type: str = "mid"   # early / mid / late
    original_team: str = ""

    @property
    def name(self) -> str:
        suffix = {"early": " (early)", "mid": "", "late": " (late)"}
        label = suffix.get(self.pick_type, "")
        return f"{self.year} {self.round}{self._round_suffix()}{label}"

    def _round_suffix(self) -> str:
        return {1: "st", 2: "nd", 3: "rd"}.get(self.round, "th")

    def value(self) -> int:
        """Draft pick value on the 0-10000 KTC scale."""
        current_year = 2026
        year_discount = max(0, self.year - current_year)
        base = PICK_BASE_VALUES.get(self.round, 500)
        type_mult = {"early": 1.25, "mid": 1.0, "late": 0.78}
        val = base * type_mult.get(self.pick_type, 1.0)
        # Future picks discounted ~15% per year out
        val *= (0.85 ** year_discount)
        return int(val)

    @property
    def position(self) -> str:
        return "PICK"

    def adjusted_value(self, league_format: str = "2QB") -> int:
        return self.value()


# ---------------------------------------------------------------------------
# League Configuration  (your specific league settings)
# ---------------------------------------------------------------------------

LEAGUE_CONFIG = {
    "teams":         12,
    "num_qbs":       2,       # 2QB format
    "ppr":           1.0,     # Full PPR
    "te_premium":    False,   # No TE premium scoring
    "format":        "2QB",
    # Starters: QB QB RB RB WR WR WR TE FLEX FLEX
    "starters": {"QB": 2, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2},
}

# In 2QB leagues QBs fill a scarce second starting slot — value jumps ~38%.
# Applied on top of age/situation adjustments.
QB_2QB_MULT = 1.38

# ---------------------------------------------------------------------------
# Age Curves by Position
# ---------------------------------------------------------------------------

POSITION_AGE_CURVES = {
    "QB":  {"peak_start": 25, "peak_end": 33, "decline_end": 38},
    "RB":  {"peak_start": 21, "peak_end": 26, "decline_end": 30},
    "WR":  {"peak_start": 22, "peak_end": 29, "decline_end": 34},
    "TE":  {"peak_start": 24, "peak_end": 30, "decline_end": 35},
}

PICK_BASE_VALUES = {1: 4800, 2: 2200, 3: 1100, 4: 550, 5: 280}

# ---------------------------------------------------------------------------
# Player Database  (KTC-calibrated values as of 2026 offseason)
# ---------------------------------------------------------------------------

def build_player_db() -> dict[str, Player]:
    raw = [
        # name, pos, team, age, yrs_pro, dynasty_rank, pos_rank, ppg, value, tier, injury, situation
        ("CeeDee Lamb",       "WR", "DAL",  25, 5,  1,  1, 25.1, 9800, 1, "low",    "starter"),
        ("Ja'Marr Chase",     "WR", "CIN",  25, 4,  2,  2, 24.3, 9600, 1, "low",    "starter"),
        ("Justin Jefferson",  "WR", "MIN",  26, 5,  3,  3, 22.8, 9300, 1, "low",    "starter"),
        ("Amon-Ra St. Brown", "WR", "DET",  25, 4,  4,  4, 22.2, 8700, 1, "low",    "starter"),
        ("Puka Nacua",        "WR", "LAR",  24, 2,  5,  5, 19.5, 7900, 1, "low",    "starter"),
        ("Garrett Wilson",    "WR", "NYJ",  24, 3,  6,  6, 18.8, 7700, 1, "low",    "starter"),
        ("Drake London",      "WR", "ATL",  23, 3,  7,  7, 17.2, 7200, 1, "low",    "starter"),
        ("Stefon Diggs",      "WR", "NE",   32, 10, 55, 22,  9.1, 3100, 3, "medium", "starter"),
        ("Tyreek Hill",       "WR", "MIA",  32, 9,  60, 24, 19.8, 3500, 3, "low",    "starter"),
        ("Cooper Kupp",       "WR", "LAR",  32, 8,  70, 28,  8.2, 2400, 4, "high",   "starter"),
        ("Davante Adams",     "WR", "LV",   33, 11, 80, 30, 12.5, 2000, 4, "medium", "starter"),
        ("DeVonta Smith",     "WR", "PHI",  28, 4,  15, 10, 16.3, 6800, 2, "low",    "starter"),
        ("Tee Higgins",       "WR", "CIN",  26, 5,  18, 11, 15.1, 6400, 2, "medium", "starter"),
        ("Chris Olave",       "WR", "NO",   24, 3,  12, 8,  17.0, 7100, 1, "high",   "starter"),
        ("Jordan Addison",    "WR", "MIN",  23, 2,  16, 9,  14.8, 6500, 2, "low",    "starter"),
        ("Rome Odunze",       "WR", "CHI",  22, 1,  20, 14, 13.2, 6000, 2, "low",    "starter"),
        ("Marvin Harrison Jr.","WR","ARI",  22, 1,  14, 9,  15.6, 7000, 1, "low",    "starter"),
        ("Brian Thomas Jr.",  "WR", "JAX",  22, 1,  22, 15, 14.1, 5800, 2, "low",    "starter"),
        ("Xavier Worthy",     "WR", "KC",   21, 1,  28, 18, 11.0, 5000, 2, "low",    "starter"),
        ("Ladd McConkey",     "WR", "LAC",  23, 1,  25, 17, 13.8, 5500, 2, "low",    "starter"),

        # RBs
        ("Breece Hall",       "RB", "NYJ",  23, 3,  8,  1, 20.5, 8500, 1, "medium", "starter"),
        ("Bijan Robinson",    "RB", "ATL",  23, 2,  9,  2, 22.1, 8700, 1, "low",    "starter"),
        ("Jahmyr Gibbs",      "RB", "DET",  23, 2,  10, 3, 21.8, 8600, 1, "low",    "starter"),
        ("De'Von Achane",     "RB", "MIA",  23, 2,  11, 4, 23.4, 8900, 1, "high",   "starter"),
        ("Jonathon Brooks",   "RB", "CAR",  22, 1,  13, 5, 11.0, 6200, 2, "high",   "starter"),
        ("Rashee Rice",       "WR", "KC",   24, 2,  19, 12, 16.0, 6300, 2, "medium", "starter"),
        ("James Cook",        "RB", "BUF",  24, 3,  17, 6, 18.2, 7200, 2, "low",    "starter"),
        ("Derrick Henry",     "RB", "BAL",  32, 9,  65, 25, 17.5, 2800, 3, "medium", "starter"),
        ("Tony Pollard",      "RB", "TEN",  28, 5,  40, 18, 12.1, 4200, 3, "medium", "starter"),
        ("Josh Jacobs",       "RB", "GB",   27, 5,  35, 15, 14.3, 4800, 3, "low",    "starter"),
        ("Alvin Kamara",      "RB", "NO",   30, 8,  58, 22, 10.5, 3200, 3, "high",   "starter"),
        ("Saquon Barkley",    "RB", "PHI",  29, 7,  50, 20, 19.8, 3800, 3, "high",   "starter"),
        ("Isiah Pacheco",     "RB", "KC",   26, 3,  30, 13, 13.5, 5200, 2, "high",   "starter"),
        ("Tank Bigsby",       "RB", "JAX",  23, 2,  26, 11, 12.0, 5100, 2, "low",    "starter"),
        ("Tyjae Spears",      "RB", "TEN",  23, 2,  32, 14, 11.5, 4700, 2, "low",    "handcuff"),
        ("MarShawn Lloyd",    "RB", "GB",   22, 1,  38, 16, 10.2, 4000, 3, "low",    "handcuff"),
        ("Ray Davis",         "RB", "BUF",  24, 1,  45, 19, 9.5,  3500, 3, "low",    "handcuff"),

        # QBs
        ("Patrick Mahomes",   "QB", "KC",   30, 7,  23, 1, 28.5, 7800, 1, "low",    "starter"),
        ("Josh Allen",        "QB", "BUF",  30, 7,  24, 2, 30.1, 7500, 1, "low",    "starter"),
        ("Lamar Jackson",     "QB", "BAL",  29, 7,  27, 3, 31.2, 7200, 1, "low",    "starter"),
        ("Jalen Hurts",       "QB", "PHI",  27, 5,  29, 4, 27.8, 7000, 1, "medium", "starter"),
        ("C.J. Stroud",       "QB", "HOU",  23, 2,  31, 5, 22.5, 7800, 1, "low",    "starter"),
        ("Caleb Williams",    "QB", "CHI",  24, 1,  34, 6, 18.5, 7200, 1, "low",    "starter"),
        ("Anthony Richardson","QB", "IND",  23, 2,  36, 7, 17.5, 6800, 2, "high",   "starter"),
        ("Drake Maye",        "QB", "NE",   23, 1,  39, 8, 16.5, 6500, 2, "low",    "starter"),
        ("Joe Burrow",        "QB", "CIN",  29, 4,  33, 5, 25.5, 6800, 2, "high",   "starter"),
        ("Dak Prescott",      "QB", "DAL",  32, 10, 62, 14, 21.5, 3800, 3, "high",   "starter"),
        ("Tua Tagovailoa",    "QB", "MIA",  28, 5,  44, 10, 22.1, 5200, 2, "high",   "starter"),

        # TEs
        ("Sam LaPorta",       "TE", "DET",  24, 2,  21, 1, 16.8, 8200, 1, "medium", "starter"),
        ("Brock Bowers",      "TE", "LV",   22, 1,  6,  1, 19.5, 9200, 1, "low",    "starter"),
        ("Trey McBride",      "TE", "ARI",  25, 3,  13, 2, 17.2, 8000, 1, "low",    "starter"),
        ("Dalton Kincaid",    "TE", "BUF",  25, 2,  37, 6, 10.5, 5000, 2, "medium", "starter"),
        ("Ja'Tavion Sanders", "TE", "CAR",  23, 1,  42, 7, 9.8,  4500, 2, "low",    "starter"),
        ("Tucker Kraft",      "TE", "GB",   24, 2,  46, 8, 9.2,  4200, 2, "low",    "starter"),
        ("Luke Musgrave",     "TE", "GB",   24, 2,  48, 9, 8.8,  3900, 3, "high",   "starter"),
        ("Travis Kelce",      "TE", "KC",   36, 13, 90, 20, 13.5, 1500, 4, "medium", "starter"),
        ("Mark Andrews",      "TE", "BAL",  30, 6,  52, 12, 12.5, 3500, 3, "medium", "starter"),
    ]

    db: dict[str, Player] = {}
    for r in raw:
        p = Player(*r)
        db[p.name.lower()] = p
    return db


PLAYER_DB: dict[str, Player] = build_player_db()


# ---------------------------------------------------------------------------
# Trade Package
# ---------------------------------------------------------------------------

@dataclass
class TradeAsset:
    """A single asset in a trade — either a Player or a DraftPick."""
    asset: Player | DraftPick

    @property
    def name(self) -> str:
        return self.asset.name

    @property
    def position(self) -> str:
        return self.asset.position

    def value(self, league_format: str = "2QB") -> int:
        return self.asset.adjusted_value(league_format)


@dataclass
class TradeSide:
    label: str
    assets: list[TradeAsset] = field(default_factory=list)

    def total_value(self, league_format: str = "2QB") -> int:
        return sum(a.value(league_format) for a in self.assets)

    def positions(self) -> list[str]:
        return [a.position for a in self.assets]


# ---------------------------------------------------------------------------
# Trade Evaluator
# ---------------------------------------------------------------------------

class TradeEvaluator:

    def evaluate(self, side_a: TradeSide, side_b: TradeSide,
                 league_format: str = "2QB") -> dict:
        val_a = side_a.total_value(league_format)
        val_b = side_b.total_value(league_format)
        total = max(val_a + val_b, 1)
        diff = val_a - val_b
        pct_diff = abs(diff) / max(val_a, val_b) * 100

        # Grade
        grade = self._grade(pct_diff, diff, side_a, side_b)
        winner = side_a.label if diff > 0 else (side_b.label if diff < 0 else "Even")
        verdict = self._verdict(pct_diff, diff, side_a, side_b)

        # Age/window analysis
        a_avg_age = self._avg_age(side_a)
        b_avg_age = self._avg_age(side_b)
        a_window = self._total_window(side_a)
        b_window = self._total_window(side_b)

        return {
            "side_a_value": val_a,
            "side_b_value": val_b,
            "difference": diff,
            "pct_difference": round(pct_diff, 1),
            "winner": winner,
            "grade": grade,
            "verdict": verdict,
            "side_a_avg_age": a_avg_age,
            "side_b_avg_age": b_avg_age,
            "side_a_window": a_window,
            "side_b_window": b_window,
        }

    def _grade(self, pct_diff: float, diff: int, side_a: TradeSide, side_b: TradeSide) -> str:
        """Grade the trade from the perspective of the winning side."""
        if pct_diff <= 5:
            return "A  (Even trade)"
        elif pct_diff <= 12:
            return "B  (Slight win)"
        elif pct_diff <= 22:
            return "C  (Moderate win)"
        elif pct_diff <= 35:
            return "D  (Significant overpay)"
        else:
            return "F  (Highway robbery)"

    def _verdict(self, pct_diff: float, diff: int, side_a: TradeSide, side_b: TradeSide) -> str:
        if pct_diff <= 5:
            return "Fair trade — both sides receive comparable value."
        winner = side_a if diff > 0 else side_b
        loser  = side_b if diff > 0 else side_a
        if pct_diff <= 12:
            return f"{winner.label} wins this trade slightly. Reasonable deal for both parties."
        elif pct_diff <= 22:
            return f"{winner.label} wins this trade. {loser.label} is overpaying — consider countering."
        elif pct_diff <= 35:
            return f"{winner.label} wins significantly. {loser.label} should push for a better return."
        else:
            return f"Lopsided deal heavily favoring {winner.label}. {loser.label} should decline."

    def _avg_age(self, side: TradeSide) -> Optional[float]:
        ages = [a.asset.age for a in side.assets if isinstance(a.asset, Player)]
        return round(sum(ages) / len(ages), 1) if ages else None

    def _total_window(self, side: TradeSide) -> Optional[int]:
        windows = []
        for a in side.assets:
            if isinstance(a.asset, Player):
                _, total = a.asset.buy_window()
                windows.append(total)
        return sum(windows) if windows else None


# ---------------------------------------------------------------------------
# Roster Analyzer
# ---------------------------------------------------------------------------

# Starter counts per format: positional starters only (FLEX counted separately)
ROSTER_POSITIONS = {
    "1QB": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1},
    "SF":  {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "SF": 1},
    "2QB": {"QB": 2, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2},   # your league
}

# Minimum usable adjusted value to be considered "startable" in a 12-team league
STARTABLE_THRESHOLD = {
    "QB": 5500,   # 2QB: QB2 in weak league ~5500
    "RB": 3500,   # RB3/FLEX floor
    "WR": 3500,   # WR4/FLEX floor
    "TE": 2800,   # TE1 with no premium; competes on raw PPR output only
}


class RosterAnalyzer:

    def analyze(self, roster: list[Player],
                league_format: str = "2QB",
                te_premium: bool = False) -> dict:
        """
        Analyze roster health for a 12-team 2QB full PPR no-TE-premium league.

        te_premium=False (your league): TEs are valued purely on PPR output and
        compete with WRs/RBs for FLEX spots — no positional scarcity bonus.
        """
        by_pos: dict[str, list[Player]] = {"QB": [], "RB": [], "WR": [], "TE": []}
        for p in roster:
            by_pos.setdefault(p.position, []).append(p)

        for pos in by_pos:
            by_pos[pos].sort(key=lambda p: p.adjusted_value(league_format), reverse=True)

        fmt = ROSTER_POSITIONS.get(league_format, ROSTER_POSITIONS["2QB"])
        num_flex = fmt.get("FLEX", 0)

        depth: dict[str, str] = {}
        needs: list[str] = []
        surpluses: list[str] = []
        sell_high: list[str] = []
        buy_low: list[str] = []

        for pos, players in by_pos.items():
            num_starters = fmt.get(pos, 1)
            threshold = STARTABLE_THRESHOLD.get(pos, 3500)

            starters = players[:num_starters]
            bench = players[num_starters:]

            starter_avg = (
                sum(p.adjusted_value(league_format) for p in starters) / max(len(starters), 1)
                if starters else 0
            )
            startable_bench = [p for p in bench if p.adjusted_value(league_format) >= threshold]

            # TE with no premium: lower the need bar — you just need one reliable TE;
            # extra TEs rarely win FLEX spots vs. elite WR/RB depth.
            if pos == "TE" and not te_premium:
                te_floor = STARTABLE_THRESHOLD["TE"]
                if not starters or starters[0].adjusted_value(league_format) < te_floor:
                    needs.append(pos)
                    depth[pos] = "NEED"
                else:
                    # Surplus only if you have 2+ startable TEs and don't need them
                    if len(startable_bench) >= 1 and starter_avg > 5000:
                        surpluses.append(pos)
                        depth[pos] = "SURPLUS"
                    else:
                        depth[pos] = "OK"

            elif pos == "QB":
                # 2QB: need 2 real starters
                if len([p for p in players if p.adjusted_value(league_format) >= threshold]) < num_starters:
                    needs.append(pos)
                    depth[pos] = "NEED"
                elif starter_avg > 9000 and len(startable_bench) >= 1:
                    surpluses.append(pos)
                    depth[pos] = "SURPLUS"
                else:
                    depth[pos] = "OK"

            else:
                # RB / WR — account for FLEX slots (each pos can absorb ~1 flex on avg)
                effective_need = num_starters + max(0, num_flex - 1)  # conservative
                total_startable = len([p for p in players if p.adjusted_value(league_format) >= threshold])

                if total_startable < num_starters:
                    needs.append(pos)
                    depth[pos] = "NEED"
                elif total_startable >= effective_need + 2 and len(startable_bench) >= 2:
                    surpluses.append(pos)
                    depth[pos] = "SURPLUS"
                else:
                    depth[pos] = "OK"

            # Sell-high: entering decline window with meaningful current value
            for p in players:
                curve = POSITION_AGE_CURVES[p.position]
                near_decline = p.age >= curve["peak_end"] - 1
                val_floor = 5000 if p.position == "QB" else 3800
                if near_decline and p.adjusted_value(league_format) > val_floor:
                    sell_high.append(p.name)

            # Buy-low: young talent in limited role that could break out
            for p in players:
                if p.tier <= 2 and p.age <= 24 and p.situation in ("handcuff", "backup"):
                    buy_low.append(p.name)

        roster_value = sum(p.adjusted_value(league_format) for p in roster)
        age_score = self._age_score(roster, league_format)

        return {
            "roster_value": roster_value,
            "depth": depth,
            "needs": needs,
            "surpluses": surpluses,
            "sell_high_candidates": sell_high,
            "buy_low_candidates": buy_low,
            "age_score": age_score,
            "window": self._window_label(age_score),
            "by_position": {pos: [p.name for p in plist] for pos, plist in by_pos.items()},
        }

    def _age_score(self, roster: list[Player], league_format: str = "2QB") -> float:
        """0-100: 100 = very young (rebuild), 0 = very old (win-now)."""
        if not roster:
            return 50.0
        weighted_ages = []
        for p in roster:
            weight = p.adjusted_value(league_format) / 1000
            weighted_ages.append(p.age * weight)
        total_weight = sum(p.adjusted_value(league_format) / 1000 for p in roster)
        if total_weight == 0:
            return 50.0
        avg_age = sum(weighted_ages) / total_weight
        score = max(0, min(100, (30 - avg_age) / 7 * 70 + 20))
        return round(score, 1)

    def _window_label(self, age_score: float) -> str:
        if age_score >= 75:
            return "Rebuild — invest in youth & picks"
        elif age_score >= 55:
            return "Retool — mix of youth and production"
        elif age_score >= 35:
            return "Win-now — peak of contention window"
        else:
            return "Aging — consider selling vets for youth"


# ---------------------------------------------------------------------------
# Trade Recommender
# ---------------------------------------------------------------------------

class TradeRecommender:

    def recommend(
        self,
        roster: list[Player],
        league_format: str = "2QB",
        te_premium: bool = False,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Suggest trade targets based on roster needs and surplus positions.
        Returns a list of recommended trades.
        """
        analyzer = RosterAnalyzer()
        analysis = analyzer.analyze(roster, league_format, te_premium=te_premium)
        needs = analysis["needs"]
        surpluses = analysis["surpluses"]
        sell_high = analysis["sell_high_candidates"]

        roster_names = {p.name.lower() for p in roster}
        recommendations = []

        # Strategy 1: Address needs by trading surplus
        for need_pos in needs:
            for surplus_pos in surpluses:
                surplus_players = [
                    p for p in roster
                    if p.position == surplus_pos and p.name not in sell_high
                ]
                target_players = [
                    p for p in PLAYER_DB.values()
                    if p.position == need_pos and p.name.lower() not in roster_names
                    and p.tier <= 2
                ]
                surplus_players.sort(key=lambda p: p.adjusted_value(league_format), reverse=True)
                target_players.sort(key=lambda p: p.adjusted_value(league_format), reverse=True)

                if surplus_players and target_players:
                    give = surplus_players[0]
                    get  = target_players[0]
                    val_ratio = give.adjusted_value(league_format) / max(get.adjusted_value(league_format), 1)
                    recommendations.append({
                        "type": "Need/Surplus",
                        "action": f"Trade {give.name} ({surplus_pos}) for {get.name} ({need_pos})",
                        "give": give.name,
                        "get":  get.name,
                        "value_give": give.adjusted_value(league_format),
                        "value_get":  get.adjusted_value(league_format),
                        "fairness": round(val_ratio, 2),
                        "rationale": (
                            f"Your {surplus_pos} depth is strong — convert it to address "
                            f"your {need_pos} need. {get.name} is a top-{need_pos} option."
                        ),
                    })

        # Strategy 2: Sell-high on aging studs
        for name in sell_high:
            player = next((p for p in roster if p.name == name), None)
            if player is None:
                continue
            pval = player.adjusted_value(league_format)
            target_picks = [
                DraftPick(2026, 1, "early"),
                DraftPick(2026, 1, "mid"),
                DraftPick(2027, 1, "early"),
            ]
            for pick in target_picks:
                if pval * 0.9 <= pick.value() <= pval * 1.3:
                    recommendations.append({
                        "type": "Sell High",
                        "action": f"Trade {player.name} for a {pick.name}",
                        "give": player.name,
                        "get":  pick.name,
                        "value_give": pval,
                        "value_get":  pick.value(),
                        "fairness": round(pick.value() / max(pval, 1), 2),
                        "rationale": (
                            f"{player.name} is {player.age} yrs old and approaching/entering decline. "
                            f"Sell while value is still high and reinvest in draft capital."
                        ),
                    })

        # Strategy 3: Buy-low on undervalued youth
        for name in analysis["buy_low_candidates"]:
            player = PLAYER_DB.get(name.lower())
            if player and player.name.lower() not in roster_names:
                recommendations.append({
                    "type": "Buy Low",
                    "action": f"Target {player.name} — buy low on young talent",
                    "give": "(varies)",
                    "get":  player.name,
                    "value_give": None,
                    "value_get":  player.adjusted_value(league_format),
                    "fairness": None,
                    "rationale": (
                        f"{player.name} ({player.position}, age {player.age}) has elite upside "
                        f"but is in a limited role ({player.situation}). Ownership/coaching changes "
                        f"could unlock massive value."
                    ),
                })

        return recommendations[:top_n]


# ---------------------------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------------------------

def parse_asset(token: str) -> TradeAsset | None:
    """
    Parse a player name or draft pick string into a TradeAsset.

    Examples:
        "CeeDee Lamb"  → Player asset
        "2026 1st"     → DraftPick(2026, 1, 'mid')
        "2027 2nd early" → DraftPick(2027, 2, 'early')
    """
    token = token.strip()
    lower = token.lower()

    # Try draft pick pattern: "2026 1st", "2027 2nd early", etc.
    import re
    pick_match = re.match(
        r"(\d{4})\s+(1st|2nd|3rd|4th|5th|\d+(?:st|nd|rd|th)?)"
        r"(?:\s+(early|mid|late))?",
        lower,
    )
    if pick_match:
        year = int(pick_match.group(1))
        round_str = pick_match.group(2)
        pick_type = pick_match.group(3) or "mid"
        round_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}
        rnd = round_map.get(round_str, int(re.sub(r"\D", "", round_str) or 1))
        return TradeAsset(DraftPick(year, rnd, pick_type))

    # Try player lookup (fuzzy)
    if lower in PLAYER_DB:
        return TradeAsset(PLAYER_DB[lower])

    # Partial name match
    matches = [name for name in PLAYER_DB if lower in name]
    if len(matches) == 1:
        return TradeAsset(PLAYER_DB[matches[0]])
    if len(matches) > 1:
        # Prefer exact word matches
        word_matches = [name for name in matches if lower.split()[0] in name.split()]
        if word_matches:
            return TradeAsset(PLAYER_DB[word_matches[0]])
        return TradeAsset(PLAYER_DB[matches[0]])

    return None


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

def _bar(value: int, max_val: int = 10000, width: int = 20) -> str:
    filled = int(value / max_val * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def print_trade_analysis(side_a: TradeSide, side_b: TradeSide,
                         league_format: str = "2QB") -> None:
    evaluator = TradeEvaluator()
    result = evaluator.evaluate(side_a, side_b, league_format)

    print("\n" + "═" * 60)
    print("  DYNASTY TRADE ANALYZER")
    print("═" * 60)

    for side, label in [(side_a, side_a.label), (side_b, side_b.label)]:
        print(f"\n  {label}:")
        for asset in side.assets:
            a = asset.asset
            v = asset.value(league_format)
            if isinstance(a, Player):
                peak_left, total_left = a.buy_window()
                print(
                    f"    {a.name:<25} {a.position:<3}  "
                    f"Age {a.age:.0f}  "
                    f"Val {v:>5}  "
                    f"{_bar(v)}  "
                    f"~{peak_left}yr peak / {total_left}yr window"
                )
            else:
                print(
                    f"    {a.name:<25} PICK  "
                    f"Val {v:>5}  "
                    f"{_bar(v)}"
                )
        print(f"    {'─'*55}")
        print(f"    {'TOTAL VALUE':>30}  {side.total_value(league_format):>5}")

    print("\n  " + "─" * 56)
    val_a, val_b = result["side_a_value"], result["side_b_value"]
    print(f"  {side_a.label:<20} {val_a:>6}  vs  {side_b.label:<20} {val_b:>6}")
    print(f"  Difference: {abs(result['difference']):,} pts  ({result['pct_difference']}%)")
    print(f"  Winner:  {result['winner']}")
    print(f"  Grade:   {result['grade']}")
    print(f"\n  Verdict: {result['verdict']}")

    # Age / window breakdown
    if result["side_a_avg_age"] and result["side_b_avg_age"]:
        print(f"\n  Avg Age:  {side_a.label}: {result['side_a_avg_age']}   {side_b.label}: {result['side_b_avg_age']}")
    if result["side_a_window"] is not None and result["side_b_window"] is not None:
        print(f"  Yrs Left: {side_a.label}: {result['side_a_window']}yr   {side_b.label}: {result['side_b_window']}yr")

    print("\n" + "═" * 60 + "\n")


def print_roster_analysis(analysis: dict, league_format: str = "2QB") -> None:
    print("\n" + "═" * 60)
    print(f"  ROSTER ANALYSIS  ({league_format}, Full PPR, No TE Premium, 12-Team)")
    print("═" * 60)
    print(f"  Total Roster Value : {analysis['roster_value']:,}")
    print(f"  Team Age Score     : {analysis['age_score']} / 100")
    print(f"  Contention Window  : {analysis['window']}")

    print("\n  Position Depth:")
    for pos, status in analysis["depth"].items():
        icon = {"NEED": "⚠ ", "SURPLUS": "✓ ", "OK": "  "}.get(status, "  ")
        players = analysis["by_position"].get(pos, [])
        names = ", ".join(players[:4]) + (" ..." if len(players) > 4 else "")
        print(f"    {icon}{pos:<3}  {status:<8}  {names}")

    if analysis["sell_high_candidates"]:
        print(f"\n  Sell High:  {', '.join(analysis['sell_high_candidates'])}")
    if analysis["buy_low_candidates"]:
        print(f"  Buy Low:    {', '.join(analysis['buy_low_candidates'])}")
    print("═" * 60 + "\n")


def print_recommendations(recs: list[dict]) -> None:
    if not recs:
        print("  No specific recommendations — roster looks balanced.\n")
        return

    print("\n" + "═" * 60)
    print("  TRADE RECOMMENDATIONS")
    print("═" * 60)
    for i, rec in enumerate(recs, 1):
        print(f"\n  {i}. [{rec['type']}]  {rec['action']}")
        if rec["value_give"] and rec["value_get"]:
            print(f"     Give: {rec['value_give']:,}  →  Get: {rec['value_get']:,}  (ratio {rec['fairness']})")
        print(f"     Rationale: {rec['rationale']}")
    print("\n" + "═" * 60 + "\n")


# ---------------------------------------------------------------------------
# FantasyCalc Market Data Fetcher
# ---------------------------------------------------------------------------

FANTASYCALC_API = "https://api.fantasycalc.com/values/current"
_FC_CACHE: dict = {}
_FC_CACHE_TIME: float = 0
_FC_CACHE_TTL = 300  # 5 minutes


def fetch_fantasycalc_values(
    num_qbs: int = 1,
    ppr: float = 1.0,
    dynasty: bool = True,
    include_adp: bool = True,
) -> list[dict]:
    """
    Fetch current dynasty player values and ADP from FantasyCalc.

    Returns a list of dicts with keys:
        name, position, team, value, overallRank, positionRank, age, adp
    """
    import urllib.request
    global _FC_CACHE, _FC_CACHE_TIME

    cache_key = f"{num_qbs}_{ppr}_{dynasty}"
    now = time.time()
    if _FC_CACHE.get(cache_key) and (now - _FC_CACHE_TIME) < _FC_CACHE_TTL:
        return _FC_CACHE[cache_key]

    params = (
        f"?isDynasty={'true' if dynasty else 'false'}"
        f"&numQbs={num_qbs}"
        f"&ppr={ppr}"
        f"&includeAdp={'true' if include_adp else 'false'}"
    )
    url = FANTASYCALC_API + params

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 dynasty-analyzer/1.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [FantasyCalc] Fetch failed: {e}")
        return []

    # API returns a list of objects; each has a 'player' sub-object and 'value'
    results = []
    for entry in raw:
        player_info = entry.get("player", {})
        adp_info = entry.get("overallAdp") or entry.get("adp") or {}
        results.append({
            "name":          player_info.get("name", ""),
            "position":      player_info.get("position", ""),
            "team":          player_info.get("maybeTeam") or player_info.get("team", ""),
            "age":           player_info.get("age") or player_info.get("maybeAge"),
            "value":         entry.get("value", 0),
            "overall_rank":  entry.get("overallRank") or entry.get("rank"),
            "pos_rank":      entry.get("positionRank"),
            "adp":           entry.get("overallAdp") or entry.get("adp"),
            "rookie":        player_info.get("rookie", False),
        })

    _FC_CACHE[cache_key] = results
    _FC_CACHE_TIME = now
    return results


# ---------------------------------------------------------------------------
# Undervalue Analyzer — compares internal model vs FantasyCalc market
# ---------------------------------------------------------------------------

@dataclass
class MarketComparison:
    player: Player
    market_value: int      # FantasyCalc value (0-10000)
    market_rank: int
    model_value: int       # Our adjusted value
    adp: Optional[float]
    delta: int             # model_value - market_value  (positive = undervalued by market)
    delta_pct: float       # % difference


def find_undervalued_players(
    num_qbs: int = 2,
    ppr: float = 1.0,
    top_n: int = 20,
    min_value: int = 1500,
    league_format: str = "2QB",
) -> tuple[list[MarketComparison], list[MarketComparison]]:
    """
    Compare internal model values against FantasyCalc market values.

    Returns (undervalued, overvalued) — each a sorted list of MarketComparison.
    undervalued: model thinks player is worth MORE than market (buy target)
    overvalued:  model thinks player is worth LESS than market (sell target)
    """
    fc_data = fetch_fantasycalc_values(num_qbs=num_qbs, ppr=ppr)
    if not fc_data:
        return [], []

    comparisons: list[MarketComparison] = []

    for entry in fc_data:
        if entry["position"] not in ("QB", "RB", "WR", "TE"):
            continue
        mkt_value = entry.get("value", 0)
        if mkt_value < min_value:
            continue

        # Match to our player DB
        fc_name = entry["name"].lower().strip()
        player = PLAYER_DB.get(fc_name)
        if player is None:
            # Try partial match
            matches = [p for k, p in PLAYER_DB.items() if fc_name in k or k in fc_name]
            if len(matches) == 1:
                player = matches[0]
            else:
                continue

        model_val = player.adjusted_value(league_format)
        delta = model_val - mkt_value
        if mkt_value == 0:
            continue
        delta_pct = delta / mkt_value * 100

        comparisons.append(
            MarketComparison(
                player=player,
                market_value=mkt_value,
                market_rank=entry.get("overall_rank") or 999,
                model_value=model_val,
                adp=entry.get("adp"),
                delta=delta,
                delta_pct=round(delta_pct, 1),
            )
        )

    undervalued = sorted(
        [c for c in comparisons if c.delta > 0], key=lambda c: c.delta, reverse=True
    )[:top_n]
    overvalued = sorted(
        [c for c in comparisons if c.delta < 0], key=lambda c: c.delta
    )[:top_n]

    return undervalued, overvalued


def print_market_analysis(
    num_qbs: int = 2,
    ppr: float = 1.0,
    top_n: int = 15,
    league_format: str = "2QB",
) -> None:
    print(f"\nFetching FantasyCalc dynasty values ({num_qbs}QB, {'PPR' if ppr == 1 else 'Half-PPR' if ppr == 0.5 else 'Standard'}, No TE Premium)...")
    undervalued, overvalued = find_undervalued_players(
        num_qbs=num_qbs, ppr=ppr, top_n=top_n, league_format=league_format
    )

    if not undervalued and not overvalued:
        print("  Could not retrieve FantasyCalc data (check network connection).")
        return

    def _row(c: MarketComparison) -> list:
        arrow = "▲" if c.delta > 0 else "▼"
        return [
            c.player.name,
            c.player.position,
            c.player.age,
            c.market_value,
            c.model_value,
            f"{arrow}{abs(c.delta_pct):.0f}%",
            f"#{c.market_rank}",
            f"ADP {c.adp:.1f}" if c.adp else "—",
        ]

    headers = ["Player", "Pos", "Age", "Mkt Val", "Model Val", "Δ%", "Rank", "ADP"]

    if undervalued:
        print("\n" + "═" * 70)
        print("  BUY TARGETS — Undervalued by Market")
        print("  (Model value significantly higher than FantasyCalc market consensus)")
        print("═" * 70)
        try:
            from tabulate import tabulate
            print(tabulate([_row(c) for c in undervalued], headers=headers, tablefmt="rounded_outline"))
        except ImportError:
            for c in undervalued:
                print(f"  {c.player.name:<25} {c.player.position:<3} Mkt:{c.market_value:>5}  Model:{c.model_value:>5}  Δ{c.delta_pct:+.0f}%")

    if overvalued:
        print("\n" + "═" * 70)
        print("  SELL TARGETS — Overvalued by Market")
        print("  (Model value significantly lower than FantasyCalc market consensus)")
        print("═" * 70)
        try:
            from tabulate import tabulate
            print(tabulate([_row(c) for c in overvalued], headers=headers, tablefmt="rounded_outline"))
        except ImportError:
            for c in overvalued:
                print(f"  {c.player.name:<25} {c.player.position:<3} Mkt:{c.market_value:>5}  Model:{c.model_value:>5}  Δ{c.delta_pct:+.0f}%")

    print("\n  Methodology: Model adjusts raw dynasty value by age curve,")
    print("  injury risk, and usage situation. Positive Δ = market sleeping on player.\n")


def print_full_market_rankings(num_qbs: int = 2, ppr: float = 1.0, top_n: int = 50,
                               league_format: str = "2QB") -> None:
    """Print top N players from FantasyCalc with our model value side-by-side."""
    print(f"\nFetching top {top_n} dynasty players from FantasyCalc...")
    fc_data = fetch_fantasycalc_values(num_qbs=num_qbs, ppr=ppr)
    if not fc_data:
        print("  Could not retrieve FantasyCalc data.")
        return

    rows = []
    for entry in sorted(fc_data, key=lambda e: e.get("overall_rank") or 9999)[:top_n]:
        if entry["position"] not in ("QB", "RB", "WR", "TE"):
            continue
        fc_name = entry["name"].lower().strip()
        player = PLAYER_DB.get(fc_name)
        if not player:
            matches = [p for k, p in PLAYER_DB.items() if fc_name in k or k in fc_name]
            player = matches[0] if len(matches) == 1 else None

        model_val = player.adjusted_value(league_format) if player else "—"
        age = player.age if player else (entry.get("age") or "?")
        adp = entry.get("adp")

        rows.append([
            entry.get("overall_rank", "?"),
            entry["name"],
            entry["position"],
            age,
            entry.get("value", 0),
            model_val,
            f"{adp:.1f}" if adp else "—",
        ])

    headers = ["Rank", "Player", "Pos", "Age", "FC Value", "Model Val", "ADP"]
    print("\n" + "═" * 70)
    print(f"  TOP {top_n} DYNASTY PLAYERS — FantasyCalc + Model Comparison")
    print("═" * 70)
    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    except ImportError:
        for r in rows:
            print("  " + "  ".join(str(x) for x in r))
    print()


# ---------------------------------------------------------------------------
# Interactive Mode
# ---------------------------------------------------------------------------

def run_interactive() -> None:
    print("\n" + "═" * 60)
    print("  DYNASTY TRADE ANALYZER — Interactive Mode")
    print("  Type 'quit' at any prompt to exit.")
    print("  Tip: Player names and picks like '2026 1st early' supported.")
    print("═" * 60)

    while True:
        print("\nOptions:")
        print("  1) Analyze a specific trade")
        print("  2) Analyze my roster & get recommendations")
        print("  3) Look up a player's value")
        print("  4) Market report — undervalued/overvalued vs FantasyCalc")
        print("  5) Top 50 dynasty rankings (FantasyCalc + model)")
        print("  6) Quit")
        choice = input("\nChoice [1-6]: ").strip()

        if choice in ("6", "quit", "q"):
            print("Good luck in your dynasty league!\n")
            break

        elif choice == "1":
            _interactive_trade()

        elif choice == "2":
            _interactive_roster()

        elif choice == "3":
            _interactive_lookup()

        elif choice == "4":
            fmt = input("  League format [1QB/2QB] (default 1QB): ").strip() or "1QB"
            num_qbs = 2 if fmt == "2QB" else 1
            ppr_raw = input("  Scoring [ppr/half/std] (default ppr): ").strip() or "ppr"
            ppr = {"ppr": 1.0, "half": 0.5, "std": 0.0}.get(ppr_raw, 1.0)
            print_market_analysis(num_qbs=num_qbs, ppr=ppr)

        elif choice == "5":
            fmt = input("  League format [1QB/2QB] (default 2QB): ").strip() or "2QB"
            num_qbs = 2 if fmt == "2QB" else 1
            ppr_raw = input("  Scoring [ppr/half/std] (default ppr): ").strip() or "ppr"
            ppr = {"ppr": 1.0, "half": 0.5, "std": 0.0}.get(ppr_raw, 1.0)
            print_full_market_rankings(num_qbs=num_qbs, ppr=ppr, league_format=fmt)


def _interactive_trade() -> None:
    print("\n  Enter assets for SIDE A (comma-separated).")
    print("  Example: CeeDee Lamb, 2026 2nd")
    raw_a = input("  Side A: ").strip()
    if raw_a.lower() == "quit":
        return

    print("  Enter assets for SIDE B (comma-separated).")
    raw_b = input("  Side B: ").strip()
    if raw_b.lower() == "quit":
        return

    label_a = input("  Label for Side A [Team A]: ").strip() or "Team A"
    label_b = input("  Label for Side B [Team B]: ").strip() or "Team B"

    side_a = TradeSide(label_a)
    side_b = TradeSide(label_b)

    unmatched = []
    for token in raw_a.split(","):
        asset = parse_asset(token)
        if asset:
            side_a.assets.append(asset)
        else:
            unmatched.append(token.strip())

    for token in raw_b.split(","):
        asset = parse_asset(token)
        if asset:
            side_b.assets.append(asset)
        else:
            unmatched.append(token.strip())

    if unmatched:
        print(f"\n  Warning: Could not find: {', '.join(unmatched)}")

    if not side_a.assets or not side_b.assets:
        print("  Error: Both sides must have at least one valid asset.\n")
        return

    print_trade_analysis(side_a, side_b, league_format="2QB")


def _interactive_roster() -> None:
    print("\n  Enter your roster (comma-separated player names).")
    print("  Example: Breece Hall, CeeDee Lamb, Trey McBride")
    raw = input("  Roster: ").strip()
    if raw.lower() == "quit":
        return

    fmt = input("  League format [1QB/2QB/SF] (default 2QB): ").strip() or "2QB"
    if fmt not in ROSTER_POSITIONS:
        fmt = "2QB"

    roster: list[Player] = []
    unmatched = []
    for token in raw.split(","):
        asset = parse_asset(token.strip())
        if asset and isinstance(asset.asset, Player):
            roster.append(asset.asset)
        else:
            unmatched.append(token.strip())

    if unmatched:
        print(f"  Warning: Could not find: {', '.join(unmatched)}")

    if not roster:
        print("  Error: No valid players found.\n")
        return

    analyzer = RosterAnalyzer()
    analysis = analyzer.analyze(roster, fmt, te_premium=False)
    print_roster_analysis(analysis, fmt)

    recommender = TradeRecommender()
    recs = recommender.recommend(roster, fmt, te_premium=False)
    print_recommendations(recs)


def _interactive_lookup() -> None:
    name = input("  Player name: ").strip()
    if name.lower() == "quit":
        return
    asset = parse_asset(name)
    if not asset or not isinstance(asset.asset, Player):
        print(f"  '{name}' not found in database.\n")
        return
    p = asset.asset
    peak_left, total_left = p.buy_window()
    print(f"\n  {p.name}  ({p.position}, {p.nfl_team})")
    print(f"    Age          : {p.age}")
    print(f"    Dynasty Rank : #{p.dynasty_rank} overall  /  #{p.positional_rank} at {p.position}")
    print(f"    Recent PPG   : {p.recent_ppg}")
    print(f"    Base Value   : {p.dynasty_value:,}")
    print(f"    Adj. Value   : {p.adjusted_value('2QB'):,}  (2QB/PPR, age/situation/injury adjusted)")
    print(f"    Age Mult     : {p.age_multiplier():.2f}")
    print(f"    Buy Window   : ~{peak_left} yrs in peak, {total_left} yrs total")
    print(f"    Injury Risk  : {p.injury_risk}")
    print(f"    Situation    : {p.situation}")
    if p.notes:
        print(f"    Notes        : {p.notes}")
    print()


# ---------------------------------------------------------------------------
# Direct Trade from CLI args
# ---------------------------------------------------------------------------

def analyze_trade_from_args(side_a_str: str, side_b_str: str,
                             label_a: str = "Team A", label_b: str = "Team B") -> None:
    side_a = TradeSide(label_a)
    side_b = TradeSide(label_b)

    unmatched = []
    for token in side_a_str.split(","):
        asset = parse_asset(token)
        if asset:
            side_a.assets.append(asset)
        else:
            unmatched.append(token.strip())

    for token in side_b_str.split(","):
        asset = parse_asset(token)
        if asset:
            side_b.assets.append(asset)
        else:
            unmatched.append(token.strip())

    if unmatched:
        print(f"Warning: Could not find: {', '.join(unmatched)}")

    if side_a.assets and side_b.assets:
        print_trade_analysis(side_a, side_b, league_format="2QB")
    else:
        print("Error: Both sides must have at least one valid asset.")


def analyze_roster_from_csv(csv_path: str, league_format: str = "2QB") -> None:
    """Load roster from a CSV with a 'player' column."""
    import csv
    roster: list[Player] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("player", row.get("name", "")).strip()
            asset = parse_asset(name)
            if asset and isinstance(asset.asset, Player):
                roster.append(asset.asset)

    if not roster:
        print(f"No recognizable players found in {csv_path}")
        return

    analyzer = RosterAnalyzer()
    analysis = analyzer.analyze(roster, league_format, te_premium=False)
    print_roster_analysis(analysis, league_format)

    recommender = TradeRecommender()
    recs = recommender.recommend(roster, league_format, te_premium=False)
    print_recommendations(recs)
