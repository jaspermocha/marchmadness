#!/usr/bin/env python3
"""
main.py — March Madness Betting Model CLI

Usage
-----
  # Generate sample team data (first-time setup)
  python main.py --setup

  # Run full analysis: simulate tournament + value bets
  python main.py

  # Simulate only
  python main.py --simulate --sims 20000

  # Print predicted bracket (deterministic)
  python main.py --bracket

  # Analyze specific game lines from a CSV
  python main.py --lines lines.csv

  # Show feature importance
  python main.py --importance

  # Estimate game total for a matchup
  python main.py --total "Duke" "Kansas"
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from tabulate import tabulate

from data import load_teams, generate_sample_teams_csv, teams_to_df
from model import get_or_train_model, pythagorean_win_prob
from bracket import Bracket, TournamentSimulator, print_simulation_results, print_bracket_predictions
from betting import (
    GameLine, analyze_lines, estimate_game_total,
    american_to_implied_prob, print_value_bets,
)


# ---------------------------------------------------------------------------
# Lines CSV loader
# ---------------------------------------------------------------------------

def load_lines_csv(path: str, teams_by_name: dict) -> list[GameLine]:
    """
    Load game lines from a CSV file.

    Expected columns:
      team_a, team_b, round, ml_a, ml_b, spread_a, total
    Optional:
      spread_juice_a, spread_juice_b, total_over_juice, total_under_juice
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    lines = []
    for _, row in df.iterrows():
        name_a = str(row["team_a"]).strip()
        name_b = str(row["team_b"]).strip()
        if name_a not in teams_by_name or name_b not in teams_by_name:
            print(f"  Warning: team not found in data — {name_a} or {name_b}, skipping.")
            continue

        def get_optional_int(col, default=None):
            if col in row.index and pd.notna(row[col]):
                return int(row[col])
            return default

        def get_optional_float(col, default=None):
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
            return default

        lines.append(
            GameLine(
                team_a=teams_by_name[name_a],
                team_b=teams_by_name[name_b],
                round_name=str(row.get("round", "Unknown")),
                ml_a=get_optional_int("ml_a"),
                ml_b=get_optional_int("ml_b"),
                spread_a=get_optional_float("spread_a"),
                spread_juice_a=get_optional_int("spread_juice_a", -110),
                spread_juice_b=get_optional_int("spread_juice_b", -110),
                total=get_optional_float("total"),
            )
        )
    return lines


# ---------------------------------------------------------------------------
# Sample lines generator
# ---------------------------------------------------------------------------

def generate_sample_lines_csv(teams: list, path: str = "lines.csv") -> None:
    """Write a sample lines.csv for the Round of 64 using first-seed matchups."""
    from data import load_teams
    import random
    random.seed(1)

    rows = []
    from bracket import REGIONS, SEED_MATCHUPS

    teams_by_region_seed = {}
    for t in teams:
        teams_by_region_seed[(t.region, t.seed)] = t

    for region in REGIONS:
        for s1, s2 in SEED_MATCHUPS:
            key1 = (region, s1)
            key2 = (region, s2)
            if key1 not in teams_by_region_seed or key2 not in teams_by_region_seed:
                continue
            ta = teams_by_region_seed[key1]
            tb = teams_by_region_seed[key2]

            # Synthetic realistic lines based on seed gap
            em_diff = ta.adj_em - tb.adj_em
            spread = -round(em_diff * 0.8 + random.uniform(-1.5, 1.5), 1)
            # Spread for ta (ta is usually better, so negative spread)
            spread = round(spread * 2) / 2  # round to nearest 0.5

            # Moneyline based on spread
            from model import _prob_to_spread
            rough_prob_a = 1 / (1 + pow(2.718, spread / 11.0))
            ml_a = int(-rough_prob_a / (1 - rough_prob_a) * 100) if rough_prob_a > 0.5 else int((1 - rough_prob_a) / rough_prob_a * 100)
            ml_b = int(-ml_a * 0.9) if ml_a < 0 else int(ml_a * -0.9)
            if rough_prob_a > 0.5:
                ml_a = -abs(ml_a)
                ml_b = abs(ml_b)
            else:
                ml_a = abs(ml_a)
                ml_b = -abs(ml_b)

            total = round(estimate_game_total(ta, tb) + random.uniform(-2, 2), 1)
            total = round(total * 2) / 2

            rows.append({
                "team_a": ta.team,
                "team_b": tb.team,
                "round": "Round of 64",
                "ml_a": ml_a,
                "ml_b": ml_b,
                "spread_a": spread,
                "total": total,
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Sample lines.csv written to {path} ({len(df)} games)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="March Madness Betting Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--setup", action="store_true", help="Generate sample teams.csv and lines.csv")
    parser.add_argument("--simulate", action="store_true", help="Run Monte Carlo tournament simulation")
    parser.add_argument("--sims", type=int, default=10_000, help="Number of simulations (default: 10000)")
    parser.add_argument("--bracket", action="store_true", help="Print deterministic bracket prediction")
    parser.add_argument("--lines", type=str, default=None, help="Path to lines CSV for value bet analysis")
    parser.add_argument("--importance", action="store_true", help="Show model feature importances")
    parser.add_argument("--total", nargs=2, metavar=("TEAM_A", "TEAM_B"), help="Estimate game total for two teams")
    parser.add_argument("--teams-csv", type=str, default="teams.csv", help="Path to teams CSV (default: teams.csv)")
    parser.add_argument("--edge", type=float, default=0.02, help="Minimum edge for value bets (default: 0.02 = 2%%)")
    parser.add_argument("--top", type=int, default=20, help="Number of teams to show in simulation results")
    parser.add_argument("--retrain", action="store_true", help="Force retrain the model")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    if args.setup:
        print("Generating sample data...")
        generate_sample_teams_csv(args.teams_csv)
        teams = load_teams(args.teams_csv)
        generate_sample_lines_csv(teams)
        print("\nSetup complete. Edit teams.csv with real data for live use.")
        return

    # ------------------------------------------------------------------
    # Load teams
    # ------------------------------------------------------------------
    if not os.path.exists(args.teams_csv):
        print(f"Error: {args.teams_csv} not found. Run `python main.py --setup` first.")
        sys.exit(1)

    print(f"Loading teams from {args.teams_csv}...")
    teams = load_teams(args.teams_csv)
    teams_by_name = {t.team: t for t in teams}
    print(f"  Loaded {len(teams)} teams across {len(set(t.region for t in teams))} regions.")

    # ------------------------------------------------------------------
    # Load / train model
    # ------------------------------------------------------------------
    if args.retrain and os.path.exists(MarchMadnessModel.MODEL_PATH if False else "model.pkl"):
        os.remove("model.pkl")

    from model import MarchMadnessModel
    print("Loading model...")
    model = get_or_train_model()
    print("  Model ready.")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    if args.importance:
        df = model.feature_importance()
        print("\nFeature Importances (logistic regression coefficients):")
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))

    # ------------------------------------------------------------------
    # Game total estimate
    # ------------------------------------------------------------------
    if args.total:
        name_a, name_b = args.total
        if name_a not in teams_by_name:
            print(f"Error: team '{name_a}' not found in {args.teams_csv}.")
            sys.exit(1)
        if name_b not in teams_by_name:
            print(f"Error: team '{name_b}' not found in {args.teams_csv}.")
            sys.exit(1)
        ta, tb = teams_by_name[name_a], teams_by_name[name_b]
        total = estimate_game_total(ta, tb)
        prob = model.predict(ta, tb)
        print(f"\nMatchup: ({ta.seed}) {ta.team} vs ({tb.seed}) {tb.team}")
        print(f"  Model win prob: {ta.team} {prob*100:.1f}%  |  {tb.team} {(1-prob)*100:.1f}%")
        print(f"  Estimated game total: {total}")
        return

    # ------------------------------------------------------------------
    # Build bracket
    # ------------------------------------------------------------------
    try:
        bracket = Bracket(teams)
    except ValueError as e:
        print(f"Bracket error: {e}")
        print("  Ensure teams.csv has exactly seeds 1-16 for each of East/West/South/Midwest.")
        sys.exit(1)

    sim = TournamentSimulator(bracket, model)

    # ------------------------------------------------------------------
    # Default: run everything unless specific flags are set
    # ------------------------------------------------------------------
    run_all = not (args.simulate or args.bracket or args.lines or args.importance or args.total)

    if args.simulate or run_all:
        print(f"\nRunning {args.sims:,} tournament simulations...")
        results = sim.simulate(n_sims=args.sims)
        print_simulation_results(results, top_n=args.top)

    if args.bracket or run_all:
        predictions = sim.predict_bracket()
        print_bracket_predictions(predictions)

    if args.lines or run_all:
        lines_path = args.lines or "lines.csv"
        if os.path.exists(lines_path):
            print(f"\nAnalyzing betting lines from {lines_path}...")
            lines = load_lines_csv(lines_path, teams_by_name)
            value_df = analyze_lines(lines, model, min_edge=args.edge)
            print_value_bets(value_df)
        else:
            if args.lines:
                print(f"Error: lines file not found: {lines_path}")
            else:
                print(f"\nNo lines.csv found. Run `python main.py --setup` to generate a sample,")
                print("or create lines.csv manually. See --help for the expected format.")


if __name__ == "__main__":
    main()
