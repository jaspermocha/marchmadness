#!/usr/bin/env python3
"""
main.py — March Madness Betting Model CLI

FULLY AUTOMATED MODE (recommended)
------------------------------------
  python main.py --auto

  Fetches live data automatically:
    • Team efficiency stats from Barttorvik (T-Rank)
    • Tournament bracket seedings from data.ncaa.com / ESPN
    • Live betting lines from The Odds API (requires free key in .env)

  Requires: ODDS_API_KEY=your_key in .env (see .env.example)
            Sign up free at https://the-odds-api.com

CONFERENCE TOURNAMENT MODE
---------------------------
  python main.py --conf                     # all conf tournaments (next 7 days)
  python main.py --conf --conf-name "ACC"   # specific conference
  python main.py --conf --days 14           # look 14 days ahead
  python main.py --conf --no-odds           # predictions only, no odds key needed

  Works exactly like --auto but for conference tournaments.
  Since conf tournaments are on neutral courts, the model applies directly.

  Supported --conf-name values:
    ACC, Big Ten, Big 12, SEC, Big East, Pac-12, A-10,
    Mountain West, WCC, American, MVC, MAC, and more.

MANUAL / SAMPLE MODE
---------------------
  python main.py --setup          # generate sample teams.csv + lines.csv
  python main.py                   # run with local CSV files

OTHER FLAGS
-----------
  --auto                   Fully automated: scrape stats + live odds
  --conf                   Conference tournament mode
  --conf-name "ACC"        Filter to a specific conference
  --days 7                 Days ahead to look for games (conf mode)
  --auto --no-odds         Auto stats only (no odds API needed)
  --simulate --sims 20000  Monte Carlo simulation
  --bracket                Deterministic bracket prediction
  --lines lines.csv        Analyze specific lines CSV
  --total "Duke" "Kansas"  Estimate game total for a matchup
  --importance             Show model feature importances
  --retrain                Force retrain the model
  --year 2026              Tournament year (default: current year)
  --edge 0.03              Minimum edge to flag a value bet (default: 2%)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import pandas as pd
from tabulate import tabulate

from data import load_teams, generate_sample_teams_csv, teams_to_df
from model import get_or_train_model, MarchMadnessModel
from bracket import Bracket, TournamentSimulator, print_simulation_results, print_bracket_predictions
from betting import (
    GameLine, analyze_lines, estimate_game_total, print_value_bets,
)


# ---------------------------------------------------------------------------
# Auto mode: live data fetching
# ---------------------------------------------------------------------------

def run_auto(
    args: argparse.Namespace,
    model: MarchMadnessModel,
) -> None:
    """Fetch live data and run the full analysis pipeline."""
    from scraper import fetch_tournament_teams

    year = args.year or date.today().year
    print(f"\n[auto] Fetching live data for {year} tournament...")

    # 1. Fetch teams
    teams = fetch_tournament_teams(year=year, verbose=True)

    if len(teams) < 32:
        print(
            f"\n[auto] WARNING: Only {len(teams)} teams found.\n"
            "  The bracket may not be announced yet, or data sources are unavailable.\n"
            "  Run `python main.py --setup` to use sample data.\n"
        )
        if len(teams) == 0:
            return

    teams_by_name = {t.team: t for t in teams}
    print(f"[auto] {len(teams)} tournament teams loaded.\n")

    # 2. Build bracket (requires exactly 64 teams with seeds 1-16 in 4 regions)
    bracket = None
    try:
        bracket = Bracket(teams)
    except ValueError as e:
        print(f"[auto] Bracket not complete yet: {e}")
        print("       Running analysis without full bracket simulation.\n")

    # 3. Fetch live odds (optional)
    lines: list[GameLine] = []
    if not args.no_odds:
        try:
            from odds_fetcher import fetch_game_lines
            lines = fetch_game_lines(teams_by_name, use_cache=True, verbose=True)
        except EnvironmentError as e:
            print(f"\n[odds] {e}\n")
            print("  Skipping live odds. Use --no-odds to suppress this message.\n")
        except Exception as e:
            print(f"[odds] Failed to fetch odds: {e}\n")

    # 4. Run analysis
    if bracket:
        sim = TournamentSimulator(bracket, model)

        sims = args.sims
        print(f"\nRunning {sims:,} tournament simulations...")
        results = sim.simulate(n_sims=sims)
        print_simulation_results(results, top_n=args.top)

        predictions = sim.predict_bracket()
        print_bracket_predictions(predictions)

    if lines:
        print(f"\nAnalyzing {len(lines)} games with live odds...")
        value_df = analyze_lines(lines, model, min_edge=args.edge)
        print_value_bets(value_df)
    elif not args.no_odds:
        print("\nNo live odds available to analyze.")


# ---------------------------------------------------------------------------
# CSV-based lines loader (manual mode)
# ---------------------------------------------------------------------------

def load_lines_csv(path: str, teams_by_name: dict) -> list[GameLine]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    lines = []
    for _, row in df.iterrows():
        name_a = str(row["team_a"]).strip()
        name_b = str(row["team_b"]).strip()
        if name_a not in teams_by_name or name_b not in teams_by_name:
            print(f"  Warning: team not found — {name_a!r} or {name_b!r}, skipping.")
            continue

        def opt_int(col, default=None):
            if col in row.index and pd.notna(row[col]):
                return int(row[col])
            return default

        def opt_float(col, default=None):
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
            return default

        lines.append(
            GameLine(
                team_a=teams_by_name[name_a],
                team_b=teams_by_name[name_b],
                round_name=str(row.get("round", "Unknown")),
                ml_a=opt_int("ml_a"),
                ml_b=opt_int("ml_b"),
                spread_a=opt_float("spread_a"),
                spread_juice_a=opt_int("spread_juice_a", -110),
                spread_juice_b=opt_int("spread_juice_b", -110),
                total=opt_float("total"),
            )
        )
    return lines


def generate_sample_lines_csv(teams: list, path: str = "lines.csv") -> None:
    import random
    random.seed(1)

    from bracket import REGIONS, SEED_MATCHUPS
    from model import _prob_to_spread

    teams_by_region_seed = {}
    for t in teams:
        teams_by_region_seed[(t.region, t.seed)] = t

    rows = []
    for region in REGIONS:
        for s1, s2 in SEED_MATCHUPS:
            ta = teams_by_region_seed.get((region, s1))
            tb = teams_by_region_seed.get((region, s2))
            if not ta or not tb:
                continue

            em_diff = ta.adj_em - tb.adj_em
            spread = -round(em_diff * 0.8 + random.uniform(-1.5, 1.5), 1)
            spread = round(spread * 2) / 2

            rough_prob_a = 1 / (1 + pow(2.718, spread / 11.0))
            if rough_prob_a > 0.5:
                ml_a = -abs(int(rough_prob_a / (1 - rough_prob_a) * 100))
                ml_b = abs(int((1 - rough_prob_a) / rough_prob_a * 100))
            else:
                ml_a = abs(int((1 - rough_prob_a) / rough_prob_a * 100))
                ml_b = -abs(int(rough_prob_a / (1 - rough_prob_a) * 100))

            total = round(estimate_game_total(ta, tb) + random.uniform(-2, 2), 1)
            total = round(total * 2) / 2

            rows.append({
                "team_a": ta.team, "team_b": tb.team,
                "round": "Round of 64",
                "ml_a": ml_a, "ml_b": ml_b,
                "spread_a": spread, "total": total,
            })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Sample lines.csv written to {path} ({len(rows)} games)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="March Madness Betting Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode flags
    parser.add_argument(
        "--auto", action="store_true",
        help="Fully automated: scrape Barttorvik + NCAA bracket + live odds",
    )
    parser.add_argument(
        "--conf", action="store_true",
        help="Conference tournament mode: fetch upcoming conf tourney games + predict",
    )
    parser.add_argument(
        "--conf-name", type=str, default=None, metavar="CONF",
        help="Filter conf tournament mode to a specific conference (e.g. 'ACC', 'SEC')",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Days ahead to look for conference tournament games (default: 7)",
    )
    parser.add_argument(
        "--no-odds", action="store_true",
        help="Skip odds fetching (no API key needed)",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Generate sample teams.csv and lines.csv",
    )

    # Analysis flags
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--bracket", action="store_true")
    parser.add_argument("--lines", type=str, default=None)
    parser.add_argument("--importance", action="store_true")
    parser.add_argument("--total", nargs=2, metavar=("TEAM_A", "TEAM_B"))
    parser.add_argument(
        "--backtest", action="store_true",
        help="Backtest model against completed games this season",
    )
    parser.add_argument(
        "--backtest-type", type=str, default="neutral",
        choices=["all", "neutral", "conf_tourney", "postseason"],
        metavar="TYPE",
        help="Game type to backtest: all | neutral | conf_tourney | postseason (default: neutral)",
    )
    parser.add_argument(
        "--time-machine", action="store_true",
        help="Use Barttorvik date snapshots for backtest (no lookahead bias, slower)",
    )

    # Options
    parser.add_argument("--teams-csv", type=str, default="teams.csv")
    parser.add_argument("--edge", type=float, default=0.02)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--year", type=int, default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup (sample data)
    # ------------------------------------------------------------------
    if args.setup:
        print("Generating sample data...")
        generate_sample_teams_csv(args.teams_csv)
        teams = load_teams(args.teams_csv)
        generate_sample_lines_csv(teams)
        print("\nSetup complete. Edit teams.csv with real data, or use --auto for live data.")
        return

    # ------------------------------------------------------------------
    # Load / train model
    # ------------------------------------------------------------------
    if args.retrain and os.path.exists("model.pkl"):
        os.remove("model.pkl")

    print("Loading model...")
    model = get_or_train_model()
    print("  Model ready.\n")

    # ------------------------------------------------------------------
    # BACKTEST MODE
    # ------------------------------------------------------------------
    if args.backtest:
        from backtest import (
            fetch_season_games, run_backtest, compute_metrics, print_backtest_report
        )
        from scraper import fetch_barttorvik_ratings

        year = args.year or date.today().year
        game_type = args.backtest_type
        use_tm = args.time_machine

        print(f"\nFetching {game_type} game results for {year} season...")
        if use_tm:
            print("  Using Barttorvik time-machine snapshots (no lookahead bias).")
        else:
            print("  Using current season stats (slight lookahead bias).")
            print("  Add --time-machine for cleaner results.\n")

        games_df = fetch_season_games(year=year, game_type=game_type)
        if games_df.empty:
            print(
                "No completed games found. The season may not have started yet, "
                "or the ESPN API returned no results."
            )
            return

        print(f"Fetching Barttorvik stats for {year}...")
        stats_df = fetch_barttorvik_ratings(year)

        results_df = run_backtest(
            games_df, model, stats_df, use_time_machine=use_tm
        )
        if results_df.empty:
            print("No games could be evaluated (team name matching failed).")
            return

        metrics = compute_metrics(results_df)
        print_backtest_report(results_df, metrics, game_type, use_tm)
        return

    # ------------------------------------------------------------------
    # CONFERENCE TOURNAMENT MODE
    # ------------------------------------------------------------------
    if args.conf:
        from conf_tournament import run_conf_analysis, CONF_GROUP_IDS
        conf_name = args.conf_name
        if conf_name and conf_name not in CONF_GROUP_IDS:
            # Case-insensitive lookup
            match = next(
                (k for k in CONF_GROUP_IDS if k.lower() == conf_name.lower()), None
            )
            if match:
                conf_name = match
            else:
                valid = ", ".join(sorted(CONF_GROUP_IDS.keys()))
                print(f"Unknown conference '{conf_name}'.\nValid options: {valid}")
                sys.exit(1)

        run_conf_analysis(
            model=model,
            conf_name=conf_name,
            days_ahead=args.days,
            min_edge=args.edge,
            fetch_odds=not args.no_odds,
        )
        return

    # ------------------------------------------------------------------
    # AUTO MODE (March Madness)
    # ------------------------------------------------------------------
    if args.auto:
        run_auto(args, model)
        return

    # ------------------------------------------------------------------
    # MANUAL MODE: load from CSV
    # ------------------------------------------------------------------
    if not os.path.exists(args.teams_csv):
        print(
            f"Error: {args.teams_csv} not found.\n"
            "  Run `python main.py --setup` for sample data, or\n"
            "  Run `python main.py --auto` for live data."
        )
        sys.exit(1)

    print(f"Loading teams from {args.teams_csv}...")
    teams = load_teams(args.teams_csv)
    teams_by_name = {t.team: t for t in teams}
    print(f"  {len(teams)} teams loaded.\n")

    # Feature importance
    if args.importance:
        df = model.feature_importance()
        print("Feature importances (logistic regression coefficients):")
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))

    # Game total estimate
    if args.total:
        name_a, name_b = args.total
        for name in [name_a, name_b]:
            if name not in teams_by_name:
                print(f"Error: team '{name}' not found.")
                sys.exit(1)
        ta, tb = teams_by_name[name_a], teams_by_name[name_b]
        total = estimate_game_total(ta, tb)
        prob = model.predict(ta, tb)
        print(f"\n({ta.seed}) {ta.team} vs ({tb.seed}) {tb.team}")
        print(f"  Model win prob: {ta.team} {prob*100:.1f}% | {tb.team} {(1-prob)*100:.1f}%")
        print(f"  Estimated game total: {total}")
        return

    # Build bracket
    try:
        bracket = Bracket(teams)
    except ValueError as e:
        print(f"Bracket error: {e}")
        print("  Ensure teams.csv has seeds 1-16 for each of East/West/South/Midwest.")
        sys.exit(1)

    sim = TournamentSimulator(bracket, model)

    run_all = not (args.simulate or args.bracket or args.lines or args.importance or args.total)

    if args.simulate or run_all:
        print(f"Running {args.sims:,} tournament simulations...")
        results = sim.simulate(n_sims=args.sims)
        print_simulation_results(results, top_n=args.top)

    if args.bracket or run_all:
        predictions = sim.predict_bracket()
        print_bracket_predictions(predictions)

    if args.lines or run_all:
        lines_path = args.lines or "lines.csv"
        if os.path.exists(lines_path):
            print(f"\nAnalyzing lines from {lines_path}...")
            lines = load_lines_csv(lines_path, teams_by_name)
            value_df = analyze_lines(lines, model, min_edge=args.edge)
            print_value_bets(value_df)
        elif args.lines:
            print(f"Error: {lines_path} not found.")
        else:
            print(
                "\nNo lines.csv found. Use `--auto` for live odds, or "
                "`--setup` to generate a sample."
            )


if __name__ == "__main__":
    main()
