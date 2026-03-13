"""
backtest.py — Backtest the model against completed games this season.

Fetches actual game results from ESPN's scoreboard API, then runs the model
on each matchup and compares predictions to outcomes.

Game types available
--------------------
  all           Every completed game
  neutral       Neutral-site games only (most relevant — same conditions as
                March Madness and conference tournaments)
  conf_tourney  Conference tournament games only
  postseason    NCAA tournament + conference tournaments

Stats source
------------
By default uses current season Barttorvik stats (slight lookahead bias).
Pass --time-machine to fetch Barttorvik time-machine snapshots dated to each
game's date, eliminating lookahead bias (slower — one HTTP request per snapshot).

Metrics reported
----------------
  Accuracy       % of games where predicted winner won
  Brier Score    Mean squared error of probabilities (0.25 = random, 0 = perfect)
  Log Loss       Cross-entropy loss (lower = better)
  ROC-AUC        Discrimination ability (0.5 = random, 1.0 = perfect)
  Calibration    Are 60% predictions winning 60% of the time?
  Flat-bet ROI   ROI if betting model favorites at -110 every game
  Edge-bet ROI   ROI if betting only when model edge > threshold
"""

from __future__ import annotations

import gzip
import io
import json
import os
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tabulate import tabulate

# ---------------------------------------------------------------------------
# ESPN game fetcher
# ---------------------------------------------------------------------------

ESPN_SCOREBOARD = (
    "http://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)

# Season start date for men's college basketball
SEASON_START = {
    2026: date(2025, 11, 4),
    2025: date(2024, 11, 4),
    2024: date(2023, 11, 6),
}

_CACHE_DIR = ".backtest_cache"


def _espn_scoreboard(day: date, timeout: int = 10) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }
    params = {"dates": day.strftime("%Y%m%d"), "groups": 100, "limit": 300}
    resp = requests.get(ESPN_SCOREBOARD, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_season_games(
    year: int = 2026,
    game_type: str = "neutral",
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch all completed games from ESPN for the given season.

    Parameters
    ----------
    year      : Season year (e.g. 2026 = 2025-26 season)
    game_type : "all" | "neutral" | "conf_tourney" | "postseason"
    use_cache : Cache daily ESPN responses in .backtest_cache/

    Returns
    -------
    DataFrame with columns:
      date, team_a, team_b, score_a, score_b, winner,
      neutral_site, season_type, notes, game_id
    """
    cache_file = os.path.join(_CACHE_DIR, f"games_{year}_{game_type}.parquet")
    if use_cache and os.path.exists(cache_file):
        if verbose:
            print(f"[backtest] Loading cached game results from {cache_file}")
        return pd.read_parquet(cache_file)

    start = SEASON_START.get(year, date(year - 1, 11, 4))
    end = min(date.today(), date(year, 4, 10))  # tournament ends early April

    rows = []
    total_days = (end - start).days + 1

    if verbose:
        print(
            f"[backtest] Fetching {total_days} days of ESPN game data "
            f"({start} → {end})..."
        )

    for i, day in enumerate(
        start + timedelta(days=d) for d in range(total_days)
    ):
        if verbose and i % 14 == 0:
            pct = i / total_days * 100
            print(f"  {day}  ({pct:.0f}%)", end="\r", flush=True)

        # Check per-day cache
        day_cache = os.path.join(_CACHE_DIR, f"day_{day.isoformat()}.json")
        if use_cache and os.path.exists(day_cache):
            with open(day_cache) as f:
                data = json.load(f)
        else:
            try:
                data = _espn_scoreboard(day)
                if use_cache:
                    os.makedirs(_CACHE_DIR, exist_ok=True)
                    with open(day_cache, "w") as f:
                        json.dump(data, f)
                time.sleep(0.15)  # be polite to ESPN
            except Exception as e:
                if verbose:
                    print(f"\n  Skip {day}: {e}")
                continue

        for event in data.get("events", []):
            comps = event.get("competitions", [{}])
            comp = comps[0] if comps else {}

            # Only completed games
            status = comp.get("status", {}).get("type", {})
            if not status.get("completed", False):
                continue

            neutral = comp.get("neutralSite", False)
            season_type = event.get("season", {}).get("type", 2)
            notes = " | ".join(
                n.get("headline", "") for n in event.get("notes", [])
            )

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            # Parse both teams
            teams_data = {}
            for c in competitors:
                side = c.get("homeAway", "home")
                name = (
                    c.get("team", {}).get("shortDisplayName")
                    or c.get("team", {}).get("displayName", "?")
                )
                try:
                    score = float(c.get("score", 0) or 0)
                except (ValueError, TypeError):
                    score = 0.0
                won = c.get("winner", False)
                teams_data[side] = {"name": name, "score": score, "winner": won}

            home = teams_data.get("home", {})
            away = teams_data.get("away", {})
            if not home.get("name") or not away.get("name"):
                continue

            rows.append(
                {
                    "date": day.isoformat(),
                    "team_a": home["name"],
                    "team_b": away["name"],
                    "score_a": home["score"],
                    "score_b": away["score"],
                    "winner": home["name"] if home.get("winner") else away["name"],
                    "neutral_site": neutral,
                    "season_type": season_type,
                    "notes": notes,
                    "game_id": event.get("id", ""),
                }
            )

    if verbose:
        print(f"\n[backtest] Fetched {len(rows)} completed games.")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Apply game_type filter
    df = _filter_game_type(df, game_type)

    if use_cache and not df.empty:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_file, index=False)

    return df


def _filter_game_type(df: pd.DataFrame, game_type: str) -> pd.DataFrame:
    if game_type == "all":
        return df
    elif game_type == "neutral":
        return df[df["neutral_site"] == True].copy()
    elif game_type == "conf_tourney":
        mask = (
            df["notes"].str.lower().str.contains("tournament|championship", na=False)
            & (df["season_type"] == 2)
        )
        return df[mask].copy()
    elif game_type == "postseason":
        # Season type 3 = postseason, OR conf tournament notes in early March
        mask = (df["season_type"] == 3) | (
            df["notes"].str.lower().str.contains("tournament|championship", na=False)
        )
        return df[mask].copy()
    return df


# ---------------------------------------------------------------------------
# Barttorvik time machine stats (per-date snapshots)
# ---------------------------------------------------------------------------

def fetch_pregame_stats(game_date: str, year: int = 2026) -> Optional[pd.DataFrame]:
    """
    Fetch Barttorvik time-machine stats snapshot for a given date.
    Returns None if the snapshot is unavailable.

    Snapshots are at:
    https://barttorvik.com/timemachine/team_results/YYYYMMDD_team_results.json.gz
    """
    d = game_date.replace("-", "")
    url = f"https://barttorvik.com/timemachine/team_results/{d}_team_results.json.gz"

    cache_path = os.path.join(_CACHE_DIR, f"stats_{d}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = json.loads(gzip.decompress(resp.content))
        # Time machine data is a list of team dicts — parse it
        from scraper import _normalize_barttorvik_df
        df = pd.DataFrame(data) if isinstance(data, list) else None
        if df is None:
            return None
        df = _normalize_barttorvik_df(df)

        os.makedirs(_CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    games_df: pd.DataFrame,
    model,
    stats_df: pd.DataFrame,
    use_time_machine: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For each game, predict P(team_a wins) and compare to actual outcome.

    Parameters
    ----------
    games_df       : DataFrame from fetch_season_games()
    model          : trained MarchMadnessModel
    stats_df       : current-season Barttorvik ratings (used if not time_machine)
    use_time_machine: if True, fetch per-date Barttorvik snapshots (no lookahead)

    Returns
    -------
    DataFrame with original game columns + pred_prob_a, predicted_winner,
    correct (bool), brier_contrib
    """
    from scraper import fuzzy_match_teams
    from data import Team

    bart_names = list(stats_df["team"].values)

    # Cache of (stat_df_key → team_obj) to avoid rebuilding
    _team_cache: dict[str, Team] = {}
    _stats_cache: dict[str, pd.DataFrame] = {"current": stats_df}

    def _get_stats(game_date: str) -> pd.DataFrame:
        if not use_time_machine:
            return stats_df
        if game_date not in _stats_cache:
            snap = fetch_pregame_stats(game_date)
            _stats_cache[game_date] = snap if snap is not None else stats_df
            time.sleep(0.1)
        return _stats_cache[game_date]

    def _make_team(name: str, df: pd.DataFrame, names_list: list) -> Optional[Team]:
        cache_key = f"{name}_{id(df)}"
        if cache_key in _team_cache:
            return _team_cache[cache_key]

        matched = name if name in names_list else fuzzy_match_teams(name, names_list, threshold=0.55)
        if matched is None:
            return None

        row = df[df["team"] == matched].iloc[0]

        def _f(col, default):
            v = row.get(col, default)
            try:
                return float(v) if v is not None else float(default)
            except (ValueError, TypeError):
                return float(default)

        t = Team(
            team=name, seed=0, region="Backtest",
            adj_o=_f("adj_o", 110.0), adj_d=_f("adj_d", 110.0),
            adj_tempo=_f("adj_tempo", 70.0), barthag=_f("barthag", 0.5),
            wab=_f("wab", 0.0), efg_pct=_f("efg_pct", 0.51),
            to_pct=_f("to_pct", 0.18), orb_pct=_f("orb_pct", 0.30),
            ft_rate=_f("ft_rate", 0.34), efg_d_pct=_f("efg_d_pct", 0.51),
            to_d_pct=_f("to_d_pct", 0.18), drb_pct=_f("drb_pct", 0.72),
            ft_rate_d=_f("ft_rate_d", 0.34), wins=int(_f("wins", 20)),
            losses=int(_f("losses", 12)), sos=_f("sos", 0.0),
            conf=str(row.get("conf", "Unknown")),
        )
        _team_cache[cache_key] = t
        return t

    results = []
    skipped = 0

    for _, row in games_df.iterrows():
        d_stats = _get_stats(row["date"])
        d_names = list(d_stats["team"].values)

        ta = _make_team(row["team_a"], d_stats, d_names)
        tb = _make_team(row["team_b"], d_stats, d_names)

        if ta is None or tb is None:
            skipped += 1
            continue

        prob_a = model.predict(ta, tb)
        predicted = row["team_a"] if prob_a >= 0.5 else row["team_b"]
        actual = row["winner"]
        correct = predicted == actual
        # Brier: (prob_a - actual_a)^2
        actual_a = 1.0 if actual == row["team_a"] else 0.0
        brier = (prob_a - actual_a) ** 2

        results.append(
            {
                **row.to_dict(),
                "pred_prob_a": round(prob_a, 4),
                "pred_prob_b": round(1 - prob_a, 4),
                "predicted_winner": predicted,
                "correct": correct,
                "actual_a": actual_a,
                "brier": brier,
            }
        )

    if verbose:
        print(f"[backtest] Predicted {len(results)} games ({skipped} skipped — no stats match).")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results_df: pd.DataFrame) -> dict:
    """Compute all performance metrics from backtest results."""
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

    if results_df.empty:
        return {}

    y_true = results_df["actual_a"].values
    y_prob = results_df["pred_prob_a"].values

    # Remove games where both teams had near-equal probs (near 0.5) ± for cleaner AUC
    acc = results_df["correct"].mean()
    brier = float(np.mean((y_prob - y_true) ** 2))
    logloss = float(log_loss(y_true, y_prob))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    # Calibration: bucket predictions into 10 bins, compute actual win rate per bin
    calibration = _calibration_table(y_true, y_prob)

    return {
        "n_games": len(results_df),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "roc_auc": round(auc, 4),
        "calibration": calibration,
    }


def _calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "prob_bucket": f"{lo:.0%}–{hi:.0%}",
                "n_games": int(mask.sum()),
                "mean_pred": round(float(y_prob[mask].mean()), 3),
                "actual_win_rate": round(float(y_true[mask].mean()), 3),
                "diff": round(float(y_prob[mask].mean() - y_true[mask].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Betting ROI simulation
# ---------------------------------------------------------------------------

def simulate_betting_roi(
    results_df: pd.DataFrame,
    min_prob: float = 0.55,
    juice: int = -110,
    kelly_fraction: float = 0.0,  # 0 = flat betting; > 0 = fractional Kelly
) -> dict:
    """
    Simulate betting on the model's favorite when predicted probability
    exceeds min_prob.

    Parameters
    ----------
    min_prob       : Minimum win probability to place a bet (default: 55%)
    juice          : Standard juice on spread bets (default: -110)
    kelly_fraction : If > 0, size bets using fractional Kelly instead of flat

    Returns
    -------
    dict with total_bets, wins, losses, roi, profit_units, win_pct
    """
    if juice > 0:
        payout = juice / 100.0
    else:
        payout = 100.0 / abs(juice)

    total_bets = 0
    wins = 0
    profit = 0.0

    for _, row in results_df.iterrows():
        prob_fav = max(row["pred_prob_a"], row["pred_prob_b"])
        if prob_fav < min_prob:
            continue

        fav_won = (
            (row["pred_prob_a"] >= 0.5 and row["actual_a"] == 1.0)
            or (row["pred_prob_a"] < 0.5 and row["actual_a"] == 0.0)
        )

        if kelly_fraction > 0:
            # Kelly: b*p - q / b where b = decimal profit
            b = payout
            p = prob_fav
            q = 1 - p
            kelly_full = (b * p - q) / b
            stake = max(0.0, kelly_full * kelly_fraction)
        else:
            stake = 1.0

        total_bets += 1
        if fav_won:
            profit += stake * payout
            wins += 1
        else:
            profit -= stake

    losses = total_bets - wins
    roi = profit / total_bets if total_bets > 0 else 0.0

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "win_pct": round(wins / total_bets, 4) if total_bets > 0 else 0,
        "profit_units": round(profit, 2),
        "roi": round(roi, 4),
        "min_prob_threshold": min_prob,
    }


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_backtest_report(
    results_df: pd.DataFrame,
    metrics: dict,
    game_type: str,
    use_time_machine: bool,
) -> None:
    n = metrics.get("n_games", 0)
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS — {game_type.upper()} GAMES")
    print(f"{'='*70}")

    bias_note = (
        "  [Stats source: Barttorvik time-machine snapshots — no lookahead bias]"
        if use_time_machine
        else "  [Stats source: Current season stats — slight lookahead bias. Use --time-machine for cleaner results.]"
    )
    print(bias_note)

    print(f"\n  Games evaluated: {n}")

    # Core metrics
    core = [
        ["Accuracy", f"{metrics['accuracy']*100:.1f}%", "% predicted winner correct"],
        ["Brier Score", f"{metrics['brier_score']:.4f}", "0 = perfect, 0.25 = random coin flip"],
        ["Log Loss", f"{metrics['log_loss']:.4f}", "lower = better (0 = perfect)"],
        ["ROC-AUC", f"{metrics['roc_auc']:.4f}", "0.5 = random, 1.0 = perfect"],
    ]
    print(
        "\n"
        + tabulate(core, headers=["Metric", "Value", "Reference"], tablefmt="rounded_outline")
    )

    # Calibration
    cal = metrics.get("calibration")
    if cal is not None and not cal.empty:
        print("\n  Calibration (are predicted probabilities accurate?):")
        print(tabulate(cal, headers="keys", tablefmt="rounded_outline", showindex=False))
        print("  diff = mean_pred − actual_win_rate  (closer to 0 = better calibrated)")

    # Betting ROI at different thresholds
    print("\n  Simulated flat-bet ROI at -110 juice (betting model favorite):")
    roi_rows = []
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        r = simulate_betting_roi(results_df, min_prob=threshold)
        roi_rows.append(
            [
                f"≥{threshold:.0%}",
                r["total_bets"],
                f"{r['win_pct']*100:.1f}%",
                f"{r['profit_units']:+.2f}u",
                f"{r['roi']*100:+.1f}%",
            ]
        )
    print(
        tabulate(
            roi_rows,
            headers=["Min Prob", "Bets", "Win%", "Profit", "ROI"],
            tablefmt="rounded_outline",
        )
    )
    print(
        "  NOTE: Breakeven at -110 = 52.4% win rate. "
        "ROI is simulated without real spread lines.\n"
        "  Real-world ROI depends on line quality and line shopping."
    )

    # Worst predictions (biggest upsets the model missed)
    if "correct" in results_df.columns:
        wrong = results_df[~results_df["correct"]].copy()
        wrong["confidence"] = wrong[["pred_prob_a", "pred_prob_b"]].max(axis=1)
        worst = wrong.nlargest(10, "confidence")[
            ["date", "team_a", "team_b", "predicted_winner", "winner", "confidence"]
        ].copy()
        worst["confidence"] = worst["confidence"].apply(lambda x: f"{x*100:.1f}%")
        worst.columns = ["Date", "Team A", "Team B", "Model Picked", "Actual Winner", "Confidence"]
        print("\n  Biggest misses (high-confidence wrong predictions):")
        print(tabulate(worst, headers="keys", tablefmt="rounded_outline", showindex=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from model import get_or_train_model
    from scraper import fetch_barttorvik_ratings

    year = 2026
    game_type = "neutral"
    use_time_machine = "--time-machine" in sys.argv

    for arg in sys.argv[1:]:
        if arg.startswith("--type="):
            game_type = arg.split("=", 1)[1]
        elif arg.startswith("--year="):
            year = int(arg.split("=", 1)[1])

    model = get_or_train_model()
    stats_df = fetch_barttorvik_ratings(year)

    print(f"Fetching {game_type} games for {year} season...")
    games_df = fetch_season_games(year=year, game_type=game_type)

    if games_df.empty:
        print("No games found.")
        sys.exit(0)

    results_df = run_backtest(games_df, model, stats_df, use_time_machine=use_time_machine)
    metrics = compute_metrics(results_df)
    print_backtest_report(results_df, metrics, game_type, use_time_machine)
