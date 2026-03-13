# March Madness Betting Model

A statistically-grounded betting model for the NCAA Men's Basketball Tournament.
Built around adjusted efficiency metrics (KenPom-style), logistic regression,
Monte Carlo simulation, and expected-value betting analysis.

---

## Features

| Module | What it does |
|---|---|
| `data.py` | Load team stats from CSV; generate sample data |
| `model.py` | Logistic regression win-probability model trained on efficiency metrics |
| `bracket.py` | Full 64-team bracket structure + Monte Carlo simulator |
| `betting.py` | Moneyline & spread value analysis, EV, Kelly sizing |
| `main.py` | CLI entry point |

---

## Quick Start

```bash
pip install -r requirements.txt

# Generate sample data (68 fictional teams)
python main.py --setup

# Run full analysis (simulate + bracket + value bets)
python main.py

# Simulate 20,000 tournaments
python main.py --simulate --sims 20000

# Print deterministic bracket prediction
python main.py --bracket

# Analyze your own lines file
python main.py --lines my_lines.csv

# Estimate a game total
python main.py --total "Duke" "Kansas"

# Show model feature importances
python main.py --importance
```

---

## Input Data Format

### `teams.csv` — one row per tournament team

| Column | Description |
|---|---|
| `team` | Team name |
| `seed` | Tournament seed (1–16) |
| `region` | East / West / South / Midwest |
| `adj_o` | Adjusted offensive efficiency (pts per 100 poss) |
| `adj_d` | Adjusted defensive efficiency (pts allowed per 100 poss) |
| `adj_tempo` | Adjusted tempo (possessions per 40 min) |
| `barthag` | Power rating (prob of beating avg D-I team) |
| `wab` | Wins above bubble |
| `efg_pct` | Effective field goal % |
| `efg_d_pct` | Opponent effective FG% (lower = better defense) |
| `to_pct` | Turnover rate (lower = better offense) |
| `to_d_pct` | Opponent turnover rate (higher = better defense) |
| `orb_pct` | Offensive rebound % |
| `drb_pct` | Defensive rebound % |
| `ft_rate` | Free throw rate (FTA/FGA) |
| `ft_rate_d` | Opponent free throw rate (lower = better) |
| `wins` | Season wins |
| `losses` | Season losses |
| `sos` | Strength of schedule |
| `conf` | Conference (optional) |

**Data sources:** KenPom.com, Barttorvik.com (T-Rank), Sports-Reference.com

### `lines.csv` — one row per game

| Column | Description |
|---|---|
| `team_a` | Home/favored team name |
| `team_b` | Away/underdog team name |
| `round` | Round name |
| `ml_a` | American moneyline for team_a (e.g. -250) |
| `ml_b` | American moneyline for team_b (e.g. +210) |
| `spread_a` | Spread for team_a (e.g. -6.5 means team_a -6.5) |
| `total` | Over/under total |

---

## Model Architecture

### Win Probability
A logistic regression is trained on synthetic historical matchup data generated
to match Pythagorean efficiency relationships observed over the KenPom era.
Features include adjusted efficiency margin, four-factor differentials, seed
differential, and strength of schedule.

The Pythagorean baseline (no training required) uses:
```
P(A wins) = sigmoid(AdjEM_diff / 11)
```

### Betting Value
For each posted line:
1. Convert odds to implied probability (vig-removed)
2. Compare to model win probability
3. Calculate Expected Value: `EV = p * profit − (1−p) * stake`
4. Size bet using ¼-Kelly criterion

### Tournament Simulation
Monte Carlo: each game is a Bernoulli trial with p = model win probability.
Run 10,000+ simulations to get robust advancement percentages.

---

## Disclaimer

This model is for **educational and research purposes only**.
- Past model performance does not guarantee future results.
- Sports betting involves substantial financial risk.
- Ensure betting is legal in your jurisdiction before wagering.
- Always bet responsibly.
