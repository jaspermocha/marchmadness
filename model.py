"""
model.py — Win-probability prediction model for March Madness matchups.

Architecture
------------
Uses a logistic regression trained on synthetic historical tournament data
(generated from known KenPom-era efficiency relationships) plus a
Pythagorean-expectation baseline.

The model predicts P(team_a beats team_b) given their combined feature
differences. You can also drop in your own historical matchup CSV to retrain.

Features used (differences: team_a − team_b unless noted)
----------------------------------------------------------
  - adj_em_diff       Adjusted efficiency margin diff
  - adj_o_diff        Adjusted offensive efficiency diff
  - adj_d_diff        Adjusted defensive efficiency diff (adj_d_a − adj_d_b)
  - barthag_diff      Power rating diff
  - efg_diff          eFG% offense diff
  - efg_d_diff        eFG% defense diff (opponent eFG, lower better)
  - to_diff           Turnover % diff (lower = better)
  - orb_diff          Offensive rebound % diff
  - tempo_diff        Tempo diff (absolute)
  - seed_diff         Seed diff (team_a seed − team_b seed)
  - sos_diff          Strength of schedule diff
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import Team


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "adj_em_diff",
    "adj_o_diff",
    "adj_d_diff",
    "barthag_diff",
    "efg_diff",
    "efg_d_diff",
    "to_diff",
    "orb_diff",
    "tempo_diff",
    "seed_diff",
    "sos_diff",
]


def build_features(team_a: Team, team_b: Team) -> np.ndarray:
    """Return a 1-D feature vector for the matchup (a vs b)."""
    return np.array(
        [
            team_a.adj_em - team_b.adj_em,
            team_a.adj_o - team_b.adj_o,
            # adj_d: lower is better, so flip sign so positive diff = a better defense
            team_b.adj_d - team_a.adj_d,
            team_a.barthag - team_b.barthag,
            team_a.efg_pct - team_b.efg_pct,
            # efg_d: lower is better (opponent eFG), flip sign
            team_b.efg_d_pct - team_a.efg_d_pct,
            # to_pct: lower is better for offense, so negative diff = a better
            team_b.to_pct - team_a.to_pct,
            team_a.orb_pct - team_b.orb_pct,
            team_a.adj_tempo - team_b.adj_tempo,
            team_b.seed - team_a.seed,   # positive = a is better seed
            team_a.sos - team_b.sos,
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Pythagorean baseline (no training required)
# ---------------------------------------------------------------------------

# Exponent tuned for college basketball (commonly cited ~11.5)
PYTH_EXP = 11.5


def pythagorean_win_prob(team_a: Team, team_b: Team) -> float:
    """
    Simple Pythagorean expectation using adjusted efficiencies.

    P(A wins) based on adj_em difference via logistic approximation.
    The 11 in the denominator converts a ±11 AdjEM diff to roughly 75% win prob,
    consistent with historical tournament data.
    """
    diff = team_a.adj_em - team_b.adj_em
    return float(expit(diff / 11.0))


# ---------------------------------------------------------------------------
# Logistic regression model
# ---------------------------------------------------------------------------

class MarchMadnessModel:
    """
    Logistic regression win-probability model.

    Training
    --------
    Call `train(matchups_df)` with a DataFrame of historical matchups, or
    call `train_synthetic()` to build a model from simulated data based on
    known efficiency relationships.

    Prediction
    ----------
    `predict(team_a, team_b)` → float probability that team_a wins.
    """

    MODEL_PATH = "model.pkl"

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, matchups_df: pd.DataFrame) -> None:
        """
        Train on a DataFrame with columns matching FEATURE_NAMES + 'label'
        where label=1 means team_a won.
        """
        X = matchups_df[FEATURE_NAMES].values
        y = matchups_df["label"].values
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._trained = True

    def train_synthetic(self, n_samples: int = 50_000, seed: int = 42) -> None:
        """
        Generate synthetic matchup data based on logistic relationships and
        train the model. This is a reasonable starting point before real data
        is available.

        The synthetic data assumes the ground truth is Pythagorean expectation
        with mild noise, which is consistent with published tournament research.
        """
        rng = np.random.default_rng(seed)

        # Sample realistic AdjEM values for D-I tournament teams
        adj_em_a = rng.normal(10, 10, n_samples)
        adj_em_b = rng.normal(10, 10, n_samples)
        adj_em_diff = adj_em_a - adj_em_b

        # Correlated sub-features from adj_em
        adj_o_diff = adj_em_diff * 0.55 + rng.normal(0, 3, n_samples)
        adj_d_diff = adj_em_diff * 0.45 + rng.normal(0, 3, n_samples)
        barthag_diff = adj_em_diff * 0.025 + rng.normal(0, 0.03, n_samples)
        efg_diff = adj_em_diff * 0.004 + rng.normal(0, 0.02, n_samples)
        efg_d_diff = adj_em_diff * 0.003 + rng.normal(0, 0.02, n_samples)
        to_diff = adj_em_diff * 0.002 + rng.normal(0, 0.01, n_samples)
        orb_diff = rng.normal(0, 0.05, n_samples)
        tempo_diff = rng.normal(0, 3, n_samples)
        seed_diff = -adj_em_diff * 0.5 + rng.normal(0, 2, n_samples)
        seed_diff = np.clip(seed_diff, -15, 15)
        sos_diff = adj_em_diff * 0.3 + rng.normal(0, 2, n_samples)

        X = np.column_stack(
            [
                adj_em_diff,
                adj_o_diff,
                adj_d_diff,
                barthag_diff,
                efg_diff,
                efg_d_diff,
                to_diff,
                orb_diff,
                tempo_diff,
                seed_diff,
                sos_diff,
            ]
        )

        # Ground truth: Pythagorean prob + noise
        true_prob = expit(adj_em_diff / 11.0)
        y = rng.binomial(1, true_prob).astype(int)

        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df["label"] = y
        self.train(df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = MODEL_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "clf": self.clf, "trained": self._trained}, f)

    def load(self, path: str = MODEL_PATH) -> None:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.scaler = obj["scaler"]
        self.clf = obj["clf"]
        self._trained = obj["trained"]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, team_a: Team, team_b: Team) -> float:
        """Return P(team_a beats team_b) in [0, 1]."""
        if not self._trained:
            # Fall back to Pythagorean if not trained
            return pythagorean_win_prob(team_a, team_b)
        x = build_features(team_a, team_b).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        return float(self.clf.predict_proba(x_scaled)[0, 1])

    def predict_matchup(
        self, team_a: Team, team_b: Team
    ) -> dict:
        """Return a rich matchup prediction dict."""
        prob_a = self.predict(team_a, team_b)
        prob_b = 1.0 - prob_a
        return {
            "team_a": team_a.team,
            "team_b": team_b.team,
            "prob_a": round(prob_a, 4),
            "prob_b": round(prob_b, 4),
            "favorite": team_a.team if prob_a >= 0.5 else team_b.team,
            "adj_em_diff": round(team_a.adj_em - team_b.adj_em, 2),
            "projected_spread": round(_prob_to_spread(prob_a), 1),
        }

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances (logistic regression coefficients)."""
        if not self._trained:
            raise RuntimeError("Model not trained yet.")
        coefs = self.clf.coef_[0]
        df = pd.DataFrame(
            {"feature": FEATURE_NAMES, "coefficient": coefs}
        ).sort_values("coefficient", ascending=False)
        return df


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _prob_to_spread(win_prob: float, k: float = 11.0) -> float:
    """
    Convert a win probability to an approximate point spread.
    Inverse of the logistic function used in pythagorean_win_prob.
    Positive spread = team_a favored by that many points.
    """
    p = np.clip(win_prob, 0.001, 0.999)
    return float(k * np.log(p / (1.0 - p)))


def get_or_train_model(model_path: str = MarchMadnessModel.MODEL_PATH) -> MarchMadnessModel:
    """Load a saved model or train a new synthetic one."""
    m = MarchMadnessModel()
    if os.path.exists(model_path):
        m.load(model_path)
    else:
        m.train_synthetic()
        m.save(model_path)
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from data import load_teams, generate_sample_teams_csv

    if not os.path.exists("teams.csv"):
        print("No teams.csv found — generating sample data...")
        generate_sample_teams_csv()

    teams = load_teams()
    model = get_or_train_model()

    print("\nModel feature importance:")
    print(model.feature_importance().to_string(index=False))

    if len(teams) >= 2:
        t1, t2 = teams[0], teams[1]
        result = model.predict_matchup(t1, t2)
        print(f"\nSample matchup: {result}")
