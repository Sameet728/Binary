"""
ml_models.py — Three ML Modules
  Model 1 — TradePredictor:    RandomForest predicts P(win) per candle
  Model 2 — StrategyOptimizer: GradientBoosting predicts which strategies pass filter
  Model 3 — SelfLearner:       Bayesian-style threshold advisor per condition name
"""

import copy
import warnings
from collections import defaultdict
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Feature columns used by TradePredictor (must exist in the feature DataFrame)
FEATURE_COLS = [
    "rsi", "ema9_vs_ema21", "bb_pct", "bb_width", "atr_pct",
    "adx", "di_plus", "di_minus", "body_pct", "wick_ratio",
    "trend_up", "trend_strong", "is_bullish",
    "bullish_engulf", "bearish_engulf", "bull_pin", "bear_pin",
    "inside_bar", "breakout_high", "breakout_low",
    "close_vs_ema9", "close_vs_ema21",
]


def _safe_float(v) -> float:
    try:
        f = float(v)
        return 0.0 if np.isnan(f) else f
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# MODEL 1 — TradePredictor
# ---------------------------------------------------------------------------

class TradePredictor:
    """
    Predict the probability that the *next* candle closes higher than the
    current candle — used as a supplementary confidence filter.
    Input : row of indicator values
    Output: float probability in [0, 1]
    """

    def __init__(self):
        self.model   = RandomForestClassifier(
            n_estimators=80, max_depth=5, min_samples_leaf=5,
            n_jobs=-1, random_state=42
        )
        self.scaler  = StandardScaler()
        self.trained = False
        self.accuracy: float = 0.5

    def _build_dataset(self, df: pd.DataFrame):
        X, y = [], []
        for i in range(len(df) - 1):
            row      = df.iloc[i]
            next_row = df.iloc[i + 1]
            X.append([_safe_float(row.get(c, 0) if hasattr(row, "get") else getattr(row, c, 0))
                      for c in FEATURE_COLS])
            y.append(1 if next_row["close"] > row["close"] else 0)
        return np.array(X, dtype=float), np.array(y, dtype=int)

    def train(self, df: pd.DataFrame) -> float:
        """Train on historical data; returns CV accuracy."""
        X, y = self._build_dataset(df)
        if len(X) < 60 or len(set(y)) < 2:
            return 0.5
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.trained = True
        cv_acc = cross_val_score(self.model, Xs, y, cv=3, scoring="accuracy").mean()
        self.accuracy = float(cv_acc)
        return self.accuracy

    def predict_proba(self, feature_row: list) -> float:
        """Return P(bullish next candle) for a single row."""
        if not self.trained:
            return 0.5
        X  = np.array(feature_row, dtype=float).reshape(1, -1)
        Xs = self.scaler.transform(X)
        p  = self.model.predict_proba(Xs)[0]
        return float(p[1]) if len(p) > 1 else 0.5

    def row_to_features(self, row: pd.Series) -> list:
        return [_safe_float(row.get(c, 0) if hasattr(row, "get") else getattr(row, c, 0))
                for c in FEATURE_COLS]


# ---------------------------------------------------------------------------
# MODEL 2 — StrategyOptimizer
# ---------------------------------------------------------------------------

class StrategyOptimizer:
    """
    Given a strategy's structural features, predict whether it will achieve
    win_rate >= 0.60 when backtested.  Used to bias random generation toward
    promising archetypes.
    """

    def __init__(self):
        self.model   = GradientBoostingClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.scaler  = StandardScaler()
        self.trained = False
        self.accuracy: float = 0.5

    def _strategy_features(self, strategy: dict) -> list:
        cc = strategy.get("call_conditions", [])
        pc = strategy.get("put_conditions",  [])
        fc = strategy.get("filter_conditions", [])
        all_names = {c["name"] for c in cc + pc}

        return [
            len(cc),
            len(pc),
            len(fc),
            strategy.get("generation", 0),
            strategy.get("stake_pct", 0.02),
            # RSI present?
            int(any("rsi" in n for n in all_names)),
            # Engulfing?
            int(any("engulf" in n for n in all_names)),
            # Bollinger?
            int(any("bb_" in n for n in all_names)),
            # ADX/DI?
            int(any("adx" in n or "di_" in n for n in all_names)),
            # Pin bar?
            int(any("pin" in n for n in all_names)),
            # Breakout?
            int(any("breakout" in n for n in all_names)),
            # EMA spread?
            int(any("ema9_vs" in n for n in all_names)),
        ]

    def train(self, strategies: list) -> float:
        """Train on a list of already-backtested strategy dicts."""
        X, y = [], []
        for s in strategies:
            m = s.get("metrics", {})
            if not m:
                continue
            X.append(self._strategy_features(s))
            y.append(1 if m.get("win_rate", 0) >= 0.60 else 0)

        if len(X) < 20 or len(set(y)) < 2:
            return 0.5
        Xs = self.scaler.fit_transform(np.array(X, dtype=float))
        self.model.fit(Xs, y)
        self.trained = True
        cv_acc = cross_val_score(self.model, Xs, y, cv=3).mean()
        self.accuracy = float(cv_acc)
        return self.accuracy

    def predict_quality(self, strategy: dict) -> float:
        """Return P(win_rate >= 0.60) for an untested strategy."""
        if not self.trained:
            return 0.5
        X  = np.array(self._strategy_features(strategy), dtype=float).reshape(1, -1)
        Xs = self.scaler.transform(X)
        p  = self.model.predict_proba(Xs)[0]
        return float(p[1]) if len(p) > 1 else 0.5


# ---------------------------------------------------------------------------
# MODEL 3 — SelfLearner
# ---------------------------------------------------------------------------

class SelfLearner:
    """
    Maintains a rolling history of (condition_name, threshold) → win_rate pairs.
    When asked to refine a strategy, it nudges each threshold toward values that
    historically correlate with higher win-rates (weighted average).
    """

    MAX_HISTORY = 2_000

    def __init__(self):
        # condition_name → list of (threshold, win_rate)
        self._history: Dict[str, List] = defaultdict(list)

    def record(self, strategy: dict, metrics: Optional[dict]) -> None:
        """Ingest results from one backtested strategy."""
        if not metrics:
            return
        wr = float(metrics.get("win_rate", 0))
        
        score = float(strategy.get("score", 0.0))
        stake = float(strategy.get("stake_pct", 0.02))
        if not hasattr(self, "_stake_history"):
            self._stake_history = []
        self._stake_history.append((stake, score))
        if len(self._stake_history) > self.MAX_HISTORY:
            self._stake_history = self._stake_history[-self.MAX_HISTORY:]

        all_conds = (
            strategy.get("call_conditions", []) +
            strategy.get("put_conditions",  []) +
            strategy.get("filter_conditions", [])
        )
        for cond in all_conds:
            name = cond["name"]
            thr  = cond.get("threshold")
            if thr is not None:
                self._history[name].append((float(thr), wr))
                # Trim
                if len(self._history[name]) > self.MAX_HISTORY:
                    self._history[name] = self._history[name][-self.MAX_HISTORY:]

    def suggest_stake_pct(self, current: float) -> float:
        """
        Suggest optimal stake_pct based on historical top-performing strategies.
        Blends 70% current / 30% suggestion to avoid sudden jumps.
        """
        if not hasattr(self, "_stake_history") or len(self._stake_history) < 8:
            return current

        arr = np.array(self._stake_history, dtype=float)
        cutoff = np.percentile(arr[:, 1], 66)  # top tercile by score
        good = arr[arr[:, 1] >= cutoff]

        if len(good) == 0:
            return current

        weights = good[:, 1]
        suggested = float(np.average(good[:, 0], weights=weights))
        blended = round(0.70 * current + 0.30 * suggested, 4)
        return float(np.clip(blended, 0.005, 0.10))

    def suggest_threshold(self, condition_name: str, current: float) -> float:
        """
        Weighted-mean threshold over the top-tercile performing samples.
        Blends 70 % current / 30 % suggestion to avoid over-fitting.
        """
        hist = self._history.get(condition_name, [])
        if len(hist) < 8:
            return current

        arr = np.array(hist, dtype=float)          # shape (N, 2): [threshold, wr]
        cutoff = np.percentile(arr[:, 1], 66)       # top tercile by win-rate
        good   = arr[arr[:, 1] >= cutoff]

        if len(good) == 0:
            return current

        weights    = good[:, 1]                    # use win-rate as weight
        suggested  = float(np.average(good[:, 0], weights=weights))
        blended    = round(0.70 * current + 0.30 * suggested, 7)
        return blended

    def apply_learning(self, strategy: dict) -> dict:
        """Return a copy of strategy with self-learned threshold adjustments."""
        s = copy.deepcopy(strategy)
        if "stake_pct" in s:
            s["stake_pct"] = self.suggest_stake_pct(s["stake_pct"])
            
        for cond in (
            s.get("call_conditions", []) +
            s.get("put_conditions",  []) +
            s.get("filter_conditions", [])
        ):
            if "threshold" in cond:
                cond["threshold"] = self.suggest_threshold(cond["name"], cond["threshold"])
        return s

    def condition_stats(self) -> dict:
        """Summary of how many samples each condition has accumulated."""
        stats = {name: len(v) for name, v in self._history.items()}
        if hasattr(self, "_stake_history"):
            stats["stake_pct"] = len(self._stake_history)
        return stats
