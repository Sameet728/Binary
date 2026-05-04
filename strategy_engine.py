"""
strategy_engine.py — Strategy Generator, Scorer, and Genetic Algorithm
Strategies are stored as JSON objects containing condition lists.
Conditions are evaluated row-by-row against the feature DataFrame.
"""

import copy
import json
import os
import random
import uuid

import numpy as np
import pandas as pd

LEADERBOARD_FILE = "leaderboard.json"
STRATEGIES_FILE = "strategies.json"

# ---------------------------------------------------------------------------
# CONDITION TEMPLATES
# Each key maps to the feature name, comparison operator, and threshold range.
# 'range' triggers random sampling; 'value' uses a fixed value.
# ---------------------------------------------------------------------------

CONDITION_TEMPLATES: dict = {
    # RSI
    "rsi_oversold":        {"feature": "rsi",           "op": "<",  "range": (20, 38)},
    "rsi_overbought":      {"feature": "rsi",           "op": ">",  "range": (62, 80)},
    "rsi_mid_bull":        {"feature": "rsi",           "op": ">",  "range": (45, 55)},
    # Trend
    "trend_up":            {"feature": "trend_up",      "op": "==", "value": 1},
    "trend_down":          {"feature": "trend_up",      "op": "==", "value": 0},
    "trend_strong_bull":   {"feature": "trend_strong",  "op": "==", "value": 1},
    "trend_strong_bear":   {"feature": "trend_strong",  "op": "==", "value": 0},
    # EMA spread
    "ema9_above_ema21":    {"feature": "ema9_vs_ema21", "op": ">",  "range": (0.0001, 0.005)},
    "ema9_below_ema21":    {"feature": "ema9_vs_ema21", "op": "<",  "range": (-0.005, -0.0001)},
    # ADX
    "adx_trending":        {"feature": "adx",           "op": ">",  "range": (20, 35)},
    "adx_ranging":         {"feature": "adx",           "op": "<",  "range": (15, 25)},
    "di_plus_dominant":    {"feature": "di_plus",       "op": ">",  "range": (20, 35)},
    "di_minus_dominant":   {"feature": "di_minus",      "op": ">",  "range": (20, 35)},
    # Bollinger
    "bb_oversold":         {"feature": "bb_pct",        "op": "<",  "range": (0.05, 0.25)},
    "bb_overbought":       {"feature": "bb_pct",        "op": ">",  "range": (0.75, 0.95)},
    "bb_mid_bounce":       {"feature": "bb_pct",        "op": ">",  "range": (0.40, 0.55)},
    "low_volatility":      {"feature": "bb_width",      "op": "<",  "range": (0.010, 0.035)},
    "high_volatility":     {"feature": "bb_width",      "op": ">",  "range": (0.030, 0.080)},
    # ATR
    "atr_low":             {"feature": "atr_pct",       "op": "<",  "range": (0.003, 0.010)},
    "atr_high":            {"feature": "atr_pct",       "op": ">",  "range": (0.008, 0.025)},
    # Structure
    "breakout_high":       {"feature": "breakout_high", "op": "==", "value": 1},
    "breakout_low":        {"feature": "breakout_low",  "op": "==", "value": 1},
    # Price action
    "bullish_engulf":      {"feature": "bullish_engulf","op": "==", "value": 1},
    "bearish_engulf":      {"feature": "bearish_engulf","op": "==", "value": 1},
    "bull_pin":            {"feature": "bull_pin",      "op": "==", "value": 1},
    "bear_pin":            {"feature": "bear_pin",      "op": "==", "value": 1},
    "inside_bar":          {"feature": "inside_bar",    "op": "==", "value": 1},
    # Candle
    "bullish_candle":      {"feature": "is_bullish",    "op": "==", "value": 1},
    "bearish_candle":      {"feature": "is_bullish",    "op": "==", "value": 0},
    "large_body":          {"feature": "body_pct",      "op": ">",  "range": (0.50, 0.80)},
    "small_body":          {"feature": "body_pct",      "op": "<",  "range": (0.20, 0.45)},
}

# Semantically appropriate conditions for CALL (long) signals
CALL_CONDITIONS = [
    "rsi_oversold", "trend_up", "trend_strong_bull", "ema9_above_ema21",
    "adx_trending", "di_plus_dominant", "bb_oversold", "bb_mid_bounce",
    "bullish_engulf", "bull_pin", "breakout_high", "bullish_candle", "large_body",
]

# Semantically appropriate conditions for PUT (short) signals
PUT_CONDITIONS = [
    "rsi_overbought", "trend_down", "trend_strong_bear", "ema9_below_ema21",
    "adx_trending", "di_minus_dominant", "bb_overbought",
    "bearish_engulf", "bear_pin", "breakout_low", "bearish_candle", "large_body",
]

# Neutral filters (apply to both directions)
FILTER_CONDITIONS = [
    "low_volatility", "high_volatility", "atr_low", "atr_high",
    "inside_bar", "small_body", "adx_ranging",
]


# ---------------------------------------------------------------------------
# CONDITION HELPERS
# ---------------------------------------------------------------------------

def _make_condition(name: str) -> dict:
    """Instantiate a condition from its template with sampled or fixed threshold."""
    t = CONDITION_TEMPLATES[name]
    cond = {"name": name, "feature": t["feature"], "op": t["op"]}
    if "range" in t:
        lo, hi = t["range"]
        cond["threshold"] = round(random.uniform(lo, hi), 7)
    else:
        cond["threshold"] = t["value"]
    return cond


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def evaluate_conditions(row: pd.Series, conditions: list) -> bool:
    """Return True only if ALL conditions hold for the given row."""
    for cond in conditions:
        feat = cond["feature"]
        if feat not in row.index:
            return False
        val = row[feat]
        if pd.isna(val):
            return False
        op, thr = cond["op"], cond["threshold"]
        if   op == ">":  result = float(val) > thr
        elif op == "<":  result = float(val) < thr
        elif op == "==": result = float(val) == thr
        else: result = False
        if not result:
            return False
    return True


# ---------------------------------------------------------------------------
# STRATEGY GENERATION
# ---------------------------------------------------------------------------

def generate_strategy(generation: int = 0) -> dict:
    """
    Randomly construct a strategy.
    Each strategy has:
      call_conditions  — AND-combined; if ALL true → CALL signal
      put_conditions   — AND-combined; if ALL true → PUT signal
      filter_conditions — optional gate applied before directional check
    """
    n_call   = random.randint(2, 4)
    n_put    = random.randint(2, 4)
    n_filter = random.randint(0, 2)

    call_names   = random.sample(CALL_CONDITIONS,   min(n_call,   len(CALL_CONDITIONS)))
    put_names    = random.sample(PUT_CONDITIONS,    min(n_put,    len(PUT_CONDITIONS)))
    filter_names = random.sample(FILTER_CONDITIONS, min(n_filter, len(FILTER_CONDITIONS)))

    return {
        "id":                _new_id(),
        "generation":        generation,
        "call_conditions":   [_make_condition(n) for n in call_names],
        "put_conditions":    [_make_condition(n) for n in put_names],
        "filter_conditions": [_make_condition(n) for n in filter_names],
        "payout":            0.71,
        "stake_pct":         round(random.uniform(0.01, 0.05), 4),
        "score":             0.0,
        "metrics":           {},
    }


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def compute_score(metrics: dict) -> float:
    """
    Composite score:
      30% win-rate  (0→1)
      30% return    (capped at 3×, normalised 0→1)
      20% inv-drawdown  (penalises as DD approaches 0.30)
      20% profit-factor (capped at 5×, normalised 0→1)
    """
    wr  = float(metrics.get("win_rate",      0))
    ret = min(float(metrics.get("total_return", 0)), 3.0) / 3.0
    dd  = float(metrics.get("max_drawdown",  1.0))
    pf  = min(float(metrics.get("profit_factor", 0)), 5.0) / 5.0

    # Inverse drawdown: 1.0 at dd=0, 0.0 at dd≥0.30
    inv_dd = max(0.0, 1.0 - dd / 0.30)

    score = 0.30 * wr + 0.30 * ret + 0.20 * inv_dd + 0.20 * pf
    return round(score, 4)


# ---------------------------------------------------------------------------
# GENETIC ALGORITHM
# ---------------------------------------------------------------------------

def mutate_strategy(strategy: dict) -> dict:
    """Return a mutated copy of the strategy."""
    s = copy.deepcopy(strategy)
    s["id"] = _new_id()
    s["generation"] = strategy.get("generation", 0) + 1
    s["score"] = 0.0
    s["metrics"] = {}

    mutation = random.choice(["threshold", "add_call", "add_put",
                               "remove_call", "remove_put", "swap_call", "swap_put", "stake_pct"])

    if mutation == "stake_pct":
        s["stake_pct"] = round(float(np.clip(s.get("stake_pct", 0.02) * random.uniform(0.80, 1.20), 0.005, 0.10)), 4)

    elif mutation == "threshold":
        # Perturb a random threshold by ±20 %
        pool = s["call_conditions"] + s["put_conditions"] + s["filter_conditions"]
        if pool:
            cond = random.choice(pool)
            if "range" in CONDITION_TEMPLATES.get(cond["name"], {}):
                lo, hi = CONDITION_TEMPLATES[cond["name"]]["range"]
                cond["threshold"] = round(
                    float(np.clip(cond["threshold"] * random.uniform(0.80, 1.20), lo, hi)), 7
                )

    elif mutation == "add_call" and len(s["call_conditions"]) < 5:
        name = random.choice(CALL_CONDITIONS)
        s["call_conditions"].append(_make_condition(name))

    elif mutation == "add_put" and len(s["put_conditions"]) < 5:
        name = random.choice(PUT_CONDITIONS)
        s["put_conditions"].append(_make_condition(name))

    elif mutation == "remove_call" and len(s["call_conditions"]) > 1:
        idx = random.randrange(len(s["call_conditions"]))
        s["call_conditions"].pop(idx)

    elif mutation == "remove_put" and len(s["put_conditions"]) > 1:
        idx = random.randrange(len(s["put_conditions"]))
        s["put_conditions"].pop(idx)

    elif mutation == "swap_call" and s["call_conditions"]:
        idx = random.randrange(len(s["call_conditions"]))
        s["call_conditions"][idx] = _make_condition(random.choice(CALL_CONDITIONS))

    elif mutation == "swap_put" and s["put_conditions"]:
        idx = random.randrange(len(s["put_conditions"]))
        s["put_conditions"][idx] = _make_condition(random.choice(PUT_CONDITIONS))

    return s


def crossover_strategies(s1: dict, s2: dict) -> dict:
    """Produce a child by mixing conditions from two parent strategies."""
    child = copy.deepcopy(s1)
    child["id"] = _new_id()
    child["generation"] = max(s1.get("generation", 0), s2.get("generation", 0)) + 1
    child["score"] = 0.0
    child["metrics"] = {}
    child["stake_pct"] = s1.get("stake_pct", 0.02) if random.random() < 0.5 else s2.get("stake_pct", 0.02)

    # Mix call conditions (deduplicate by condition name)
    all_call = s1["call_conditions"] + s2["call_conditions"]
    random.shuffle(all_call)
    seen: set = set()
    mixed_call = []
    for c in all_call:
        if c["name"] not in seen and len(mixed_call) < 4:
            mixed_call.append(copy.deepcopy(c))
            seen.add(c["name"])
    child["call_conditions"] = mixed_call or [_make_condition(random.choice(CALL_CONDITIONS))]

    # Mix put conditions
    all_put = s1["put_conditions"] + s2["put_conditions"]
    random.shuffle(all_put)
    seen = set()
    mixed_put = []
    for c in all_put:
        if c["name"] not in seen and len(mixed_put) < 4:
            mixed_put.append(copy.deepcopy(c))
            seen.add(c["name"])
    child["put_conditions"] = mixed_put or [_make_condition(random.choice(PUT_CONDITIONS))]

    return child


# ---------------------------------------------------------------------------
# LEADERBOARD
# ---------------------------------------------------------------------------

def load_leaderboard() -> list:
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE) as fh:
                return json.load(fh)
        except Exception:
            pass
    return []


def save_leaderboard(leaderboard: list) -> None:
    with open(LEADERBOARD_FILE, "w") as fh:
        json.dump(leaderboard, fh, indent=2)


def update_leaderboard(strategy: dict, leaderboard: list, max_size: int = 20) -> list:
    """Insert / replace strategy in leaderboard and keep top max_size."""
    leaderboard = [s for s in leaderboard if s["id"] != strategy["id"]]
    leaderboard.append(strategy)
    leaderboard.sort(key=lambda x: x.get("score", 0), reverse=True)
    return leaderboard[:max_size]
