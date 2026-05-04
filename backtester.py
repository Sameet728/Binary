"""
backtester.py — Binary-Options Backtest Engine
Entry:  candle close price
Expiry: next candle close price
WIN if price moves in predicted direction; LOSS otherwise.
"""

import numpy as np
import pandas as pd

from strategy_engine import evaluate_conditions


def run_backtest(
    df: pd.DataFrame,
    strategy: dict,
    initial_balance: float = 1_000.0,
    min_trades: int = 20,
) -> dict | None:
    """
    Vectorised binary backtest.

    Returns a metrics dict (or None if fewer than min_trades generated).
    The equity_curve is downsampled to ≤ 300 points to keep storage light.
    Only the last 50 individual trades are stored.
    """
    payout         = float(strategy.get("payout", 0.71))
    stake_pct      = float(strategy.get("stake_pct", 0.02))
    call_conds     = strategy["call_conditions"]
    put_conds      = strategy["put_conditions"]
    filter_conds   = strategy.get("filter_conditions", [])

    balance        = initial_balance
    equity_curve   = [balance]
    trades         = []

    for i in range(len(df) - 1):      # need i+1 for expiry price
        row      = df.iloc[i]
        next_row = df.iloc[i + 1]

        # Optional gate: all filter conditions must pass
        if filter_conds and not evaluate_conditions(row, filter_conds):
            equity_curve.append(balance)
            continue

        # Directional signal
        if   evaluate_conditions(row, call_conds): signal = "CALL"
        elif evaluate_conditions(row, put_conds):  signal = "PUT"
        else:
            equity_curve.append(balance)
            continue

        # Outcome
        entry_price  = float(row["close"])
        expiry_price = float(next_row["close"])
        stake        = balance * stake_pct

        if signal == "CALL":
            win = expiry_price > entry_price
        else:
            win = expiry_price < entry_price

        pnl      = stake * payout if win else -stake
        balance  = max(balance + pnl, 0.0)

        trades.append({
            "timestamp":    str(df.index[i]),
            "signal":       signal,
            "entry":        round(entry_price, 4),
            "expiry":       round(expiry_price, 4),
            "win":          bool(win),
            "pnl":          round(pnl, 4),
            "balance":      round(balance, 2),
        })
        equity_curve.append(balance)

    if len(trades) < min_trades:
        return None

    return _compute_metrics(trades, equity_curve, initial_balance)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def _compute_metrics(trades: list, equity_curve: list, initial_balance: float) -> dict:
    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]

    win_rate     = len(wins) / len(trades)
    final_bal    = equity_curve[-1]
    total_return = (final_bal - initial_balance) / initial_balance

    # Max drawdown
    eq   = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / np.where(peak == 0, 1, peak)
    max_drawdown = float(dd.max())

    # Profit factor
    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses)) or 1e-9
    profit_factor = gross_win / gross_loss

    # Sharpe (annualised, assuming 252 trading "sessions")
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    std  = pnls.std()
    sharpe = float(pnls.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    # Max losing streak
    streak = max_streak = 0
    for t in trades:
        if not t["win"]:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Monthly PnL
    df_t = pd.DataFrame(trades)
    df_t["timestamp"] = pd.to_datetime(df_t["timestamp"])
    df_t["month"]     = df_t["timestamp"].dt.to_period("M")
    monthly_pnl = {
        str(k): round(float(v), 2)
        for k, v in df_t.groupby("month")["pnl"].sum().items()
    }

    # Yearly PnL
    df_t["year"] = df_t["timestamp"].dt.year
    yearly_pnl = {
        str(k): round(float(v), 2)
        for k, v in df_t.groupby("year")["pnl"].sum().items()
    }

    # Downsample equity curve to ≤ 300 points
    step = max(1, len(equity_curve) // 300)
    eq_sampled = [round(float(v), 2) for v in equity_curve[::step]]

    return {
        "total_trades":      len(trades),
        "win_count":         len(wins),
        "loss_count":        len(losses),
        "win_rate":          round(win_rate, 4),
        "total_return":      round(total_return, 4),
        "final_balance":     round(final_bal, 2),
        "max_drawdown":      round(max_drawdown, 4),
        "profit_factor":     round(profit_factor, 4),
        "sharpe_ratio":      round(sharpe, 4),
        "max_losing_streak": int(max_streak),
        "monthly_pnl":       monthly_pnl,
        "yearly_pnl":        yearly_pnl,
        "equity_curve":      eq_sampled,
        "trades":            trades[-5000:],   # keep last 5000
    }
