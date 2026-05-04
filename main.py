"""
main.py — Binary Strategy Auto-Discovery System  (FastAPI Backend)
Run:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET /           → serve index.html
  GET /start      → start discovery engine
  GET /stop       → stop discovery engine
  GET /status     → live stats
  GET /leaderboard→ top strategies (lightweight)
  GET /equity/{id}→ equity curve for one strategy
  GET /download/strategies
  GET /download/results
"""

import csv
import json
import logging
import os
import random
import threading
import time
from datetime import datetime
from typing import Optional

import io
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from backtester import run_backtest
from ml_models import SelfLearner, StrategyOptimizer, TradePredictor
from strategy_engine import (
    compute_score,
    crossover_strategies,
    generate_strategy,
    load_leaderboard,
    mutate_strategy,
    save_leaderboard,
    update_leaderboard,
)
from utils import compute_features, fetch_ohlcv

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")
# ---------------------------------------------------------------------------

app = FastAPI(title="Binary Strategy Discovery API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRADING_PAIRS = [
    "EUR/USDT", "GBP/USDT"
]
TIMEFRAMES = ["5m", "15m"]

RESULTS_FILE = "results.csv"

# ---------------------------------------------------------------------------
# Shared state (accessed from background thread + API handlers)
# ---------------------------------------------------------------------------
state: dict = {
    "running":            False,
    "strategies_tested":  0,
    "best_score":         0.0,
    "best_win_rate":      0.0,
    "current_status":     "Idle",
    "errors":             0,
    "leaderboard":        [],
    "last_equity":        [],       # equity curve of best strategy
}

# ML singletons
trade_predictor      = TradePredictor()
strategy_optimizer   = StrategyOptimizer()
self_learner         = SelfLearner()

_stop_event   = threading.Event()
_engine_thread: Optional[threading.Thread] = None


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _init_results_csv() -> None:
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as fh:
            csv.writer(fh).writerow([
                "timestamp", "strategy_id", "generation",
                "win_rate", "total_return", "max_drawdown",
                "profit_factor", "sharpe_ratio", "trades", "score", "stake_pct"
            ])


def _append_result(strategy: dict) -> None:
    m = strategy.get("metrics", {})
    with open(RESULTS_FILE, "a", newline="") as fh:
        csv.writer(fh).writerow([
            datetime.utcnow().isoformat(),
            strategy["id"],
            strategy.get("generation", 0),
            m.get("win_rate",      0),
            m.get("total_return",  0),
            m.get("max_drawdown",  0),
            m.get("profit_factor", 0),
            m.get("sharpe_ratio",  0),
            m.get("total_trades",  0),
            strategy.get("score",  0),
            strategy.get("stake_pct", 0.02),
        ])


def _add_noise(df, noise_pct: float = 0.0003):
    """Simulate minor broker price differences with Gaussian noise."""
    df = df.copy()
    for col in ("open", "high", "low", "close"):
        noise = np.random.normal(0, noise_pct, len(df))
        df[col] = df[col] * (1 + noise)
    # Ensure high >= low after noise
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"]  = df[["low",  "open", "close"]].min(axis=1)
    return df


# ---------------------------------------------------------------------------
# DISCOVERY ENGINE
# ---------------------------------------------------------------------------

GLOBAL_DATA_CACHE = {}

def _discovery_loop() -> None:
    global GLOBAL_DATA_CACHE
    logger.info("Discovery engine starting…")
    state["current_status"] = "Fetching market data…"
    _init_results_csv()

    # ---- initial data fetch ----
    try:
        data_cache = GLOBAL_DATA_CACHE
        if not data_cache:
            for pair in TRADING_PAIRS:
                for tf in TIMEFRAMES:
                    try:
                        df_raw = fetch_ohlcv(symbol=pair, timeframe=tf, exchange_id="binance")
                        data_cache[(pair, tf)] = compute_features(df_raw)
                        logger.info("Feature data ready for %s %s: %d rows", pair, tf, len(data_cache[(pair, tf)]))
                    except Exception as e:
                        logger.warning("Could not fetch initial data for %s %s: %s", pair, tf, e)
                
        if not data_cache:
            raise ValueError("No data available for any trading pair")
            
    except Exception as exc:
        logger.error("Cannot fetch data: %s", exc)
        state["current_status"] = f"Data error: {exc}"
        state["running"] = False
        return

    # ---- initial ML training ----
    try:
        # Use first available pair for initial TradePredictor training
        first_pair_df = next(iter(data_cache.values()))
        acc = trade_predictor.train(first_pair_df)
        logger.info("TradePredictor trained | CV accuracy: %.3f", acc)
    except Exception as exc:
        logger.warning("TradePredictor training skipped: %s", exc)

    leaderboard       = load_leaderboard()
    state["leaderboard"] = leaderboard

    data_counter    = 0     # refresh data every N iterations
    ml_counter      = 0     # retrain ML every N strategies

    DATA_REFRESH_EVERY = 60
    ML_RETRAIN_EVERY   = 150

    while not _stop_event.is_set():
        try:
            # ---- periodic data refresh ----
            if data_counter >= DATA_REFRESH_EVERY:
                try:
                    for pair in TRADING_PAIRS:
                        for tf in TIMEFRAMES:
                            try:
                                df_raw = fetch_ohlcv(symbol=pair, timeframe=tf, exchange_id="binance")
                                GLOBAL_DATA_CACHE[(pair, tf)] = compute_features(df_raw)
                            except Exception as e:
                                pass
                    data_counter = 0
                    logger.info("Data refreshed for all pairs and timeframes")
                except Exception as exc:
                    logger.warning("Data refresh failed: %s", exc)

            # ---- periodic ML retrain ----
            if ml_counter >= ML_RETRAIN_EVERY and len(leaderboard) >= 20:
                try:
                    acc = strategy_optimizer.train(leaderboard)
                    logger.info("StrategyOptimizer retrained | acc: %.3f", acc)
                    ml_counter = 0
                except Exception as exc:
                    logger.warning("ML retrain failed: %s", exc)

            # ---- strategy selection ----
            use_evolution = (
                len(leaderboard) >= 4 and
                random.random() < 0.55   # 55 % evolve, 45 % fresh
            )

            if use_evolution:
                top = leaderboard[:min(8, len(leaderboard))]
                op  = random.choice(["mutate", "crossover"])
                if op == "mutate":
                    parent   = random.choice(top)
                    strategy = mutate_strategy(parent)
                else:
                    p1, p2   = random.sample(top, k=2)
                    strategy = crossover_strategies(p1, p2)

                # Apply self-learner adjustments after ~100 strategies
                if state["strategies_tested"] > 100:
                    strategy = self_learner.apply_learning(strategy)
            else:
                strategy = generate_strategy()

            # ---- optional ML pre-filter ----
            if strategy_optimizer.trained:
                q = strategy_optimizer.predict_quality(strategy)
                if q < 0.20 and random.random() < 0.70:
                    # Skip low-confidence fresh strategies most of the time
                    state["strategies_tested"] += 1
                    data_counter += 1
                    ml_counter   += 1
                    continue

            # ---- noise + backtest ----
            current_pair, current_tf = random.choice(list(data_cache.keys()))
            df_current = data_cache[(current_pair, current_tf)]
            
            df_noisy = _add_noise(df_current)
            state["current_status"] = f"Testing {strategy['id']} on {current_pair} ({current_tf})…"

            metrics = run_backtest(df_noisy, strategy, min_trades=20)
            
            if metrics is not None:
                metrics["symbol"] = current_pair
                metrics["timeframe"] = current_tf
            state["strategies_tested"] += 1
            data_counter  += 1
            ml_counter    += 1

            if metrics is None:
                continue

            # ---- score & record ----
            score              = compute_score(metrics)
            strategy["metrics"] = metrics
            strategy["score"]   = score

            self_learner.record(strategy, metrics)
            _append_result(strategy)

            # ---- update live stats ----
            if score > state["best_score"]:
                state["best_score"]    = round(score, 4)
                state["last_equity"]   = metrics.get("equity_curve", [])

            wr = metrics.get("win_rate", 0)
            if wr > state["best_win_rate"]:
                state["best_win_rate"] = round(wr, 4)

            state["current_status"] = (
                f"Tested: {state['strategies_tested']} | "
                f"Best score: {state['best_score']:.4f} | "
                f"Best WR: {state['best_win_rate']:.2%}"
            )

            # ---- leaderboard filter ----
            if (wr >= 0.60
                    and metrics.get("max_drawdown", 1) < 0.30
                    and metrics.get("total_trades",  0) >= 20):
                leaderboard          = update_leaderboard(strategy, leaderboard)
                state["leaderboard"] = leaderboard
                save_leaderboard(leaderboard)
                logger.info(
                    "LB update  id=%s  WR=%.2f%%  DD=%.2f%%  PF=%.2f  score=%.4f",
                    strategy["id"], wr * 100,
                    metrics["max_drawdown"] * 100,
                    metrics["profit_factor"],
                    score,
                )

            time.sleep(0.02)   # yield CPU

        except Exception as exc:
            state["errors"] += 1
            logger.error("Engine error: %s", exc, exc_info=True)
            time.sleep(1)

    state["current_status"] = "Stopped"
    state["running"]        = False
    logger.info("Discovery engine stopped")


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/")
def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)


@app.get("/start")
def start_engine():
    global _engine_thread
    if state["running"]:
        return {"status": "already_running", "strategies_tested": state["strategies_tested"]}

    _stop_event.clear()
    state["running"] = True
    state["current_status"] = "Starting…"

    _engine_thread = threading.Thread(target=_discovery_loop, daemon=True, name="DiscoveryEngine")
    _engine_thread.start()
    return {"status": "started"}


@app.get("/stop")
def stop_engine():
    _stop_event.set()
    state["running"] = False
    return {"status": "stopping"}


@app.get("/status")
def get_status():
    return {
        "running":           state["running"],
        "strategies_tested": state["strategies_tested"],
        "best_score":        state["best_score"],
        "best_win_rate":     state["best_win_rate"],
        "current_status":    state["current_status"],
        "errors":            state["errors"],
        "leaderboard_size":  len(state["leaderboard"]),
        "ml": {
            "trade_predictor_accuracy":    round(trade_predictor.accuracy, 3),
            "optimizer_accuracy":          round(strategy_optimizer.accuracy, 3),
            "selflearner_conditions":      len(self_learner.condition_stats()),
        },
    }


@app.get("/leaderboard")
def get_leaderboard():
    lb = state["leaderboard"] or load_leaderboard()
    result = []
    for s in lb[:20]:
        m = s.get("metrics", {})
        result.append({
            "id":         s["id"],
            "symbol":     m.get("symbol", "BTC/USDT"),
            "timeframe":  m.get("timeframe", "15m"),
            "score":      s.get("score", 0),
            "generation": s.get("generation", 0),
            "win_rate":   m.get("win_rate",      0),
            "total_return": m.get("total_return", 0),
            "max_drawdown": m.get("max_drawdown", 0),
            "profit_factor": m.get("profit_factor", 0),
            "sharpe_ratio":  m.get("sharpe_ratio",  0),
            "total_trades":  m.get("total_trades",  0),
            "max_losing_streak": m.get("max_losing_streak", 0),
            "stake_pct":  s.get("stake_pct", 0.02),
        })
    return result


@app.get("/equity/{strategy_id}")
def get_equity(strategy_id: str):
    lb = state["leaderboard"] or load_leaderboard()
    for s in lb:
        if s["id"] == strategy_id:
            return s.get("metrics", {}).get("equity_curve", [])
    return []


@app.get("/best_equity")
def get_best_equity():
    return state.get("last_equity", [])


@app.get("/download/strategies")
def download_strategies():
    if os.path.exists("leaderboard.json"):
        return FileResponse("leaderboard.json", media_type="application/json",
                            filename="strategies.json")
    return JSONResponse({"error": "No strategies yet"}, status_code=404)


@app.get("/download/results")
def download_results():
    if os.path.exists(RESULTS_FILE):
        return FileResponse(RESULTS_FILE, media_type="text/csv", filename="results.csv")
    return JSONResponse({"error": "No results yet"}, status_code=404)


@app.get("/download/trades/{strategy_id}")
def download_trades(strategy_id: str):
    lb = state["leaderboard"] or load_leaderboard()
    for s in lb:
        if s["id"] == strategy_id:
            trades = s.get("metrics", {}).get("trades", [])
            if not trades:
                return JSONResponse({"error": "No trades stored"}, status_code=404)
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["timestamp", "signal", "entry", "expiry", "win", "pnl", "balance"])
            writer.writeheader()
            writer.writerows(trades)
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=trades_{strategy_id}.csv"}
            )
    return JSONResponse({"error": "Strategy not found"}, status_code=404)


def _render_poster_loop() -> None:
    render_url = os.environ.get("RENDER_URL", "https://binary-kp67.onrender.com/api/data")
    logger.info(f"Render poster active. Will send data every 60s to {render_url}")
    while True:
        time.sleep(60)
        
        if not state["running"] and not state["leaderboard"]:
            continue
            
        try:
            lb = get_leaderboard()
            equities = {}
            # state["leaderboard"] has the full objects including "metrics"
            for s in state["leaderboard"]:
                equities[s["id"]] = s.get("metrics", {}).get("equity_curve", [])
            
            payload = {
                "status": get_status(),
                "leaderboard": lb,
                "equities": equities
            }
            
            resp = requests.post(render_url, json=payload, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Failed to post to backend UI: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"Error posting to backend UI at {render_url}: {e}")

threading.Thread(target=_render_poster_loop, daemon=True, name="RenderPoster").start()

# Auto-start the engine when the server starts
start_engine()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
