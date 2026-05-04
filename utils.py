"""
utils.py — Data Engine & Feature Engineering
Fetches OHLCV from Binance via CCXT, caches to CSV,
and computes all technical indicators needed by the strategy engine.
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import ccxt

logger = logging.getLogger(__name__)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DATA ENGINE
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    years: int = 5,
    exchange_id: str = "binance",
    cache_minutes: int = 60,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles.  Returns a DataFrame with DatetimeIndex.
    Results are cached to CSV and reused for `cache_minutes` minutes.
    Falls back to cached data on network errors.
    """
    safe_sym = symbol.replace("/", "_")
    cache_file = os.path.join(CACHE_DIR, f"{safe_sym}_{timeframe}_{years}y.csv")

    # Return cached data if fresh enough
    if os.path.exists(cache_file):
        age_min = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 60
        if age_min < cache_minutes:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info("Cache hit: %d candles loaded", len(df))
            return df

    if exchange_id == "yfinance":
        try:
            import yfinance as yf
            logger.info(f"Fetching max available {timeframe} data for {symbol} via yfinance...")
            yf_symbol = symbol.replace("/", "") + "=X"
            ticker = yf.Ticker(yf_symbol)
            # YFinance max for 15m is 60d
            df = ticker.history(period="60d" if timeframe in ["5m", "15m", "1h"] else "max", interval=timeframe)
            if df.empty:
                raise ValueError("No data returned from yfinance")
            
            df.reset_index(inplace=True)
            # YFinance might return 'Datetime' or 'Date'
            if "Datetime" in df.columns:
                df.rename(columns={"Datetime": "timestamp"}, inplace=True)
            elif "Date" in df.columns:
                df.rename(columns={"Date": "timestamp"}, inplace=True)
                
            if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
                
            # lowercase column names
            df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
            df.set_index("timestamp", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]]
            
            df.to_csv(cache_file)
            logger.info("Fetched %d candles from yfinance for %s", len(df), symbol)
            return df
        except Exception as exc:
            logger.error("Fetch error (yfinance): %s", exc)
            if os.path.exists(cache_file):
                logger.warning("Using stale cache as fallback")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            raise

    try:
        import time
        exchange_cls = getattr(ccxt, exchange_id)
        exchange = exchange_cls({"enableRateLimit": True})
        
        limit = 1000
        since = int((datetime.now().timestamp() - years * 365 * 24 * 3600) * 1000)
        all_candles = []
        
        logger.info(f"Fetching {years} years of {timeframe} data for {symbol}...")
        while True:
            raw = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not raw:
                break
            all_candles.extend(raw)
            since = raw[-1][0] + 1
            if len(raw) < limit:
                break
            time.sleep(exchange.rateLimit / 1000.0 if hasattr(exchange, 'rateLimit') else 0.1)
            
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.to_csv(cache_file)
        logger.info("Fetched %d candles from %s", len(df), exchange_id)
        return df
    except Exception as exc:
        logger.error("Fetch error (%s): %s", exchange_id, exc)
        if os.path.exists(cache_file):
            logger.warning("Using stale cache as fallback")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise


# ---------------------------------------------------------------------------
# INDICATORS  (all implemented manually — no external TA library needed)
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Return DataFrame with columns: adx, di_plus, di_minus."""
    high, low = df["high"], df["low"]
    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    # Zero out when the other is larger
    dm_plus = dm_plus.where(dm_plus > (-low.diff()).clip(lower=0), 0)
    dm_minus = dm_minus.where(dm_minus > high.diff().clip(lower=0), 0)

    atr = _atr(df, period)
    di_p = 100 * dm_plus.rolling(period).mean() / atr.replace(0, np.nan)
    di_m = 100 * dm_minus.rolling(period).mean() / atr.replace(0, np.nan)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return pd.DataFrame({"adx": adx, "di_plus": di_p, "di_minus": di_m})


# ---------------------------------------------------------------------------
# FEATURE ENGINE
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full feature set from raw OHLCV data.
    Returns a cleaned DataFrame (NaN rows dropped).
    """
    df = df.copy()
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]

    # --- Trend: EMA 9 / 21 / 50 ---
    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    # --- Momentum: RSI ---
    df["rsi"] = _rsi(close, 14)

    # --- Volatility: Bollinger Bands ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_mid"] = bb_mid
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"] = (close - df["bb_lower"]) / bb_range           # 0=lower, 1=upper
    df["bb_width"] = bb_range / bb_mid                            # normalised width

    # --- Volatility: ATR ---
    df["atr"] = _atr(df, 14)
    df["atr_pct"] = df["atr"] / close.replace(0, np.nan)

    # --- Market State: ADX ---
    adx_df = _adx(df, 14)
    df["adx"] = adx_df["adx"]
    df["di_plus"] = adx_df["di_plus"]
    df["di_minus"] = adx_df["di_minus"]

    # --- Structure: Previous High/Low Breakout ---
    df["prev_high"] = high.shift(1)
    df["prev_low"] = low.shift(1)
    df["breakout_high"] = (close > df["prev_high"]).astype(int)
    df["breakout_low"] = (close < df["prev_low"]).astype(int)

    # --- Candle Metrics ---
    df["body_size"] = (close - open_).abs()
    df["candle_range"] = (high - low).replace(0, np.nan)
    df["body_pct"] = df["body_size"] / df["candle_range"]

    top_of_body = pd.concat([open_, close], axis=1).max(axis=1)
    bot_of_body = pd.concat([open_, close], axis=1).min(axis=1)
    df["upper_wick"] = high - top_of_body
    df["lower_wick"] = bot_of_body - low
    df["wick_ratio"] = (df["upper_wick"] - df["lower_wick"]) / df["candle_range"]

    # --- Price Action: Engulfing Candle ---
    prev_bull = open_.shift(1) < close.shift(1)
    prev_bear = open_.shift(1) > close.shift(1)
    curr_bull = open_ < close
    curr_bear = open_ > close

    df["bullish_engulf"] = (
        curr_bull & prev_bear &
        (open_ <= close.shift(1)) &
        (close >= open_.shift(1))
    ).astype(int)

    df["bearish_engulf"] = (
        curr_bear & prev_bull &
        (open_ >= close.shift(1)) &
        (close <= open_.shift(1))
    ).astype(int)

    # --- Price Action: Pin Bar ---
    df["bull_pin"] = (
        (df["lower_wick"] > 2 * df["body_size"]) &
        (df["upper_wick"] < df["body_size"])
    ).astype(int)

    df["bear_pin"] = (
        (df["upper_wick"] > 2 * df["body_size"]) &
        (df["lower_wick"] < df["body_size"])
    ).astype(int)

    # --- Price Action: Inside Bar ---
    df["inside_bar"] = (
        (high < high.shift(1)) & (low > low.shift(1))
    ).astype(int)

    # --- Composite Trend Signals ---
    df["trend_up"] = (df["ema9"] > df["ema21"]).astype(int)
    df["trend_strong"] = (df["ema21"] > df["ema50"]).astype(int)
    df["is_bullish"] = (close > open_).astype(int)

    # --- Normalised Price vs EMA ---
    df["close_vs_ema9"] = (close - df["ema9"]) / df["ema9"].replace(0, np.nan)
    df["close_vs_ema21"] = (close - df["ema21"]) / df["ema21"].replace(0, np.nan)
    df["ema9_vs_ema21"] = (df["ema9"] - df["ema21"]) / df["ema21"].replace(0, np.nan)

    df.dropna(inplace=True)
    return df
