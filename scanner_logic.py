"""
MITS 360 Free Scanner — Core Logic Module
===========================================
4 rule-based scanners: Base Breakout (VCP-style), RS Rank, MA Crossover, Volume Shocker.
No AI/LLM involved — pure pandas/numpy calculations on daily OHLCV data.

Input contract for every function: a pandas DataFrame with columns
    ['date', 'open', 'high', 'low', 'close', 'volume']
sorted ascending by date (oldest first), for ONE stock.

These functions are data-source agnostic — feed them data from Angel One
SmartAPI's getCandleData (or any other EOD source) once wired up.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. BASE BREAKOUT (VCP-style, with lifecycle stage)
# ---------------------------------------------------------------------------

def detect_base_breakout(df: pd.DataFrame, lookback: int = 40,
                          freshness_days: int = 10,
                          magnitude_pct: float = 5.0,
                          volume_sma_period: int = 10,
                          volume_pct_above: float = 25.0) -> dict:
    """
    Matches the exact rule set specified (screener-style config):
      1. Break Out of `lookback` (40) period High — today's close > highest
         high of the preceding `lookback` candles.
      2. Freshness: that prior high must NOT have already been closed above
         in the last `freshness_days` (10) sessions — avoids flagging a
         breakout that already happened recently (whipsaw/repeat signals).
      3. Magnitude: today's close must be at least `magnitude_pct` (5%)
         above that prior high — filters out marginal/weak breaks.
      4. Volume: today's volume >= (1 + volume_pct_above/100) × the
         `volume_sma_period` (10) day SMA of volume, i.e. >=25% above the
         10-day average.
      5. Bullish confirmation: today's close > previous day's open (OHLC
         compare rule from the screener config).
    """
    min_len = lookback + max(freshness_days, volume_sma_period) + 2
    if len(df) < min_len:
        return {'stage': 'insufficient_data'}

    today = df.iloc[-1]
    prev = df.iloc[-2]

    # Highest high over the lookback window, excluding today
    window = df.iloc[-(lookback + 1):-1]
    prev_high = window['high'].max()

    # Freshness: was this level already closed above in the last freshness_days?
    recent_closes = df.iloc[-(freshness_days + 1):-1]['close']
    already_broken_recently = bool((recent_closes > prev_high).any())

    breakout_close = bool(today['close'] > prev_high)

    magnitude_pct_actual = ((today['close'] - prev_high) / prev_high * 100) if prev_high else None
    magnitude_ok = magnitude_pct_actual is not None and magnitude_pct_actual >= magnitude_pct

    vol_sma = df['volume'].iloc[-(volume_sma_period + 1):-1].mean()
    volume_ratio = (today['volume'] / vol_sma) if vol_sma else None
    volume_ok = vol_sma > 0 and today['volume'] >= vol_sma * (1 + volume_pct_above / 100)

    bullish_ok = bool(today['close'] > prev['open'])

    breakout_today = bool(
        breakout_close and (not already_broken_recently) and magnitude_ok and volume_ok and bullish_ok
    )

    return {
        'stage': 'fresh_breakout' if breakout_today else 'forming',
        'base_high': round(float(prev_high), 2),
        'today_close': round(float(today['close']), 2),
        'magnitude_pct': round(magnitude_pct_actual, 2) if magnitude_pct_actual is not None else None,
        'today_volume_vs_avg': round(volume_ratio, 2) if volume_ratio is not None else None,
        'already_broken_recently': already_broken_recently,
        'bullish_confirm': bullish_ok,
        'breakout_today': breakout_today,
    }


def track_breakout_stage(prior_status: dict | None, today_result: dict,
                          today_date, today_close: float, base_high: float,
                          fresh_window_days: int = 5) -> dict:
    """
    Day-over-day state machine. Call this once per stock per day AFTER
    detect_base_breakout(), passing in yesterday's stored record
    (prior_status) from your database (e.g. Supabase 'breakout_tracking' table).

    Stages: forming -> fresh_breakout -> climbing -> played_out

    prior_status shape (what you'd load from DB):
        {'stage': ..., 'breakout_date': ..., 'base_high': ...}
    Returns the new record to save back to DB.
    """
    # No prior breakout on record, and no breakout today -> still forming
    if prior_status is None or prior_status.get('stage') in (None, 'forming'):
        if today_result.get('breakout_today'):
            return {'stage': 'fresh_breakout', 'breakout_date': today_date,
                    'base_high': base_high, 'last_close': today_close}
        return {'stage': 'forming', 'breakout_date': None,
                'base_high': today_result.get('base_high'), 'last_close': today_close}

    # Had a breakout before -> check if still holding above base_high
    ref_high = prior_status.get('base_high', base_high)
    still_holding = today_close > ref_high

    if not still_holding:
        return {'stage': 'played_out', 'breakout_date': prior_status.get('breakout_date'),
                'base_high': ref_high, 'last_close': today_close}

    days_since = (pd.to_datetime(today_date) - pd.to_datetime(prior_status['breakout_date'])).days
    new_stage = 'fresh_breakout' if days_since <= fresh_window_days else 'climbing'

    return {'stage': new_stage, 'breakout_date': prior_status['breakout_date'],
            'base_high': ref_high, 'last_close': today_close}


# ---------------------------------------------------------------------------
# 2. RS RANK (multi-period, weighted, vs Nifty 50)
# ---------------------------------------------------------------------------

def calculate_return(df: pd.DataFrame, days: int) -> float | None:
    if len(df) < days + 1:
        return None
    past_close = df['close'].iloc[-(days + 1)]
    today_close = df['close'].iloc[-1]
    if past_close == 0:
        return None
    return ((today_close - past_close) / past_close) * 100


def calculate_rs_score(stock_df: pd.DataFrame, nifty50_df: pd.DataFrame,
                        weights: dict = None) -> dict:
    """
    Weighted RS score vs Nifty 50, using 1M (~21 trading days),
    3M (~63 days), 6M (~126 days) returns.
    Default weights: 1M 40%, 3M 35%, 6M 25% (recency-tilted).
    Returns raw weighted excess-return score; caller ranks all stocks'
    scores into a 1-99 percentile RS Rank across the scan universe.
    """
    weights = weights or {'1m': 0.40, '3m': 0.35, '6m': 0.25}
    periods = {'1m': 21, '3m': 63, '6m': 126}

    stock_returns, nifty_returns, excess = {}, {}, {}
    for label, days in periods.items():
        sr = calculate_return(stock_df, days)
        nr = calculate_return(nifty50_df, days)
        stock_returns[label] = sr
        nifty_returns[label] = nr
        excess[label] = (sr - nr) if (sr is not None and nr is not None) else None

    valid = {k: v for k, v in excess.items() if v is not None}
    if not valid:
        return {'rs_score': None, 'excess_returns': excess}

    total_weight = sum(weights[k] for k in valid)
    weighted_score = sum(excess[k] * weights[k] for k in valid) / total_weight

    return {
        'rs_score': round(weighted_score, 2),
        'excess_returns': {k: (round(v, 2) if v is not None else None) for k, v in excess.items()},
        'stock_returns': {k: (round(v, 2) if v is not None else None) for k, v in stock_returns.items()},
    }


def rank_universe_rs(rs_scores: dict) -> dict:
    """
    rs_scores: {symbol: rs_score} for the whole scan universe (Nifty 50 + 500).
    Returns {symbol: rs_rank} where rank is a 1-99 percentile (99 = strongest).
    """
    symbols = [s for s, v in rs_scores.items() if v is not None]
    if not symbols:
        return {}
    series = pd.Series({s: rs_scores[s] for s in symbols})
    pct_rank = series.rank(pct=True) * 98 + 1   # scale to 1-99
    return {s: int(round(r)) for s, r in pct_rank.items()}


# ---------------------------------------------------------------------------
# 3. MA CROSSOVER (EMA 50/200 — Golden/Death Cross)
# ---------------------------------------------------------------------------

def detect_ma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> dict:
    """EMA-based (per your correction — not SMA)."""
    if len(df) < slow + 2:
        return {'signal': 'insufficient_data'}

    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()

    today_fast, today_slow = df['ema_fast'].iloc[-1], df['ema_slow'].iloc[-1]
    prev_fast, prev_slow = df['ema_fast'].iloc[-2], df['ema_slow'].iloc[-2]

    crossed_up = prev_fast <= prev_slow and today_fast > today_slow
    crossed_down = prev_fast >= prev_slow and today_fast < today_slow

    if crossed_up:
        signal = 'golden_cross'
    elif crossed_down:
        signal = 'death_cross'
    elif today_fast > today_slow:
        signal = 'bullish'          # already crossed earlier, still above
    else:
        signal = 'bearish'

    # 'approaching' = within 1% of each other, not yet crossed
    gap_pct = abs(today_fast - today_slow) / today_slow * 100 if today_slow else None
    if signal in ('bullish', 'bearish') and gap_pct is not None and gap_pct <= 1.0:
        signal = 'approaching'

    return {
        'signal': signal,
        'ema_fast': round(today_fast, 2),
        'ema_slow': round(today_slow, 2),
        'gap_pct': round(gap_pct, 2) if gap_pct is not None else None,
    }


# ---------------------------------------------------------------------------
# 4. VOLUME SHOCKER
# ---------------------------------------------------------------------------

def detect_volume_shocker(df: pd.DataFrame, avg_days: int = 20, multiple: float = 3.0) -> dict:
    if len(df) < avg_days + 1:
        return {'is_shocker': False}

    avg_vol = df['volume'].iloc[-(avg_days + 1):-1].mean()
    today_vol = df['volume'].iloc[-1]
    ratio = today_vol / avg_vol if avg_vol else None
    prev_close = df['close'].iloc[-2]
    today_close = df['close'].iloc[-1]
    change_pct = ((today_close - prev_close) / prev_close) * 100 if prev_close else None

    return {
        'is_shocker': bool(ratio is not None and ratio >= multiple),
        'volume_ratio': round(ratio, 2) if ratio is not None else None,
        'today_volume': int(today_vol),
        'avg_volume': round(avg_vol, 0) if avg_vol else None,
        'change_pct': round(change_pct, 2) if change_pct is not None else None,
    }
