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

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — used to make the base 'tightness' check dynamic
    per stock's own volatility instead of a fixed % for every stock."""
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_base_breakout(df: pd.DataFrame, min_base_days: int = 20,
                          atr_multiple: float = 2.5,
                          volume_mult: float = 1.5,
                          volume_avg_days: int = 50) -> dict:
    """
    Base = last `min_base_days` (20-25) candles before today.
    Base is 'tight' if (base_high - base_low) <= atr_multiple * ATR(14) — this
    is the dynamic, volatility-adjusted range check (no fixed % for all stocks).
    Breakout = today's close > base_high AND today's volume >= volume_mult * avg volume.

    Returns a dict describing today's classification for this stock:
    stage in {'forming', 'fresh_breakout', 'not_applicable'}
    plus the numbers used, so the caller can persist state for
    climbing/played_out tracking (which needs day-over-day history, not
    a single day's data — see track_breakout_stage below).
    """
    if len(df) < min_base_days + 15:
        return {'stage': 'insufficient_data'}

    df = df.copy()
    df['atr'] = calculate_atr(df)

    base = df.iloc[-(min_base_days + 1):-1]     # the base window, excluding today
    today = df.iloc[-1]

    base_high = base['high'].max()
    base_low = base['low'].min()
    base_range = base_high - base_low
    today_atr = df['atr'].iloc[-1]

    is_tight = today_atr > 0 and base_range <= (atr_multiple * today_atr)

    avg_vol = df['volume'].iloc[-(volume_avg_days + 1):-1].mean()
    vol_ok = avg_vol > 0 and today['volume'] >= (volume_mult * avg_vol)

    breakout_today = is_tight and (today['close'] > base_high) and vol_ok

    return {
        'stage': 'fresh_breakout' if breakout_today else 'forming',
        'base_high': round(base_high, 2),
        'base_low': round(base_low, 2),
        'base_range_pct': round((base_range / base_low) * 100, 2) if base_low else None,
        'is_tight_base': bool(is_tight),
        'today_close': round(today['close'], 2),
        'today_volume_vs_avg': round(today['volume'] / avg_vol, 2) if avg_vol else None,
        'breakout_today': bool(breakout_today),
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
