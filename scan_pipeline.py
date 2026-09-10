"""
MITS 360 Free Scanner — Daily Scan Pipeline
==============================================
Entry point run once per day by GitHub Actions (after market close).
Orchestrates: Angel One login -> fetch Nifty 50+500 EOD data -> run all
4 scanners -> write results to Supabase.

Environment variables required (set as GitHub Actions Secrets):
    ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
    SUPABASE_URL, SUPABASE_SERVICE_KEY   (service key — bypasses RLS, backend only)
"""

import os
import time
import datetime as dt
from supabase import create_client

from angel_connector import AngelOneConnector
from instrument_master import build_scan_universe, load_nifty500_symbols
from scanner_logic import (
    detect_base_breakout, track_breakout_stage,
    calculate_rs_score, rank_universe_rs,
    detect_ma_crossover, detect_volume_shocker
)

RATE_LIMIT_DELAY = 1.0    # seconds between candle fetches (Angel One: 3/sec, 180/min documented)
MAX_RETRIES = 3


def fetch_with_retry(connector, symbol, token, exchange):
    """Fetches one stock's candles, retrying with backoff on rate-limit-like
    failures instead of giving up immediately (Angel One's server has
    reported intermittent throttling even under the documented limit)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return connector.get_daily_candles(token, exchange)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            backoff = RATE_LIMIT_DELAY * (3 ** attempt)   # 3s, 9s, 27s
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {symbol}: {e} — waiting {backoff:.0f}s")
            time.sleep(backoff)


def get_supabase():
    url = os.environ['SUPABASE_URL']
    key = os.environ['SUPABASE_SERVICE_KEY']
    return create_client(url, key)


def fetch_nifty50_benchmark(connector: AngelOneConnector):
    """Nifty 50 index token on Angel One's NSE indices segment is 99926000."""
    return connector.get_daily_candles(symbol_token='99926000', exchange='NSE')


def run_daily_scan():
    today = dt.date.today().isoformat()
    print(f"=== MITS 360 Free Scanner — daily run {today} ===")

    supabase = get_supabase()

    connector = AngelOneConnector()
    connector.login()
    print("Angel One session started.")

    universe = build_scan_universe()
    nifty50_df = fetch_nifty50_benchmark(connector)

    # Pull existing breakout_tracking state once, so we don't hit the DB
    # per-stock in the loop.
    prior_rows = supabase.table('breakout_tracking').select('*').execute().data
    prior_by_symbol = {row['symbol']: row for row in prior_rows}

    rs_scores_raw = {}
    per_stock_data = {}   # symbol -> everything computed, before RS rank is known

    for inst in universe:
        symbol = inst['symbol']
        try:
            df = fetch_with_retry(connector, symbol, inst['token'], inst['exchange'])
        except Exception as e:
            print(f"[SKIP] {symbol}: fetch failed — {e}")
            time.sleep(RATE_LIMIT_DELAY)
            continue
        time.sleep(RATE_LIMIT_DELAY)

        if len(df) < 60:
            print(f"[SKIP] {symbol}: not enough history ({len(df)} rows)")
            continue

        breakout_result = detect_base_breakout(df)
        rs_result = calculate_rs_score(df, nifty50_df)
        ma_result = detect_ma_crossover(df)
        volume_result = detect_volume_shocker(df)

        rs_scores_raw[symbol] = rs_result.get('rs_score')

        per_stock_data[symbol] = {
            'df': df,
            'breakout': breakout_result,
            'rs': rs_result,
            'ma': ma_result,
            'volume': volume_result,
        }

    # Rank RS scores across the whole universe now that we have them all
    rs_ranks = rank_universe_rs(rs_scores_raw)

    scan_rows = []
    tracking_upserts = []

    for symbol, data in per_stock_data.items():
        df = data['df']
        today_close = float(df['close'].iloc[-1])
        breakout = data['breakout']

        # --- Update breakout stage state machine ---
        prior = prior_by_symbol.get(symbol)
        if breakout.get('stage') not in ('insufficient_data',):
            new_status = track_breakout_stage(
                prior, breakout, today, today_close,
                base_high=breakout.get('base_high', today_close)
            )
            tracking_upserts.append({'symbol': symbol, **new_status,
                                      'breakout_date': str(new_status['breakout_date']) if new_status.get('breakout_date') else None})

        # --- Build today's snapshot row ---
        rs = data['rs']
        ma = data['ma']
        vol = data['volume']

        scan_rows.append({
            'symbol': symbol,
            'scan_date': today,
            'price': today_close,
            'base_range_pct': breakout.get('magnitude_pct'),
            'volume_vs_avg': breakout.get('today_volume_vs_avg'),
            'rs_score': rs.get('rs_score'),
            'rs_rank': rs_ranks.get(symbol),
            'return_1m': rs.get('stock_returns', {}).get('1m'),
            'return_3m': rs.get('stock_returns', {}).get('3m'),
            'return_6m': rs.get('stock_returns', {}).get('6m'),
            'ma_signal': ma.get('signal'),
            'ema_fast': ma.get('ema_fast'),
            'ema_slow': ma.get('ema_slow'),
            'is_volume_shocker': vol.get('is_shocker', False),
            'volume_ratio': vol.get('volume_ratio'),
            'change_pct': vol.get('change_pct'),
        })

    # --- Write to Supabase ---
    if tracking_upserts:
        supabase.table('breakout_tracking').upsert(tracking_upserts, on_conflict='symbol').execute()
        print(f"Upserted {len(tracking_upserts)} breakout_tracking rows.")

    if scan_rows:
        # Wipe yesterday's snapshot, insert today's (simple "latest only" table)
        supabase.table('latest_scan_results').delete().neq('symbol', '').execute()
        supabase.table('latest_scan_results').insert(scan_rows).execute()
        print(f"Inserted {len(scan_rows)} latest_scan_results rows.")

    print("=== Scan complete ===")


if __name__ == '__main__':
    run_daily_scan()
