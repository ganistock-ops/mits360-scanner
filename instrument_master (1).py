"""
MITS 360 Free Scanner — Instrument Master / Token Resolver
==============================================================
Angel One's OpenAPIScripMaster.json gives every NSE instrument's token, but
it does NOT tell you which ~500 of those thousands of stocks make up the
Nifty 500 index — that list comes from NSE separately and is rebalanced
quarterly (Mar/Jun/Sep/Dec).

This module:
  1. Downloads Angel One's instrument master (token lookup for ANY NSE stock)
  2. Cross-references against a Nifty 500 symbol list you supply (CSV)
  3. Produces the {'symbol', 'token', 'exchange'} list that
     angel_connector.scan_universe() needs

WHERE TO GET THE NIFTY 500 LIST (update quarterly):
  https://niftyindices.com/IndexConstituent/ind_nifty500list.csv
  (Official NSE Indices site. Download the CSV, keep it in your repo,
  refresh it each quarter when NSE announces index reshuffles.)

Runs on Render.com (needs network access to margincalculator.angelbroking.com
and niftyindices.com — not available in this sandbox).
"""

import requests
import pandas as pd

INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

# NSE's website blocks automated/bot requests (like GitHub Actions runners),
# so we can't fetch this live every day. Instead: download it ONCE manually
# from https://niftyindices.com/IndexConstituent/ind_nifty500list.csv in a
# real browser, and commit it to the repo as nifty500_list.csv.
# Re-download and re-commit it whenever NSE rebalances the index
# (quarterly — March/June/September/December).
NIFTY500_LOCAL_FILE = "nifty500_list.csv"


def load_instrument_master() -> pd.DataFrame:
    """Downloads and parses Angel One's full instrument list, filtered to
    NSE cash-market equities only (symbol ends with '-EQ')."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    resp = requests.get(INSTRUMENT_MASTER_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    equities = df[(df['exch_seg'] == 'NSE') & (df['symbol'].str.endswith('-EQ'))].copy()
    equities['name_clean'] = equities['symbol'].str.replace('-EQ', '', regex=False)
    return equities[['token', 'symbol', 'name_clean', 'exch_seg']]


def load_nifty500_symbols() -> list[str]:
    """Reads the Nifty 500 constituent list from the LOCAL CSV in the repo
    (see NIFTY500_LOCAL_FILE note above — NSE blocks live automated fetches)."""
    df = pd.read_csv(NIFTY500_LOCAL_FILE)
    return df['Symbol'].str.strip().tolist()


def build_scan_universe() -> list[dict]:
    """
    Produces the final instrument list ready for angel_connector.scan_universe():
        [{'symbol': 'RELIANCE', 'token': '2885', 'exchange': 'NSE'}, ...]
    Any Nifty 500 symbol not found in Angel One's master (renamed/delisted/
    mismatch) is skipped and reported, not fatal.
    """
    master = load_instrument_master()
    nifty500 = load_nifty500_symbols()

    master_lookup = dict(zip(master['name_clean'], master['token']))

    universe, missing = [], []
    for symbol in nifty500:
        token = master_lookup.get(symbol)
        if token:
            universe.append({'symbol': symbol, 'token': str(token), 'exchange': 'NSE'})
        else:
            missing.append(symbol)

    if missing:
        print(f"[WARN] {len(missing)} Nifty 500 symbols not matched in Angel One master: {missing}")

    print(f"Scan universe built: {len(universe)} / {len(nifty500)} Nifty 500 stocks resolved to tokens.")
    return universe


if __name__ == '__main__':
    # Runs only where network access to margincalculator.angelbroking.com and
    # niftyindices.com exists (Render.com) — not this sandbox.
    universe = build_scan_universe()
    print(universe[:5])
