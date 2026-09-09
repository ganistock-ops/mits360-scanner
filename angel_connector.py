"""
MITS 360 Free Scanner — Angel One SmartAPI Connector
=======================================================
Handles login/session + EOD historical candle fetching for the scan universe
(Nifty 50 + Nifty 500). Runs on the Render.com backend — NOT in this sandbox
(no network access to Angel One's servers from here).

SETUP (on Render.com, as environment variables — never hardcode/commit these):
    ANGEL_API_KEY        - from SmartAPI developer portal
    ANGEL_CLIENT_CODE     - your Angel One client/login ID
    ANGEL_PASSWORD        - your Angel One MPIN/password
    ANGEL_TOTP_SECRET     - the TOTP secret shown when you enable 2FA for
                             API access (NOT the 6-digit code itself — pyotp
                             generates the live code from this secret each run)

Install:
    pip install smartapi-python logzero pyotp websocket-client==1.6.4
"""

import os
import time
import datetime as dt
import pandas as pd
import pyotp
from SmartApi import SmartConnect


class AngelOneConnector:
    def __init__(self):
        self.api_key = os.environ['ANGEL_API_KEY']
        self.client_code = os.environ['ANGEL_CLIENT_CODE']
        self.password = os.environ['ANGEL_PASSWORD']
        self.totp_secret = os.environ['ANGEL_TOTP_SECRET']
        self.smart = SmartConnect(api_key=self.api_key)
        self.session = None

    def login(self):
        """Authenticates and starts a session. Call once per scan run —
        SmartAPI sessions are valid for the trading day."""
        totp_code = pyotp.TOTP(self.totp_secret).now()
        self.session = self.smart.generateSession(
            self.client_code, self.password, totp_code
        )
        if not self.session.get('status'):
            raise RuntimeError(f"Angel One login failed: {self.session.get('message')}")
        return self.session

    def get_daily_candles(self, symbol_token: str, exchange: str = 'NSE',
                           lookback_days: int = 400) -> pd.DataFrame:
        """
        Fetches daily OHLCV candles for one instrument.
        symbol_token = Angel One's numeric instrument token (NOT the trading
        symbol like 'TCS' — you need the token-symbol master file, see
        load_instrument_master() below).
        lookback_days ~400 gives enough history for the 200-EMA and 6M return
        calculations used by the scanners.
        """
        to_date = dt.datetime.now()
        from_date = to_date - dt.timedelta(days=int(lookback_days * 1.6))  # buffer for weekends/holidays

        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": "ONE_DAY",
            "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
            "todate": to_date.strftime("%Y-%m-%d 15:30"),
        }
        response = self.smart.getCandleData(params)
        if not response.get('status'):
            raise RuntimeError(f"Candle fetch failed for token {symbol_token}: {response.get('message')}")

        data = response['data']  # list of [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df.sort_values('date').reset_index(drop=True)


def scan_universe(connector: AngelOneConnector, instruments: list[dict],
                   rate_limit_delay: float = 0.35) -> dict:
    """
    instruments: list of {'symbol': 'TCS', 'token': '11536', 'exchange': 'NSE'}
                 — this is your Nifty 50 + Nifty 500 list with Angel One tokens.
    rate_limit_delay: seconds between calls, to stay well under Angel One's
                       API rate limits when looping ~500 stocks.

    Returns {symbol: DataFrame} for every stock successfully fetched.
    Failed fetches are skipped and logged, not fatal to the whole scan.
    """
    results = {}
    for inst in instruments:
        try:
            df = connector.get_daily_candles(inst['token'], inst.get('exchange', 'NSE'))
            results[inst['symbol']] = df
        except Exception as e:
            print(f"[SKIP] {inst['symbol']}: {e}")
        time.sleep(rate_limit_delay)
    return results


if __name__ == '__main__':
    # This block only runs on Render (or any machine with real network access
    # to Angel One + the ANGEL_* env vars set). It will NOT run in this
    # sandbox — shown here as the intended usage pattern.
    connector = AngelOneConnector()
    connector.login()
    print("Logged in. Fetching sample candles for RELIANCE (token 2885)...")
    df = connector.get_daily_candles(symbol_token='2885', exchange='NSE')
    print(df.tail())
