"""Brent–WTI spread mean reversion (redesign of CorrReversion).

Why the original failed
-----------------------
The original went LONG the laggard when 30-day return-correlation fell below
0.6. On clean data, WTI/Brent correlation has not been below ~0.65 since 2016,
so it never traded. Its only profitable trade came from the April-2020
negative-price print. It was also long-only, i.e. a directional oil bet
that happened to be triggered by a correlation statistic.

The redesign
------------
Trade the thing that actually mean-reverts: the relative price of the two
grades. Spread s = log(Brent) - log(WTI), z = rolling z-score over `lookback`.

    z >  z_in  -> Brent rich: SHORT Brent, LONG WTI   (spread position -1)
    z < -z_in  -> Brent cheap: LONG Brent, SHORT WTI  (spread position +1)
    exit when z reverts past +/- z_out, or |z| >= z_stop (divergence stop),
    or after `max_hold` bars. After a stop / time exit, no re-entry until
    |z| < z_in again (stops fresh positions straight after a stop-out).

Legs are dollar-neutral (equal capital each), so the outright oil price is
hedged out. The P&L is the spread.

Optional regime filter `corr_min`: only open when the legs are still co-moving
(rolling return correlation >= corr_min). The idea is that a divergence while correlation
is intact is noise, but a divergence with broken correlation is more likely a
structural break (e.g. a supply shock that hits seaborne Brent only).

Symbol order is [WTI-like, Brent-like]; works for futures (CL=F, BZ=F) or
ETFs (USO, BNO).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import BaseStrategy, Panel, Signals


def spread_positions(z: np.ndarray, corr: np.ndarray, z_in: float, z_out: float, z_stop: float,
                     max_hold: int, corr_min: float) -> np.ndarray:
    n = len(z)
    pos = np.zeros(n)
    cur, held, blocked = 0, 0, False
    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            pos[i] = cur
            continue
        if blocked and abs(zi) < z_in:
            blocked = False
        if cur == 0:
            ok = (not blocked) and (corr_min <= -1 or corr[i] >= corr_min)
            if ok and zi > z_in:
                cur, held = -1, 0
            elif ok and zi < -z_in:
                cur, held = 1, 0
        else:
            held += 1
            if (cur == -1 and zi <= z_out) or (cur == 1 and zi >= -z_out):
                cur = 0
            elif abs(zi) >= z_stop or held >= max_hold:
                cur, blocked = 0, True
        pos[i] = cur
    return pos


class OilSpreadReversion(BaseStrategy):
    DEFAULT_SYMBOLS = ["CL=F", "BZ=F"]   # [WTI, Brent]
    DEFAULT_PARAMS = {
        "lookback": 40,     # bars for the z-score mean/std
        "z_in": 2.0,
        "z_out": 0.5,       # exit when z is back inside +/- z_out
        "z_stop": 4.0,      # divergence stop
        "max_hold": 30,     # bars
        "corr_window": 30,
        "corr_min": -1.0,   # -1 disables the regime filter
    }
    WARMUP_BARS = 150

    def __init__(self, name="Oil Spread Reversion", symbols=None, **params):
        super().__init__(name, symbols, **params)
        if len(self.symbols) != 2:
            raise ValueError("OilSpreadReversion needs [WTI, Brent]")

    def indicators(self, panel: Panel) -> pd.DataFrame:
        a, b = self.symbols
        p = self.params
        close = panel["Close"]
        s = np.log(close[b]) - np.log(close[a])
        mu = s.rolling(p["lookback"]).mean()
        sd = s.rolling(p["lookback"]).std()
        r = np.log(close).diff()
        corr = r[a].rolling(p["corr_window"]).corr(r[b])
        z = (s - mu) / sd
        pos = spread_positions(z.to_numpy(), corr.to_numpy(), p["z_in"], p["z_out"], p["z_stop"],
                               p["max_hold"], p["corr_min"])
        return pd.DataFrame({"spread": s, "z": z, "corr": corr, "pos": pos}, index=close.index)

    def signals(self, panel: Panel, ind: pd.DataFrame) -> Signals:
        a, b = self.symbols
        pos = ind["pos"]
        long_b, short_b = pos == 1, pos == -1          # Brent leg follows the spread
        long_a, short_a = short_b, long_b               # WTI leg is the mirror
        df = lambda xa, xb: pd.DataFrame({a: xa, b: xb})
        return Signals(entries=df(long_a, long_b), exits=df(~long_a, ~long_b),
                       short_entries=df(short_a, short_b), short_exits=df(~short_a, ~short_b))
