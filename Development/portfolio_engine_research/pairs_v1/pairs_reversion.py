"""Correlation-triggered pairs reversion (generalises Henry's CorrArb_Class idea).

Idea: two economically linked assets that drift apart tend to come back together;
the original version used a WTI/Brent correlation drop as the trigger.

Measures (daily, data to the close; the engine fills at the next open):
    gap z   = z-score of log(A) - log(B) over `lookback` days
    corr z  = z-score of the `lookback`-day return correlation vs its own past `corr_hist` days
              (<= corr_trigger means "unusually low for this pair")

Versions (param `version`):
    "A" textbook:   enter |gap z| >= z_in;                          long cheap leg, short rich leg (50/50)
    "B" hedged:     enter |gap z| >= z_in AND corr z <= corr_trigger; same position as A
    "C" original:   enter corr z <= corr_trigger; long the laggard (lower `lookback`-day return) 100%
Exits:
    A/B: gap z back through z_out (0 = the mean), or |gap z| >= z_stop, or max_hold days.
         After a stop/time exit, no re-entry until |gap z| < z_in again.
    C:   corr z back to >= 0, or max_hold days. After a time exit, wait for a fresh trigger.

Symbols: [A, B]. Output: target weights per leg (fraction of the pair's capital).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import BaseStrategy, Panel, Signals


def pair_positions_ab(gz, cz, use_corr, z_in, z_out, z_stop, max_hold, corr_trigger):
    """Spread position per bar: +1 = long A / short B, -1 = short A / long B, 0 = flat."""
    n = len(gz)
    pos = np.zeros(n)
    cur, held, blocked = 0, 0, False
    for i in range(n):
        g, c = gz[i], cz[i]
        if np.isnan(g):
            pos[i] = cur
            continue
        if blocked and abs(g) < z_in:
            blocked = False
        if cur == 0:
            corr_ok = (not use_corr) or (not np.isnan(c) and c <= corr_trigger)
            if not blocked and corr_ok:
                if g >= z_in:
                    cur, held = -1, 0       # A rich vs B: short A, long B
                elif g <= -z_in:
                    cur, held = 1, 0        # A cheap: long A, short B
        else:
            held += 1
            if (cur == -1 and g <= z_out) or (cur == 1 and g >= -z_out):
                cur = 0
            elif abs(g) >= z_stop or held >= max_hold:
                cur, blocked = 0, True
        pos[i] = cur
    return pos


def pair_positions_c(cz, lag_a, corr_trigger, max_hold):
    """Long-only laggard: +1 = long A, -1 = long B (NOT short), 0 = flat."""
    n = len(cz)
    pos = np.zeros(n)
    cur, held, armed = 0, 0, True
    for i in range(n):
        c = cz[i]
        if np.isnan(c):
            pos[i] = cur
            continue
        if not armed and c > corr_trigger:
            armed = True                     # trigger condition must clear before a fresh entry
        if cur == 0:
            if armed and c <= corr_trigger and not np.isnan(lag_a[i]):
                cur, held = (1 if lag_a[i] else -1), 0
        else:
            held += 1
            if c >= 0:
                cur = 0
            elif held >= max_hold:
                cur, armed = 0, False
        pos[i] = cur
    return pos


class PairsReversion(BaseStrategy):
    DEFAULT_SYMBOLS = ["XOM", "CVX"]
    DEFAULT_PARAMS = {
        "version": "B",
        "lookback": 60,
        "corr_hist": 252,
        "corr_trigger": -1.0,
        "z_in": 2.0,
        "z_out": 0.0,
        "z_stop": 4.0,
        "max_hold": 30,
    }
    WARMUP_BARS = 340

    def __init__(self, name=None, symbols=None, **params):
        symbols = symbols or self.DEFAULT_SYMBOLS
        super().__init__(name or f"Pairs {symbols[0]}/{symbols[1]}", symbols, **params)
        if len(self.symbols) != 2:
            raise ValueError("PairsReversion needs exactly two symbols")
        if self.params["version"] not in ("A", "B", "C"):
            raise ValueError("version must be A, B or C")
        self.WARMUP_BARS = self.params["lookback"] + self.params["corr_hist"] + 30

    def indicators(self, panel: Panel) -> pd.DataFrame:
        a, b = self.symbols
        p = self.params
        lb = p["lookback"]
        c = panel["Close"][[a, b]]
        spread = np.log(c[a]) - np.log(c[b])
        gz = (spread - spread.rolling(lb).mean()) / spread.rolling(lb).std()
        r = np.log(c).diff()
        corr = r[a].rolling(lb).corr(r[b])
        cz = (corr - corr.rolling(p["corr_hist"]).mean()) / corr.rolling(p["corr_hist"]).std()
        ret_a, ret_b = c[a].pct_change(lb), c[b].pct_change(lb)
        lag_a = (ret_a < ret_b).astype(float).where(ret_a.notna() & ret_b.notna())
        v = p["version"]
        if v in ("A", "B"):
            pos = pair_positions_ab(gz.to_numpy(), cz.to_numpy(), v == "B", p["z_in"], p["z_out"],
                                    p["z_stop"], p["max_hold"], p["corr_trigger"])
        else:
            pos = pair_positions_c(cz.to_numpy(), lag_a.to_numpy(), p["corr_trigger"], p["max_hold"])
        return pd.DataFrame({"gap_z": gz, "corr": corr, "corr_z": cz, "lag_a": lag_a, "pos": pos}, index=c.index)

    def signals(self, panel: Panel, ind: pd.DataFrame) -> Signals:
        a, b = self.symbols
        pos = ind["pos"]
        if self.params["version"] == "C":
            w = pd.DataFrame({a: (pos == 1).astype(float), b: (pos == -1).astype(float)})
        else:
            w = pd.DataFrame({a: 0.5 * pos, b: -0.5 * pos})
        return Signals(target=w)
