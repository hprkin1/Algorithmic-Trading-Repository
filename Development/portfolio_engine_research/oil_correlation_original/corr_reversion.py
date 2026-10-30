"""RETIRED (2026-10-30) - kept for reference only; do not use as a live sleeve.
Reason: on clean data WTI/Brent correlation never drops below ~0.65, so it never trades; its only
historical profit came from the April-2020 negative-price print. The idea was generalised and
tested in strategies/pairs_reversion.py (reports/pairs/).

Correlation mean-reversion on a pair (port of Portfolio/CorrArb_Class.py).

Logic is unchanged from the original:
    when rolling return-correlation < entry_corr_low, go LONG whichever leg
    has the lower trailing `rolling_window` return (the laggard);
    exit both legs when correlation > exit_corr_high. Long only.

Change: the original shifted its entry inputs by one bar but not its exits,
then filled on the same close. Under the engine contract the strategy uses
bar-t information and the engine executes on bar t+1, for entries AND exits.
"""
from __future__ import annotations

import pandas as pd

from engine.strategy import BaseStrategy, Panel, Signals


class CorrReversion(BaseStrategy):
    DEFAULT_SYMBOLS = ["CL=F", "BZ=F"]
    DEFAULT_PARAMS = {"rolling_window": 30, "entry_corr_low": 0.6, "exit_corr_high": 0.9}
    WARMUP_BARS = 60

    def __init__(self, name="Oil Corr Reversion", symbols=None, **params):
        super().__init__(name, symbols, **params)
        if len(self.symbols) != 2:
            raise ValueError("CorrReversion needs exactly two symbols")

    def indicators(self, panel: Panel) -> pd.DataFrame:
        s1, s2 = self.symbols
        w = self.params["rolling_window"]
        close = panel["Close"]
        r1, r2 = close[s1].pct_change(), close[s2].pct_change()
        return pd.DataFrame({
            "Rolling_Corr": r1.rolling(w).corr(r2),
            "R1_window": close[s1].pct_change(w),
            "R2_window": close[s2].pct_change(w),
        })

    def signals(self, panel: Panel, ind: pd.DataFrame) -> Signals:
        s1, s2 = self.symbols
        low_corr = ind["Rolling_Corr"] < self.params["entry_corr_low"]
        high_corr = ind["Rolling_Corr"] > self.params["exit_corr_high"]
        entries = pd.DataFrame({s1: low_corr & (ind["R1_window"] < ind["R2_window"]),
                                s2: low_corr & (ind["R2_window"] < ind["R1_window"])})
        exits = pd.DataFrame({s1: high_corr, s2: high_corr})
        none = pd.DataFrame(False, index=ind.index, columns=self.symbols)
        return Signals(entries, exits, none, none.copy())
