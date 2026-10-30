"""Multi-asset time-series trend following on ETFs (Moskowitz-Ooi-Pedersen 2012; Hurst-Ooi-Pedersen 2017).

On the first trading day of each month (at its close), for every asset:
    signal  = average of sign(1-month), sign(3-month), sign(12-month) total return   (-1 .. +1)
    weight  = signal * asset_vol_target / (realised vol over vol_lb days) / N
Long and short. If gross exposure (sum |w|) exceeds max_gross, all weights are scaled down.
Weights are only sent monthly (NaN on other days = hold, no trade); the engine fills at the next open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import BaseStrategy, Panel, Signals

TREND_UNIVERSE = {
    "equities": ["SPY", "IWM", "EFA", "EEM", "EWJ", "VNQ"],
    "bonds": ["IEF", "TLT", "TIP", "LQD"],
    "commodities": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "currencies": ["UUP", "FXE", "FXY"],
}


class MultiAssetTrend(BaseStrategy):
    DEFAULT_SYMBOLS = [s for g in TREND_UNIVERSE.values() for s in g]
    DEFAULT_PARAMS = {
        "lookbacks": (21, 63, 252),
        "vol_lb": 60,
        "asset_vol_target": 0.40,
        "max_gross": 1.0,
        "long_only": False,
    }
    WARMUP_BARS = 330

    def __init__(self, name="Multi-Asset Trend", symbols=None, **params):
        super().__init__(name, symbols, **params)

    def indicators(self, panel: Panel) -> pd.DataFrame:
        p = self.params
        c = panel["Close"][self.symbols]
        lc = np.log(c)
        sig = sum(np.sign(lc.diff(lb)) for lb in p["lookbacks"]) / len(p["lookbacks"])
        vol = lc.diff().rolling(p["vol_lb"]).std() * np.sqrt(252)
        w = sig * (p["asset_vol_target"] / vol) / len(self.symbols)
        if p["long_only"]:
            w = w.clip(lower=0)
        gross = w.abs().sum(axis=1)
        w = w.mul(np.minimum(1.0, p["max_gross"] / gross.replace(0, np.nan)).fillna(1.0), axis=0)
        ready = sig.notna().all(axis=1) & vol.notna().all(axis=1)
        month = pd.Series(c.index.to_period("M"), index=c.index)
        first_of_month = month != month.shift(1)   # causal: known on the day itself
        w = w.where(ready, 0.0)
        self._weights = w.where(first_of_month)    # NaN = no change on other days
        return pd.DataFrame({"gross": w.abs().sum(axis=1), "net": w.sum(axis=1)})

    def signals(self, panel: Panel, ind: pd.DataFrame) -> Signals:
        return Signals(target=self._weights.reindex(ind.index))
