"""Systematic pairs portfolio (v2): choose pairs on year N, trade them in year N+1.

Each calendar year Y (formation = the previous calendar year, data up to its last close):
  1. For every pair inside a group with complete formation data:
       hedge ratio  beta from OLS  log(A) = alpha + beta*log(B)   (hedge="dollar" -> beta = 1)
       spread       s = log(A) - beta*log(B) - alpha
       ADF p-value of s (lag 1), half-life from  ds = lambda*s_lag  ->  hl = -ln2/lambda
       distance     sum of squared differences of the two normalised price paths (Gatev)
  2. Keep pairs with beta > 0, ADF p < adf_p and 0 < hl <= max_half_life (None = no hl filter).
  3. Rank by distance, take the best n_pairs with no symbol in more than max_per_symbol pairs.
Trading in year Y, per pair (signals at the close, filled next open by the engine):
  z = (s - mean_formation) / std_formation
  enter |z| >= z_in (short the rich leg, long the cheap leg); exit when z crosses z_out;
  stop at |z| >= z_stop; time limit = fixed_hold or clip(hold_mult*hl, hold_min, hold_max) days.
  After a stop/time exit, no re-entry until |z| < z_in.
  corr_exit: if the corr_window-day return correlation drops below formation mean - corr_k*sd,
  exit and block until it recovers above formation mean - 1*sd (relationship looks broken).
  Pairs not selected for the new year get weight 0 from its first day (positions closed).
Sizing: n_pairs equal slots; a slot's legs are split 1 : beta (dollar-hedged in log terms).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from engine.strategy import BaseStrategy, Panel, Signals
from strategies.pairs_universe import all_groups, all_symbols


def _adf_p(s: np.ndarray) -> float:
    from statsmodels.tsa.stattools import adfuller
    try:
        return float(adfuller(s, maxlag=1, autolag=None, regression="c")[1])
    except Exception:
        return 1.0


def _half_life(s: np.ndarray) -> float:
    ds, lag = np.diff(s), s[:-1] - s[:-1].mean()
    denom = (lag * lag).sum()
    if denom <= 0:
        return math.inf
    lam = (lag * ds).sum() / denom
    return -math.log(2) / lam if lam < 0 else math.inf


def run_pair_state(z, corr, z_in, z_out, z_stop, max_hold, corr_exit, corr_lo, corr_ok):
    n = len(z)
    pos = np.zeros(n)
    cur, held, blocked, broken = 0, 0, False, False
    for i in range(n):
        zi, ci = z[i], corr[i]
        if np.isnan(zi):
            pos[i] = cur
            continue
        if blocked and abs(zi) < z_in:
            blocked = False
        if corr_exit and not np.isnan(ci):
            if broken and ci >= corr_ok:
                broken = False
            if ci < corr_lo:
                broken = True
        if cur != 0:
            held += 1
            if corr_exit and broken:
                cur, blocked = 0, True
            elif (cur == -1 and zi <= z_out) or (cur == 1 and zi >= -z_out):
                cur = 0
            elif abs(zi) >= z_stop or held >= max_hold:
                cur, blocked = 0, True
        if cur == 0 and not blocked and not (corr_exit and broken):
            if zi >= z_in:
                cur, held = -1, 0
            elif zi <= -z_in:
                cur, held = 1, 0
        pos[i] = cur
    return pos


class PairsPortfolio(BaseStrategy):
    DEFAULT_SYMBOLS: list[str] = []
    DEFAULT_PARAMS = {
        "groups": None,           # {group: [symbols]}; None = strategies.pairs_universe.all_groups()
        "n_pairs": 20,
        "max_per_symbol": 3,
        "min_formation_days": 200,
        "hedge": "beta",          # "beta" | "dollar"
        "adf_p": 0.10,
        "max_half_life": 30,      # None = no half-life filter
        "z_in": 2.0,
        "z_out": 0.0,
        "z_stop": 4.0,
        "hold_mult": 3.0,
        "hold_min": 10,
        "hold_max": 60,
        "fixed_hold": None,       # e.g. 30 -> overrides the half-life-based limit
        "corr_exit": True,
        "corr_window": 20,
        "corr_k": 2.0,
    }
    WARMUP_BARS = 300

    def __init__(self, name="Pairs Portfolio v2", symbols=None, **params):
        groups = params.get("groups") or all_groups()
        symbols = symbols or sorted({s for g in groups.values() for s in g})
        super().__init__(name, symbols, **params)
        self.params["groups"] = groups
        self.selection: list[dict] = []

    # ---- formation ----
    def _select(self, close: pd.DataFrame, form_idx: pd.Index) -> list[dict]:
        p = self.params
        f = close.loc[form_idx]
        cands, seen = [], set()
        for g, syms in p["groups"].items():
            syms = [s for s in syms if s in f.columns]
            for a, b in itertools.combinations(sorted(set(syms)), 2):
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                pa, pb = f[a].to_numpy(), f[b].to_numpy()
                if np.isnan(pa).any() or np.isnan(pb).any() or len(pa) < p["min_formation_days"]:
                    continue
                la, lb = np.log(pa), np.log(pb)
                if p["hedge"] == "beta":
                    beta, alpha = np.polyfit(lb, la, 1)
                else:
                    beta, alpha = 1.0, float((la - lb).mean())
                if beta <= 0:
                    continue
                s = la - beta * lb - alpha
                hl = _half_life(s)
                if p["max_half_life"] is not None and not (0 < hl <= p["max_half_life"]):
                    continue
                adf = _adf_p(s)
                if adf >= p["adf_p"]:
                    continue
                ssd = float(((pa / pa[0] - pb / pb[0]) ** 2).sum())
                ra, rb = np.diff(la), np.diff(lb)
                rc = pd.Series(ra).rolling(p["corr_window"]).corr(pd.Series(rb)).dropna()
                cands.append(dict(group=g, a=a, b=b, beta=float(beta), alpha=float(alpha), hl=hl, adf=adf, ssd=ssd,
                                  mu=float(s.mean()), sd=float(s.std()), corr_mu=float(rc.mean()),
                                  corr_sd=float(rc.std()) if len(rc) > 2 else 0.0))
        cands.sort(key=lambda d: d["ssd"])
        chosen, count = [], {}
        for d in cands:
            if count.get(d["a"], 0) >= p["max_per_symbol"] or count.get(d["b"], 0) >= p["max_per_symbol"]:
                continue
            chosen.append(d)
            count[d["a"]] = count.get(d["a"], 0) + 1
            count[d["b"]] = count.get(d["b"], 0) + 1
            if len(chosen) >= p["n_pairs"]:
                break
        return chosen

    def indicators(self, panel: Panel) -> pd.DataFrame:
        p = self.params
        close = panel["Close"][self.symbols]
        logc = np.log(close)
        rets = logc.diff()
        idx = close.index
        weights = pd.DataFrame(0.0, index=idx, columns=self.symbols)
        n_open = pd.Series(0.0, index=idx)
        self.selection = []
        years = sorted(set(idx.year))
        for y in years[1:]:
            form_idx = idx[idx.year == y - 1]
            trade_idx = idx[idx.year == y]
            if len(form_idx) < p["min_formation_days"] or len(trade_idx) == 0:
                continue
            for d in self._select(close, form_idx):
                a, b = d["a"], d["b"]
                s = (logc.loc[trade_idx, a] - d["beta"] * logc.loc[trade_idx, b] - d["alpha"]).to_numpy()
                z = (s - d["mu"]) / d["sd"] if d["sd"] > 0 else np.full(len(s), np.nan)
                corr = rets[a].rolling(p["corr_window"]).corr(rets[b]).loc[trade_idx].to_numpy()
                if p["fixed_hold"]:
                    hold = int(p["fixed_hold"])
                elif math.isfinite(d["hl"]) and d["hl"] > 0:
                    hold = int(np.clip(round(p["hold_mult"] * d["hl"]), p["hold_min"], p["hold_max"]))
                else:
                    hold = 30
                pos = run_pair_state(z, corr, p["z_in"], p["z_out"], p["z_stop"], hold, p["corr_exit"],
                                     d["corr_mu"] - p["corr_k"] * d["corr_sd"], d["corr_mu"] - d["corr_sd"])
                slot = 1.0 / p["n_pairs"]
                wa, wb = slot / (1 + d["beta"]), slot * d["beta"] / (1 + d["beta"])
                weights.loc[trade_idx, a] += pos * wa
                weights.loc[trade_idx, b] -= pos * wb
                n_open.loc[trade_idx] += (pos != 0)
                self.selection.append({"year": y, **{k: d[k] for k in ("group", "a", "b", "beta", "hl", "adf", "ssd")},
                                       "hold": hold, "trades": int((np.diff(np.r_[0, pos]) != 0).sum() // 2 + (pos[-1] != 0)),
                                       "days_open": int((pos != 0).sum())})
        self._weights = weights
        return pd.DataFrame({"open_pairs": n_open, "gross": weights.abs().sum(axis=1)})

    def signals(self, panel: Panel, ind: pd.DataFrame) -> Signals:
        return Signals(target=self._weights.reindex(ind.index).fillna(0.0))
