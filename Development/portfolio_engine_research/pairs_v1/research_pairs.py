"""Correlation-triggered pairs reversion: research run following the pre-declared protocol
(see reports/pairs/README.md and the strategy doc).

  DEV 2008-04-01..2015-12-31 · VAL 2016-01-01..2021-12-31 · HOLDOUT 2022-01-01..today (run once, last)
  Headline basket = 6 economically linked pairs, equal capital, never rebalanced.
  Controls QQQ/SPY and SPY/EWU reported separately.
  Rule 1: basket Sharpe >= 0.5 (after costs + borrow) in DEV and VAL.
  Rule 2: B replaces A only if Sharpe(B) >= Sharpe(A) + 0.05 in DEV and VAL.
  Rule 3: stability of the chosen version over lookback {40,60,90} x z_in {1.5,2,2.5}, DEV only.
  Rule 4: holdout once.  Rule 5: book impact if it passes.

Usage: python scripts/research_pairs.py [--holdout]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from engine import ExecConfig, backtest_strategy  # noqa: E402
from engine.data import load_panel  # noqa: E402
from engine.metrics import compute_metrics  # noqa: E402
from strategies import PairsReversion  # noqa: E402

PERIODS = {"DEV": ("2008-04-01", "2015-12-31"), "VAL": ("2016-01-01", "2021-12-31"), "HOLDOUT": ("2022-01-01", None)}
CORE = [("EWA", "EWC"), ("XOM", "CVX"), ("KO", "PEP"), ("V", "MA"), ("HD", "LOW"), ("GLD", "SLV")]
CONTROLS = [("QQQ", "SPY"), ("SPY", "EWU")]
BORROW = 0.005
CAPITAL = 600_000
OUT = ROOT / "reports" / "pairs"
CFG = ExecConfig()
COLS = ["Total Return %", "CAGR %", "Ann. Vol %", "Sharpe", "Max DD %", "Calmar", "Exposure %", "Trades/Yr"]


def run_pair(pair, version, period, panel, **kw):
    start, end = PERIODS[period]
    p = {k: v.loc[:end, list(pair)] for k, v in panel.items()} if end else {k: v[list(pair)] for k, v in panel.items()}
    p = {k: v.dropna() for k, v in p.items()}
    s = PairsReversion(symbols=list(pair), version=version, **kw)
    r = backtest_strategy(s, CAPITAL / len(CORE), start, config=CFG, panel=p)
    # borrow cost on the short leg (not modelled in the engine): annual rate x short weight, daily
    w = r.signals.target.shift(1).reindex(r.equity.index).fillna(0)   # weight held during each day
    short_w = w.clip(upper=0).abs().sum(axis=1)
    ret = r.equity.pct_change().fillna(0) - short_w * BORROW / 252
    eq = (1 + ret).cumprod() * r.capital
    trades = int((r.signals.target.diff().abs().sum(axis=1) > 0).sum() / 2)
    return r, eq, trades


def basket(pairs, version, period, panel, **kw):
    eqs, per_pair, trades = {}, {}, 0
    for pr in pairs:
        r, eq, n = run_pair(pr, version, period, panel, **kw)
        name = f"{pr[0]}/{pr[1]}"
        eqs[name] = eq
        m = compute_metrics(eq, exposure=r.metrics.get("Exposure %", 0) / 100)
        m["Trades/Yr"] = n / max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        per_pair[name] = m
        trades += n
    df = pd.DataFrame(eqs).ffill()
    df = df.fillna(CAPITAL / len(CORE))
    book = df.sum(axis=1)
    m = compute_metrics(book)
    m["Trades/Yr"] = trades / max((book.index[-1] - book.index[0]).days / 365.25, 1e-9)
    m["Exposure %"] = np.mean([v.get("Exposure %", 0) for v in per_pair.values()])
    return book, m, per_pair, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="run the holdout (only after the DEV/VAL decision)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    syms = sorted({s for pr in CORE + CONTROLS for s in pr})
    panel = load_panel(syms, "2007-01-01", join="outer")

    res, curves = {}, {}
    for v in ("A", "B", "C"):
        for per in ("DEV", "VAL"):
            book, m, pp, df = basket(CORE, v, per, panel)
            res[(v, per)] = (m, pp)
            curves[(v, per)] = book
    tbl = pd.DataFrame({f"{v} {per}": res[(v, per)][0] for v in "ABC" for per in ("DEV", "VAL")}).T.reindex(columns=COLS)
    print(tbl.astype(float).round(2).to_string())

    sh = lambda v, per: res[(v, per)][0]["Sharpe"]
    rule2 = all(sh("B", per) >= sh("A", per) + 0.05 for per in ("DEV", "VAL"))
    chosen = "B" if rule2 else "A"
    rule1 = {v: all(sh(v, per) >= 0.5 for per in ("DEV", "VAL")) for v in "ABC"}
    print("rule2 B beats A:", rule2, "-> chosen", chosen, "| rule1 pass:", rule1)

    per_pair = {}
    for v in "ABC":
        for per in ("DEV", "VAL"):
            for name, m in res[(v, per)][1].items():
                per_pair.setdefault(name, {})[f"{v} {per}"] = m["Sharpe"]
    ctrl = {}
    for v in "ABC":
        for per in ("DEV", "VAL"):
            for pr in CONTROLS:
                r, eq, n = run_pair(pr, v, per, panel)
                ctrl.setdefault(f"{pr[0]}/{pr[1]}", {})[f"{v} {per}"] = compute_metrics(eq)["Sharpe"]

    # Rule 3: stability, DEV only
    grid = pd.DataFrame(index=[40, 60, 90], columns=[1.5, 2.0, 2.5], dtype=float)
    for lb in grid.index:
        for zi in grid.columns:
            grid.loc[lb, zi] = basket(CORE, chosen, "DEV", panel, lookback=int(lb), z_in=float(zi))[1]["Sharpe"]
    print("stability (DEV, version", chosen, ")\n", grid.round(2))

    out = {"table": tbl.astype(float).round(3).to_dict(orient="index"), "rule1": rule1, "rule2": rule2,
           "chosen": chosen, "per_pair_sharpe": per_pair, "controls_sharpe": ctrl,
           "stability_dev": grid.round(3).to_dict()}

    if args.holdout:
        hold = {}
        for v in "ABC":
            book, m, pp, df = basket(CORE, v, "HOLDOUT", panel)
            hold[v] = {"basket": {k: m.get(k) for k in COLS}, "pairs": {n: x["Sharpe"] for n, x in pp.items()}}
            curves[(v, "HOLDOUT")] = book
            print("HOLDOUT", v, {k: round(float(m[k]), 2) for k in ("CAGR %", "Sharpe", "Max DD %")})
        for pr in CONTROLS:
            for v in "ABC":
                r, eq, n = run_pair(pr, v, "HOLDOUT", panel)
                ctrl[f"{pr[0]}/{pr[1]}"][f"{v} HOLDOUT"] = compute_metrics(eq)["Sharpe"]
        out["holdout"] = hold
        # current signals
        today = {}
        for pr in CORE + CONTROLS:
            p = {k: v[list(pr)].dropna() for k, v in panel.items()}
            ind, sig = PairsReversion(symbols=list(pr), version=chosen).run(p)
            last = ind.iloc[-1]
            today[f"{pr[0]}/{pr[1]}"] = {"gap_z": round(float(last.gap_z), 2), "corr_z": round(float(last.corr_z), 2),
                                         "position": {1: f"long {pr[0]} / short {pr[1]}", -1: f"short {pr[0]} / long {pr[1]}",
                                                      0: "flat"}[int(last.pos)], "date": str(ind.index[-1].date())}
        out["today"] = today
    out["controls_sharpe"] = ctrl
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    full = {}
    for v in "ABC":
        parts = [curves[(v, per)] for per in ("DEV", "VAL", "HOLDOUT") if (v, per) in curves]
        rets = pd.concat([c.pct_change().dropna() for c in parts])
        full[v] = (1 + rets).cumprod()
    pd.DataFrame(full).to_csv(OUT / "basket_growth.csv")
    print("saved")


if __name__ == "__main__":
    main()
