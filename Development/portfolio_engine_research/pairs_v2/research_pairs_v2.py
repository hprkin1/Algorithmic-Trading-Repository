"""Pairs v2 research: systematic yearly pair selection, per the plan fixed in the strategy doc.

Bar (after 5+5 bp per side and 0.5%/yr borrow): Sharpe >= 0.5 over 2009-now AND positive in
2009-15, 2016-21, 2022-now; correlation to the BTC sleeves < 0.3.
Ablations (diagnostic; a simpler one is preferred only if >= v2 in all three periods):
equal dollars, no half-life filter, fixed 30-day hold, no correlation exit.

Usage: python scripts/research_pairs_v2.py
"""
from __future__ import annotations

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

from engine import ExecConfig, PortfolioManager, Sleeve, backtest_strategy  # noqa: E402
from engine.data import load_panel  # noqa: E402
from engine.metrics import compute_metrics  # noqa: E402
from strategies import ACMomentum, OBVMomentum, PairsPortfolio  # noqa: E402
from strategies.pairs_universe import ETF_GROUPS, STOCK_GROUPS, all_symbols  # noqa: E402

PERIODS = {"2009–15": ("2009-01-01", "2015-12-31"), "2016–21": ("2016-01-01", "2021-12-31"),
           "2022–now": ("2022-01-01", None), "Full 2009–now": ("2009-01-01", None)}
SUB = ["2009–15", "2016–21", "2022–now"]
VARIANTS = {
    "v2 (all parts)": {},
    "Equal dollars (no β)": {"hedge": "dollar"},
    "No half-life filter": {"max_half_life": None},
    "Fixed 30-day hold": {"fixed_hold": 30},
    "No correlation exit": {"corr_exit": False},
}
BORROW, CAPITAL = 0.005, 1_000_000
OUT = ROOT / "reports" / "pairs_v2"


def prepared_panel():
    panel = load_panel(all_symbols(), "2007-01-01", join="outer")
    # forward-fill gaps; back-fill only before a listing (weights are 0 there: selection needs a full year)
    return {k: v.ffill().bfill() for k, v in panel.items()}


def run(panel, **kw):
    s = PairsPortfolio(**kw)
    r = backtest_strategy(s, CAPITAL, "2008-01-01", config=ExecConfig(), panel=panel)
    w = r.signals.target.shift(1).reindex(r.equity.index).fillna(0)
    short = w.clip(upper=0).abs().sum(axis=1)
    ret = r.equity.pct_change().fillna(0) - short * BORROW / 252
    eq = (1 + ret).cumprod() * CAPITAL
    return s, r, eq, w


def period_metrics(eq, w, sel):
    out = {}
    for name, (a, b) in PERIODS.items():
        e = eq.loc[a:b] if b else eq.loc[a:]
        m = compute_metrics(e)
        ww = w.loc[e.index]
        yrs = sel[(sel.year >= e.index[0].year) & (sel.year <= e.index[-1].year)]
        m["Trades/Yr"] = yrs.trades.sum() / max(len(set(yrs.year)), 1)
        m["Avg gross exposure %"] = 100 * ww.abs().sum(axis=1).mean()
        out[name] = m
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepared_panel()
    res, curves, sels = {}, {}, {}
    for name, kw in VARIANTS.items():
        s, r, eq, w = run(panel, **kw)
        sel = pd.DataFrame(s.selection)
        res[name] = period_metrics(eq, w, sel)
        curves[name], sels[name] = eq, sel
        print(name, {p: round(float(res[name][p]["Sharpe"]), 2) for p in PERIODS})
    sels["v2 (all parts)"].to_csv(OUT / "selected_pairs.csv", index=False)

    # Split: stocks only vs ETFs only (diagnostic)
    for lab, groups in (("Stocks only", {f"stock:{k}": v for k, v in STOCK_GROUPS.items()}),
                        ("ETFs only", {f"etf:{k}": v for k, v in ETF_GROUPS.items()})):
        s, r, eq, w = run(panel, groups=groups)
        res[lab] = period_metrics(eq, w, pd.DataFrame(s.selection))
        curves[lab] = eq
        print(lab, {p: round(float(res[lab][p]["Sharpe"]), 2) for p in PERIODS})

    # Correlation to the BTC sleeves (daily returns on common dates, 2016-now)
    btc = PortfolioManager([Sleeve(OBVMomentum()), Sleeve(ACMomentum())], capital=200_000, start="2015-06-01").run()
    v2 = curves["v2 (all parts)"]
    common = v2.index.intersection(btc.equity.index)
    br = btc.equity.reindex(common).pct_change().loc["2016":]
    pr = v2.reindex(common).pct_change().loc["2016":]
    corr_btc = float(pr.corr(br))
    corr_spy = float(pr.corr(panel["Close"]["SPY"].reindex(common).pct_change().loc["2016":]))
    print("corr to BTC book", round(corr_btc, 3), "corr to SPY", round(corr_spy, 3))

    m = res["v2 (all parts)"]
    passes = (m["Full 2009–now"]["Sharpe"] >= 0.5 and all(m[p]["Sharpe"] > 0 for p in SUB) and abs(corr_btc) < 0.3)
    simpler = [n for n in VARIANTS if n != "v2 (all parts)" and
               all(res[n][p]["Sharpe"] >= m[p]["Sharpe"] for p in SUB)]
    print("PASSES:", passes, "| simpler versions at least as good in all periods:", simpler)

    book = None
    if passes:
        chosen = simpler[0] if simpler else "v2 (all parts)"
        kw = VARIANTS[chosen]
        rows = {}
        for lab, sl in (("BTC only", [Sleeve(OBVMomentum()), Sleeve(ACMomentum())]),
                        ("BTC + pairs", [Sleeve(OBVMomentum()), Sleeve(ACMomentum()), Sleeve(PairsPortfolio(**kw))])):
            for start in ("2016-01-01", "2022-01-01"):
                b = PortfolioManager(sl, capital=300_000, start=start, weighting="inverse_vol", max_weight=0.5).run()
                rows[f"{lab} from {start[:4]}"] = {k: b.metrics[k] for k in ("CAGR %", "Ann. Vol %", "Sharpe", "Max DD %")}
        book = rows
        print(pd.DataFrame(rows).T.round(2))

    yearly = {}
    for n in ("v2 (all parts)", "Stocks only", "ETFs only"):
        e = curves[n].loc["2009":]
        y = e.resample("YE").last()
        yearly[n] = (100 * (y / y.shift(1).fillna(e.iloc[0]) - 1)).round(2)
    yt = pd.DataFrame(yearly); yt.index = yt.index.year

    out = {"results": {k: {p: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else str(vv))
                                for kk, vv in d.items()} for p, d in v.items()} for k, v in res.items()},
           "corr_btc": corr_btc, "corr_spy": corr_spy, "passes": passes, "simpler": simpler, "book": book,
           "yearly_return_pct": {str(k): v for k, v in yt.to_dict(orient="index").items()},
           "pairs_per_year": sels["v2 (all parts)"].groupby("year").size().to_dict(),
           "group_share": sels["v2 (all parts)"].group.value_counts(normalize=True).round(3).to_dict()}
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=str))
    pd.DataFrame({k: v for k, v in curves.items()}).to_csv(OUT / "equity.csv")
    print(yt.to_string())


if __name__ == "__main__":
    main()
