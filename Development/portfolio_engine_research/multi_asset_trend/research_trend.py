"""Multi-asset trend: pre-declared test. Run from the portfolio-engine root."""
import logging, sys, warnings, json
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from engine import ExecConfig, PortfolioManager, Sleeve, backtest_strategy
from engine.data import load_panel
from engine.metrics import compute_metrics
from strategies import ACMomentum, OBVMomentum
from strategies.multi_asset_trend import MultiAssetTrend

PERIODS = {"2008–15": ("2008-04-01", "2015-12-31"), "2016–21": ("2016-01-01", "2021-12-31"),
           "2022–now": ("2022-01-01", None), "Full 2008–now": ("2008-04-01", None)}
BORROW = 0.005
s0 = MultiAssetTrend()
panel = load_panel(s0.symbols, "2006-01-01")

def run(cost_mult=1, **kw):
    s = MultiAssetTrend(**kw)
    from engine import CostModel
    cfg = ExecConfig(costs=CostModel(0.0005 * cost_mult, 0.0005 * cost_mult))
    r = backtest_strategy(s, 1_000_000, "2008-04-01", config=cfg, panel=panel)
    w = r.signals.target.ffill().shift(1).reindex(r.equity.index).fillna(0)
    ret = r.equity.pct_change().fillna(0) - w.clip(upper=0).abs().sum(axis=1) * BORROW / 252
    return r, (1 + ret).cumprod() * 1_000_000, w

def per(eq):
    return {k: compute_metrics(eq.loc[a:b] if b else eq.loc[a:]) for k, (a, b) in PERIODS.items()}

r, eq, w = run()
m = per(eq)
cols = ["CAGR %", "Ann. Vol %", "Sharpe", "Max DD %", "Calmar"]
print(pd.DataFrame(m).T[cols].astype(float).round(2))
print("avg gross %.0f%%, avg net %.0f%%, trades %d" % (100 * w.abs().sum(axis=1).mean(), 100 * w.sum(axis=1).mean(), len(r.portfolio.orders.records_readable)))

# 60/40 benchmark (SPY/IEF, monthly rebalanced, same costs ignored)
c = panel["Close"]; br = (0.6 * c["SPY"].pct_change() + 0.4 * c["IEF"].pct_change()).loc["2008-04-01":].fillna(0)
bench = (1 + br).cumprod()
print("60/40:", {k: round(float(v["Sharpe"]), 2) for k, v in per(bench).items()})

# correlations
btc = PortfolioManager([Sleeve(OBVMomentum()), Sleeve(ACMomentum())], capital=200_000, start="2015-06-01").run()
common = eq.index.intersection(btc.equity.index)
cb = eq.reindex(common).pct_change().loc["2016":].corr(btc.equity.reindex(common).pct_change().loc["2016":])
cs = eq.pct_change().loc["2008-05":].corr(c["SPY"].pct_change().loc["2008-05":])
print("corr to BTC book %.3f, corr to SPY %.3f" % (cb, cs))

passes = m["Full 2008–now"]["Sharpe"] >= 0.5 and all(m[k]["Sharpe"] > 0 for k in ["2008–15", "2016–21", "2022–now"]) and abs(cb) < 0.3
print("PASSES bar (excluding book check):", passes)

# robustness (no selection): costs x2/x3, single lookbacks, long-only
rob = {}
for lab, kw in [("costs x2", dict(cost_mult=2)), ("costs x3", dict(cost_mult=3)), ("12m only", dict(lookbacks=(252,))),
                ("3m only", dict(lookbacks=(63,))), ("1m only", dict(lookbacks=(21,))), ("long-only", dict(long_only=True))]:
    _, e2, _ = run(**kw)
    rob[lab] = {k: v["Sharpe"] for k, v in per(e2).items()}
print(pd.DataFrame(rob).T.astype(float).round(2))

# yearly
y = eq.resample("YE").last(); yr = (100 * (y / y.shift(1).fillna(1_000_000) - 1)).round(1)
yb = bench.resample("YE").last(); ybr = (100 * (yb / yb.shift(1).fillna(1.0) - 1)).round(1)
print(pd.DataFrame({"trend": yr.values, "60/40": ybr.values, "SPY": (100 * (c['SPY'].resample('YE').last().pct_change())).loc['2008':].round(1).values[:len(yr)]}, index=yr.index.year).T.to_string())

# book impact
book = {}
for lab, sl in [("BTC only", [Sleeve(OBVMomentum()), Sleeve(ACMomentum())]),
                ("BTC + trend", [Sleeve(OBVMomentum()), Sleeve(ACMomentum()), Sleeve(MultiAssetTrend())])]:
    for wt in ("capital", "inverse_vol"):
        for start in ("2016-01-01", "2022-01-01"):
            kw = dict(weighting=wt, max_weight=0.5) if wt == "inverse_vol" else {}
            b = PortfolioManager(sl, capital=300_000, start=start, **kw).run()
            book[f"{lab} | {wt} | from {start[:4]}"] = {k: b.metrics[k] for k in ("CAGR %", "Ann. Vol %", "Sharpe", "Max DD %")}
print(pd.DataFrame(book).T.astype(float).round(2))
json.dump({"m": {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float, np.floating))} for k, v in m.items()},
           "corr_btc": float(cb), "corr_spy": float(cs), "passes": bool(passes), "robust": rob, "book": book},
          open("/tmp/claude-0/-home-claude/01593eb7-a125-592e-8b77-a453ef11144f/scratchpad/trend_results.json", "w"), default=float, indent=1)
