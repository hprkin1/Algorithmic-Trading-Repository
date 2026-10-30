"""Oil spread research: in-sample grid -> pick robust params -> one out-of-sample run.

In-sample:      2012-01-01 .. 2021-12-31  (all parameter choices made here)
Out-of-sample:  2022-01-01 .. today       (looked at once, after selection)

Usage: python scripts/research_oil.py
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)  # noqa: E401,E702

from engine import ExecConfig, backtest_strategy  # noqa: E402
from engine.data import load_panel  # noqa: E402
from strategies import CorrReversion, OilSpreadReversion  # noqa: E402

IS_START, IS_END, OOS_START = "2012-01-01", "2021-12-31", "2022-01-01"
INSTRUMENTS = {"futures": ["CL=F", "BZ=F"], "etf": ["USO", "BNO"]}
GRID = dict(lookback=[20, 30, 40, 60, 90], z_in=[1.5, 2.0, 2.5], z_out=[0.0, 0.5],
            z_stop=[3.5, 99.0], max_hold=[15, 30, 60], corr_min=[-1.0, 0.7])
OUT = ROOT / "reports" / "oil"


def slice_panel(panel, end):
    return {k: v.loc[:end] for k, v in panel.items()}


def run(params, syms, panel, start, end=None, cfg=ExecConfig()):
    p = slice_panel(panel, end) if end else panel
    return backtest_strategy(OilSpreadReversion(symbols=syms, **params), 100_000, start, config=cfg, panel=p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Oil spread research", "",
             f"In-sample {IS_START} → {IS_END} (grid of {np.prod([len(v) for v in GRID.values()])} combos per "
             f"instrument); out-of-sample {OOS_START} → today, run once on the selected params.", ""]
    keys = list(GRID)
    best = {}
    for inst, syms in INSTRUMENTS.items():
        panel = load_panel(syms, "2011-06-01")
        rows = []
        for combo in itertools.product(*GRID.values()):
            params = dict(zip(keys, combo))
            m = run(params, syms, panel, IS_START, IS_END).metrics
            rows.append({**params, **{k: m.get(k) for k in ("Sharpe", "CAGR %", "Max DD %", "Trades", "Win Rate %")}})
        g = pd.DataFrame(rows)
        g.to_csv(OUT / f"grid_is_{inst}.csv", index=False)
        g = g[g["Trades"] >= 30]  # need enough trades to mean anything (~3/yr)

        # Robust selection: score each combo by the MEDIAN Sharpe of itself and its grid neighbours
        # (one step in any single parameter). Prefers plateaus over isolated peaks.
        def neighbours(r):
            out = []
            for k in keys:
                vals = GRID[k]; i = vals.index(r[k])
                for j in (i - 1, i + 1):
                    if 0 <= j < len(vals):
                        q = g
                        for kk in keys:
                            q = q[q[kk] == (vals[j] if kk == k else r[kk])]
                        out += list(q["Sharpe"])
            return out
        g["Robust Sharpe"] = [np.nanmedian([r["Sharpe"]] + neighbours(r)) for _, r in g.iterrows()]
        top = g.sort_values("Robust Sharpe", ascending=False)
        pick = {k: top.iloc[0][k] for k in keys}
        pick["lookback"], pick["max_hold"] = int(pick["lookback"]), int(pick["max_hold"])
        best[inst] = (pick, syms, panel)

        lines += [f"## {inst} {syms}: in-sample", "",
                  f"Combos with ≥30 trades: {len(g)} · share with Sharpe > 0: {100*(g['Sharpe']>0).mean():.0f}% · "
                  f"median Sharpe {g['Sharpe'].median():.2f}", "",
                  "Median in-sample Sharpe by each parameter value (all other params pooled):", ""]
        for k in keys:
            lines.append(f"- **{k}**: " + ", ".join(f"{v}: {g[g[k]==v]['Sharpe'].median():.2f}" for v in GRID[k]))
        lines += ["", "Top 8 by robust (neighbourhood-median) Sharpe:", "",
                  top.head(8).round(2).to_markdown(index=False), "", f"**Selected:** `{pick}`", ""]
        print(inst, "selected", pick, "IS robust sharpe", round(top.iloc[0]["Robust Sharpe"], 2))

    # ---- Out-of-sample: run once ----
    lines += ["## Out-of-sample (selected params, run once)", ""]
    oos_tbl, curves = {}, {}
    cols = ["Total Return %", "CAGR %", "Ann. Vol %", "Sharpe", "Max DD %", "Trades", "Win Rate %",
            "Avg Win %", "Avg Loss %", "Exposure %", "Costs Paid"]
    for inst, (pick, syms, panel) in best.items():
        for label, start, end in [("IS", IS_START, IS_END), ("OOS", OOS_START, None)]:
            r = run(pick, syms, panel, start, end)
            oos_tbl[f"{inst} {label}"] = r.metrics
            curves[f"{inst} {label}"] = r
        # cost sensitivity OOS
        for mult in (2, 3):
            base = ExecConfig().cost_model(syms)
            from engine import CostModel
            cfg = ExecConfig(costs=CostModel(base.fee * mult, base.slippage * mult))
            oos_tbl[f"{inst} OOS costs×{mult}"] = run(pick, syms, panel, OOS_START, cfg=cfg).metrics
        # original strategy for reference
    orig = backtest_strategy(CorrReversion(), 100_000, OOS_START)
    oos_tbl["original CorrReversion OOS"] = orig.metrics
    t = pd.DataFrame(oos_tbl).T.reindex(columns=cols)
    lines += [t.round(2).to_markdown(), ""]
    print(t[["Total Return %", "Sharpe", "Max DD %", "Trades", "Win Rate %"]].round(2))

    # OOS by calendar year + current position
    lines += ["## OOS by year (return %)", ""]
    yr = {}
    for inst in best:
        e = curves[f"{inst} OOS"].equity
        yr[inst] = (e.resample("YE").last() / e.resample("YE").last().shift(1).fillna(e.iloc[0]) - 1) * 100
    yt = pd.DataFrame(yr); yt.index = yt.index.year
    lines += [yt.round(2).to_markdown(), ""]

    lines += ["## Signal state today", ""]
    for inst, (pick, syms, panel) in best.items():
        ind = curves[f"{inst} OOS"].indicators.iloc[-1]
        side = {1: "LONG spread (long Brent / short WTI)", -1: "SHORT spread (short Brent / long WTI)", 0: "flat"}
        lines.append(f"- **{inst}**: z = {ind['z']:.2f}, 30d corr = {ind['corr']:.2f}, position: {side[int(ind['pos'])]}")
        print(inst, "today z", round(ind["z"], 2), "pos", ind["pos"])
    (OUT / "README.md").write_text("\n".join(lines))
    _plot(curves, best)


def _plot(curves, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1.3, 1]})
    for inst in best:
        for label, ls in (("IS", ":"), ("OOS", "-")):
            e = curves[f"{inst} {label}"].equity
            axes[0].plot(e.index, e / 100_000, ls, label=f"{inst} {label}")
    axes[0].axvline(pd.Timestamp(OOS_START), color="grey", lw=0.8)
    axes[0].set_title("Oil spread reversion: growth of $1 (dotted = in-sample, solid = out-of-sample)")
    axes[0].legend(fontsize=8)
    ind = curves["futures OOS"].indicators
    panel = best["futures"][2]
    c = panel["Close"].loc["2015":]
    axes[1].plot(c.index, c["BZ=F"] - c["CL=F"], color="black", lw=0.8)
    axes[1].set_ylabel("Brent − WTI $")
    axes[2].plot(ind.index, ind["z"], lw=0.7)
    for y in (2, -2):
        axes[2].axhline(y, color="red", lw=0.5)
    axes[2].set_ylabel("z (futures, OOS)")
    fig.tight_layout(); fig.savefig(OUT / "oil_spread.png", dpi=110)


if __name__ == "__main__":
    main()
