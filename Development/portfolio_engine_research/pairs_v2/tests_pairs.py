"""Tests for pairs v1/v2, moved from portfolio-engine/tests/test_engine.py (2026-10-30).
To run: paste back into test_engine.py (they use its synthetic_panel helper and imports)."""

@pytest.mark.parametrize("version", ["A", "B", "C"])
def test_pairs_versions(version):
    panel = synthetic_panel(("AAA-USD", "BBB-USD"), n=900, seed=3)
    s = REGISTRY["PairsReversion"](symbols=["AAA-USD", "BBB-USD"], version=version, corr_hist=120)
    ind, sig = s.run(panel)
    w = sig.target
    if version == "C":
        assert (w >= 0).all().all() and (w.sum(axis=1) <= 1).all()        # long-only, one leg at a time
    else:
        assert np.allclose(w.sum(axis=1), 0)                                # dollar-neutral every bar
        assert set(np.unique(w.values)) <= {-0.5, 0.0, 0.5}
    # no look-ahead
    _, part = s.run({k: v.iloc[:600] for k, v in panel.items()})
    pd.testing.assert_frame_equal(part.target, w.iloc[:600], check_freq=False)
    r = backtest_strategy(s, 10_000, "2020-06-01", panel=panel)
    assert np.isfinite(r.equity).all()


def test_pairs_entry_logic():
    from strategies.pairs_reversion import pair_positions_ab
    gz = np.array([0, 2.1, 1.5, 0.5, -0.1, -2.2, -4.1, -3.0, -1.0, -2.5])
    cz = np.full(len(gz), -2.0)
    pos = pair_positions_ab(gz, cz, True, 2.0, 0.0, 4.0, 30, -1.0)
    # short spread at 2.1, exit when gap <= 0 (-0.1); long at -2.2; stopped at -4.1; blocked until |z|<2 (-1.0); re-enter -2.5
    assert list(pos) == [0, -1, -1, -1, 0, 1, 0, 0, 0, 1]
    pos_b = pair_positions_ab(gz, np.full(len(gz), 0.0), True, 2.0, 0.0, 4.0, 30, -1.0)
    assert (pos_b == 0).all()                                                # B needs the correlation trigger



def _coint_panel(n=900, seed=7):
    """Four symbols: two cointegrated pairs plus noise, ~3.5 years of daily bars."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 0.01, n))
    out = {}
    for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        ou = np.zeros(n)
        for t in range(1, n):
            ou[t] = 0.85 * ou[t - 1] + rng.normal(0, 0.006)
        out[s] = 50 * np.exp(common * (1 + 0.1 * i) + ou)
    c = pd.DataFrame(out, index=idx)
    return {"Open": c.shift(1).bfill(), "High": c * 1.005, "Low": c * 0.995, "Close": c,
            "Volume": pd.DataFrame(1e6, index=idx, columns=c.columns)}


def test_pairs_portfolio_no_lookahead_and_hedge():
    panel = _coint_panel()
    kw = dict(groups={"g": ["AAA", "BBB", "CCC", "DDD"]}, n_pairs=3, adf_p=1.0, max_half_life=None)
    s = REGISTRY["PairsPortfolio"](**kw)
    _, full = s.run(panel)
    sel = pd.DataFrame(s.selection)
    assert len(sel) and set(sel.year) <= {2020, 2021, 2022}
    for cut in (400, 650):
        s2 = REGISTRY["PairsPortfolio"](**kw)
        _, part = s2.run({k: v.iloc[:cut] for k, v in panel.items()})
        pd.testing.assert_frame_equal(part.target, full.target.iloc[:cut], check_freq=False)
    # each open pair is hedged 1 : beta, so with one pair the legs' ratio equals beta
    one = REGISTRY["PairsPortfolio"](**{**kw, "n_pairs": 1})
    _, sig = one.run(panel)
    d = one.selection[0]
    w = sig.target.loc[sig.target.index.year == d["year"], [d["a"], d["b"]]]
    w = w[(w != 0).any(axis=1)]
    if len(w):
        assert np.allclose(-w[d["b"]] / w[d["a"]], d["beta"])
        assert np.allclose(w.abs().sum(axis=1), 1.0)
    r = backtest_strategy(REGISTRY["PairsPortfolio"](**kw), 100_000, "2020-01-01", panel=panel)
    assert np.isfinite(r.equity).all()


def test_pair_state_corr_exit():
    from strategies.pairs_portfolio import run_pair_state
    z = np.array([0, 2.5, 2.2, 2.1, 1.0, 2.6, 0.0])
    corr = np.array([0.9, 0.9, 0.2, 0.3, 0.9, 0.9, 0.9])
    pos = run_pair_state(z, corr, 2.0, 0.0, 4.0, 30, True, 0.5, 0.7)
    # enter short at 2.5; corr collapses (0.2 < 0.5) -> exit; blocked; recovers + |z|<2 -> can re-enter at 2.6
    assert list(pos) == [0, -1, 0, 0, 0, -1, 0]
