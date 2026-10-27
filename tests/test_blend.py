import numpy as np
import pandas as pd
import pytest

from nbamodel import blend


def _preds(n_seasons=4, per=400, elo_noise=0.02, ml_noise=0.12, early_only_noise=False, seed=0):
    """Synthetic walk-forward predictions with a known truth: whichever input is less noisy should win."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2016, 2016 + n_seasons):
        gp = np.tile(np.arange(0, 80, 80 / per * 2), 2)[:per]
        truth = rng.uniform(0.25, 0.75, per)
        ml_sd = np.where(gp < 15, ml_noise, 0.01) if early_only_noise else np.full(per, ml_noise)
        rows.append(pd.DataFrame({
            "season": s, "date": pd.Timestamp(f"{s}-10-20") + pd.to_timedelta(gp * 2, unit="D"),
            "home_season_games": gp, "away_season_games": gp,
            "p_elo": np.clip(truth + rng.normal(0, elo_noise, per), 0.02, 0.98),
            "p_ml": np.clip(truth + rng.normal(0, ml_sd), 0.02, 0.98),
            "home_win": (rng.uniform(size=per) < truth).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def test_blend_formula():
    assert blend.blend(0.7, 0.5, 0.25) == pytest.approx(0.25 * 0.7 + 0.75 * 0.5)
    assert blend.blend(0.7, 0.5, 2.0) == pytest.approx(0.7)              # weight clipped to [0, 1]


def test_fixed_weight_prefers_less_noisy_input():
    p = _preds(elo_noise=0.01, ml_noise=0.15)
    assert blend.fit(p, "fixed", "p_ml")["w"] >= 0.7
    p = _preds(elo_noise=0.15, ml_noise=0.01)
    assert blend.fit(p, "fixed", "p_ml")["w"] <= 0.3


def test_early_only_weights_switch_off_after_threshold():
    params = {"method": "early_only", "games": 15, "w": 0.6}
    df = pd.DataFrame({"home_season_games": [0, 14, 15, 60], "away_season_games": [3, 20, 15, 70]})
    assert list(blend.weights(df, params)) == [0.6, 0.6, 0.0, 0.0]


def test_forward_chain_fits_only_on_earlier_seasons():
    p = _preds()
    oos, chosen = blend.forward_chain(p, "p_ml", methods=("fixed",))
    assert sorted(oos["season"].unique()) == [2017, 2018, 2019]          # first season only trains
    for c in chosen:
        assert c["w"] == blend.fit(p[p["season"] < c["season"]], "fixed", "p_ml")["w"]
    # changing a test season's outcomes cannot change the weight used for that season
    flipped = p.copy()
    flipped.loc[flipped["season"] == 2019, "home_win"] ^= 1
    _, chosen2 = blend.forward_chain(flipped, "p_ml", methods=("fixed",))
    assert [c["w"] for c in chosen if c["season"] == 2019] == [c["w"] for c in chosen2 if c["season"] == 2019]


def test_study_enables_only_on_clear_out_of_sample_gain():
    no_gain = _preds(elo_noise=0.12, ml_noise=0.02)                 # Elo is just noise: never enable
    assert blend.study(no_gain.rename(columns={"p_ml": "p_base"}), "p_base", n_boot=50).report["enabled"] is False
    gain = _preds(elo_noise=0.01, ml_noise=0.15, per=800)
    st = blend.study(gain.rename(columns={"p_ml": "p_base"}), "p_base", n_boot=50)
    assert st.report["enabled"] is True and st.report["best_method"] in blend.METHODS
    seg = st.report["by_segment"]
    assert set(seg) == set(blend.SEGMENTS)


def test_holdout_confirmation_does_not_refit():
    p = _preds().rename(columns={"p_ml": "p_base"})
    params = {"method": "fixed", "w": 0.3}
    r = blend.confirm_on_holdout(p, params, "p_base", n_boot=20)
    expected = 0.3 * p["p_elo"] + 0.7 * p["p_base"]
    y = p["home_win"]
    ll = -np.mean(y * np.log(expected) + (1 - y) * np.log(1 - expected))
    assert r["overall"]["p_blend"]["log_loss"] == pytest.approx(ll)
