"""Fase 4 (model): ridge, validação por verão, features e a execução completa, sem rede."""

import numpy as np
import pandas as pd
import pytest

from pechincha.eda import analysis as eda
from pechincha.model import core
from pechincha.model import fair_price as fp
from pechincha.pipeline import save
from test_eda import purchases as eda_purchases


def test_ridge_recovers_linear_relation_and_contributions_add_up():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = pd.Series(1.0 + 2.0 * X["a"] - 0.5 * X["b"])
    model = core.Ridge(alpha=1e-6).fit(X, y)
    assert model.predict(X) == pytest.approx(y.to_numpy(), abs=1e-6)
    # Coeficientes padronizados: efeito de +1 desvio-padrão.
    assert model.coefficients["a"] == pytest.approx(2.0 * X["a"].std(ddof=0), rel=1e-6)
    assert model.intercept + model.contributions(X).sum(axis=1).to_numpy() == pytest.approx(model.predict(X))


def test_ridge_shrinks_and_fills_missing_with_training_median():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    loose, tight = core.Ridge(0.01).fit(X, y), core.Ridge(100).fit(X, y)
    assert abs(tight.coefficients["a"]) < abs(loose.coefficients["a"])
    assert loose.predict(pd.DataFrame({"a": [np.nan]}))[0] == pytest.approx(loose.predict(pd.DataFrame({"a": [2.5]}))[0])


def test_summer_folds_never_train_on_the_predicted_summer():
    groups = pd.Series([2019, 2019, 2020, 2021])
    folds = list(core.summer_folds(groups))
    assert [g for g, _, _ in folds] == [2019, 2020, 2021]
    for g, train, test in folds:
        assert not (train & test).any() and (groups[test] == g).all() and (groups[train] != g).all()


def test_scores():
    s = core.scores(np.log([10, 20]), np.log([10, 40]))
    assert s["median_abs_pct_error"] == pytest.approx(50.0)
    assert s["within_25pct"] == 50.0 and s["n"] == 2


def model_purchases() -> pd.DataFrame:
    df = eda_purchases()
    df["minutes_before"] = df["minutes_before"].replace(300, 950)
    df["in_price_model"] = df["minutes_before"] >= 900
    df["in_ranking"] = df["in_price_model"] & (df["summer"] == 2026)
    df["model_exclusion"] = np.where(df["minutes_before"] <= 0, "no_league_minutes_before", None)
    df["market_value_listed"] = df["market_value_before"].fillna(5e6)
    df["origin_league_teams"] = 20.0
    df["bought_twice_in_summer"] = False
    return df


def test_features_use_listed_value_when_value_before_is_missing():
    df = model_purchases()
    pct = df[eda.KEY].assign(**{f"pct_{m}": 50.0 for m in fp.PERFORMANCE})
    f = fp.features(df, pct)
    assert set(fp.columns(list(fp.FEATURES))) <= set(f.columns)
    gap = df["market_value_before"].isna()
    assert f.loc[gap, "market_value_from_listing"].all() and not f.loc[~gap, "market_value_from_listing"].any()
    assert f.loc[gap, "log_mv_adjusted"].to_numpy() == pytest.approx(np.log(5e6))
    assert f["origin_rel_position"].between(0, 1).all()
    assert (f[["buyer_ENG", "buyer_ESP", "buyer_ITA", "buyer_GER"]].sum(axis=1) == (f["league"] != "FRA")).all()


def test_run_writes_ranking_predictions_and_charts(cfg):
    save(model_purchases(), cfg.root / "data" / "processed" / "purchases.parquet")
    eda.run(cfg)  # percentiles.parquet
    summary = fp.run(cfg)

    out = cfg.root / "outputs" / "model"
    charts = sorted(p.stem for p in out.glob("*.html"))
    assert charts == ["01_fee_vs_fair_2026", "02_ranking_2026", "03_coefficients", "04_model_comparison", "05_backtest",
                      "06_verdicts_by_league_2026", "07_verdicts_by_position_2026"]
    assert {p.stem for p in out.glob("*.json")} == set(charts) | {"ranking_2026"}
    verdicts = pd.read_csv(out / "verdicts_2026.csv")
    total = verdicts[verdicts["dimension"] == "all"].iloc[0]
    assert total["n"] == len(pd.read_csv(out / "ranking_2026.csv")) == total[["Bargain", "Fair", "Overpaid"]].sum()
    for dim in ["buying league", "position"]:  # cada dimensão reparte o total
        assert verdicts.loc[verdicts["dimension"] == dim, ["Bargain", "Fair", "Overpaid"]].sum().tolist() == total[["Bargain", "Fair", "Overpaid"]].tolist()
    for table in ["ranking_2026", "verdicts_2026", "model_comparison", "alpha_search", "cv_by_summer", "coefficients"]:
        assert (out / f"{table}.csv").exists(), table
    assert "2026" in (out / "summary.md").read_text(encoding="utf-8")

    ranking = pd.read_csv(out / "ranking_2026.csv")
    expected = model_purchases()
    assert len(ranking) == summary["n_rank"] == int(expected["in_ranking"].sum())
    assert ranking["fee_vs_fair"].is_monotonic_decreasing
    assert set(ranking["verdict"]) <= {"Overpaid", "Fair", "Bargain"}
    assert (ranking["fair_low"] <= ranking["fair_price"]).all() and (ranking["fair_price"] <= ranking["fair_high"]).all()

    pred = pd.read_parquet(cfg.root / "data" / "processed" / "fair_price.parquet")
    assert len(pred) == int(expected["in_price_model"].sum())
    assert pred.loc[pred["summer"] < 2026, "out_of_sample"].all()
    # A previsão em log é a base mais a soma das partes de cada grupo.
    cur = pred[pred["summer"] == 2026]
    parts = cur.filter(like="contrib_").sum(axis=1)
    assert np.log(cur["fair_price"] / cur["fee_index"]).to_numpy() == pytest.approx((np.log(cur["baseline_price"]) + parts).to_numpy())
    assert pred.loc[pred["out_of_sample"], "contrib_market_value"].isna().all()
