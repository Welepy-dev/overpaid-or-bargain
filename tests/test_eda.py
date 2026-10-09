"""Fase 3 (eda): percentis por posição, tabelas auxiliares e a execução completa, sem rede."""

import numpy as np
import pandas as pd
import pytest

from pechincha.eda import analysis as eda
from pechincha.pipeline import save


def purchases(n_per_summer: int = 24, seed: int = 0) -> pd.DataFrame:
    """Tabela por compra sintética com as colunas que a EDA lê (como a do ``build``)."""
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for summer in range(2019, 2027):
        for i in range(n_per_summer):
            pid += 1
            mv = float(rng.lognormal(16, 1))
            fee = float(mv * rng.lognormal(0, 0.4))
            minutes = float(rng.choice([0, 300, 1200, 2500, 3200]))
            rows.append({
                "summer": summer, "player_id": pid, "club_id": 100 + i % 8, "club_name": f"Club {i % 8}",
                "league": list(eda.LEAGUE_NAMES)[i % 5], "player_name": f"Player {pid}",
                "position_group": eda.POSITIONS[i % 3], "position": "Centre-Forward", "other_club_name": f"Seller {i % 6}",
                "origin_league_top5": list(eda.LEAGUE_NAMES)[(i + 1) % 5], "origin_league_position": float(i % 20 + 1),
                "origin_ppg": 1 + rng.random(), "origin_uefa_5y": 50 + 50 * rng.random(), "height_cm": 170 + 20 * rng.random(),
                "age_at_transfer": 18 + 16 * rng.random(), "fee_known": True, "fee": fee, "fee_index": 1.0, "fee_adjusted": fee,
                "market_value_before": mv if i % 10 else np.nan, "minutes_before": minutes,
                "transfer_date": f"{summer}-07-15", "contract_expires": f"{summer + 1 + i % 5}-06-30",
                "in_price_model": minutes > 0, "in_ranking": summer == 2026 and minutes > 0,
            })
    df = pd.DataFrame(rows)
    metrics = sorted({m for ms in eda.PROFILE_METRICS.values() for m in ms} | set(eda.HEATMAP_METRICS) | set(eda.METRIC_LABELS))
    for m in metrics:
        df[m] = rng.random(len(df)) * 5
    df["ss_ballRecovery_p90"] = np.nan  # fica fora dos percentis (SKIP)
    return df


def test_prepare_derived_columns():
    df = eda.prepare(pd.DataFrame({
        "fee": [2e7, 0.0], "fee_adjusted": [2.2e7, 0.0], "fee_index": [1.1, 1.0], "market_value_before": [1.1e7, np.nan],
        "transfer_date": ["2024-07-01", "2024-07-01"], "contract_expires": ["2029-06-30", None],
    }))
    assert df["fee_m"].tolist() == [20.0, 0.0]
    assert df["mv_adjusted"].iloc[0] == pytest.approx(1e7)
    assert df["log_fee_adjusted"].iloc[0] == pytest.approx(np.log(2.2e7))
    assert np.isnan(df["log_fee_adjusted"].iloc[1])  # fee 0 não tem log
    assert df["contract_years"].iloc[0] == pytest.approx(5, abs=0.01)
    assert np.isnan(df["contract_years"].iloc[1])


def test_percentiles_reference_ties_and_inverted_metrics():
    df = pd.DataFrame({
        "summer": 2025, "player_id": range(6), "club_id": 1, "player_name": list("abcdef"),
        "position_group": ["ATT"] * 5 + ["DEF"],
        "minutes_before": [1000, 1000, 1000, 1000, 100, 1000],
        "in_price_model": [True, True, True, False, True, True],
        "x_p90": [1.0, 2.0, 2.0, 9.0, 3.0, 5.0],
        "ss_fouls_p90": [1.0, 2.0, 3.0, 0.0, 0.0, 1.0],
    })
    out = eda.percentiles(df, ["x_p90", "ss_fouls_p90"], min_minutes=900)
    # Referência ATT: só as três primeiras (a 4.ª não está no modelo, a 5.ª tem poucos minutos).
    assert out["reference_n"].tolist() == [3, 3, 3, 3, 3, 1]
    assert out["low_minutes"].tolist() == [False, False, False, False, True, False]
    # Empates contam metade: 2.0 fica entre 1 e 2 compras abaixo.
    assert out["pct_x_p90"].tolist()[:5] == pytest.approx([16.7, 66.7, 66.7, 100.0, 100.0])
    assert out["pct_x_p90"].iloc[5] == 50.0
    # Faltas: menos é melhor, por isso o percentil é invertido.
    assert out["pct_ss_fouls_p90"].iloc[0] == pytest.approx(83.3)
    assert out["pct_ss_fouls_p90"].iloc[2] == pytest.approx(16.7)


def test_percentiles_empty_without_minutes():
    df = pd.DataFrame({"summer": 2025, "player_id": [1, 2], "club_id": 1, "player_name": ["a", "b"], "position_group": "MID",
                       "minutes_before": [1500, np.nan], "in_price_model": True, "x_p90": [1.0, 2.0]})
    out = eda.percentiles(df, ["x_p90"])
    assert out["pct_x_p90"].iloc[0] == 50.0
    assert np.isnan(out["pct_x_p90"].iloc[1])
    assert out["low_minutes"].tolist() == [False, True]


def test_percentile_metrics_skips_ball_recovery():
    df = pd.DataFrame(columns=["us_xg_p90", "ss_ballRecovery_p90", "fee"])
    assert eda.percentile_metrics(df) == ["us_xg_p90"] + eda.RATIOS


def test_minutes_thresholds_counts():
    df = pd.DataFrame({"fee_known": [True, True, True, False], "summer": [2024, 2026, 2026, 2026],
                       "minutes_before": [1000, 500, np.nan, 3000], "position_group": ["ATT", "MID", "DEF", "ATT"]})
    t = eda.minutes_thresholds(df, [1, 900]).set_index("min_minutes")
    assert t.loc[1, "price_model"] == 2 and t.loc[1, "ranking_2026"] == 1 and t.loc[1, "ranking_2026_MID"] == 1
    assert t.loc[900, "price_model"] == 1 and t.loc[900, "train_2019_2025"] == 1 and t.loc[900, "ranking_2026"] == 0


def test_redundant_pairs_finds_duplicates():
    rng = np.random.default_rng(1)
    x = rng.random(60)
    df = pd.DataFrame({"position_group": eda.POSITIONS * 20, "a": x, "b": 2 * x + 0.01, "c": rng.random(60)})
    pairs = eda.redundant_pairs(df, ["a", "b", "c"])
    assert list(zip(pairs["metric_a"], pairs["metric_b"])) == [("a", "b")]
    assert pairs["max_abs_rho"].iloc[0] == 1.0


def test_spearman_ignores_missing():
    a = pd.Series([1, 2, 3, np.nan, 5])
    b = pd.Series([10, 20, 30, 40, np.nan])
    assert eda._spearman(a, b) == pytest.approx(1.0)


def test_run_writes_charts_tables_and_percentiles(cfg):
    save(purchases(), cfg.root / "data" / "processed" / "purchases.parquet")
    months = pd.period_range("2019-01", "2026-09", freq="M")
    save(pd.DataFrame({"period": months.astype(str), "year": months.year, "month": months.month,
                       "hicp": 100 + np.arange(len(months)) * 0.25}), cfg.interim_dir / "hicp_monthly.parquet")
    summary = eda.run(cfg)

    out = cfg.root / "outputs" / "eda"
    charts = sorted(p.stem for p in out.glob("*.html"))
    assert len(charts) == 20 and {"03b_spend_by_summer_real", "06b_age_by_summer", "12b_top_bargains_2026"} <= set(charts)
    assert {p.stem for p in out.glob("*.json")} == set(charts)
    assert "cdn.plot.ly" in (out / "01_fee_distribution.html").read_text(encoding="utf-8")
    for table in ["fees_by_summer", "ages_by_summer", "minutes_thresholds", "purchases_2026_by_fee", "bargains_2026", "spend_2026_by_league", "spend_2026_by_club",
                  "fee_by_age", "feature_correlations", "redundant_metric_pairs"]:
        assert (out / f"{table}.csv").exists(), table
    assert "2026" in (out / "summary.md").read_text(encoding="utf-8")
    fees = pd.read_csv(out / "fees_by_summer.csv")
    assert (fees["total_fee_real_m"] > fees["total_fee_m"])[fees["summer"] < 2026].all()
    assert (fees.loc[fees["summer"] == 2026, "hicp_deflator"] == 1).all()
    bargains = pd.read_csv(out / "bargains_2026.csv")
    assert len(bargains) and (bargains["discount_m"] > 0).all() and bargains["discount_m"].is_monotonic_decreasing

    pct = pd.read_parquet(cfg.root / "data" / "processed" / "percentiles.parquet")
    assert len(pct) == summary["n"] == 8 * 24
    assert not pct.duplicated(eda.KEY).any()
    assert "pct_ss_ballRecovery_p90" not in pct
    vals = pct.filter(like="pct_").stack().dropna()
    assert len(vals) and vals.between(0, 100).all()


def test_run_without_hicp_skips_real_chart(cfg):
    save(purchases(), cfg.root / "data" / "processed" / "purchases.parquet")
    eda.run(cfg)
    out = cfg.root / "outputs" / "eda"
    assert not (out / "03b_spend_by_summer_real.html").exists()
    assert "total_fee_real_m" not in pd.read_csv(out / "fees_by_summer.csv")
    assert "No HICP file yet" in (out / "summary.md").read_text(encoding="utf-8")
