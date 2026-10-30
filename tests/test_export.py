"""Exportação para o portfólio: os três JSON a partir das saídas da Fase 5, sem rede."""

import json

import pandas as pd
import pytest

from pechincha import export
from pechincha.performance import analysis as perf
from pechincha.pipeline import save
from test_performance import scenario


@pytest.fixture
def performance_outputs(cfg):
    """Corre a Fase 5 no cenário de ``test_performance`` (com o índice a entrar com 270 minutos)."""
    df, cur, matches, schedule, sofa, understat, tm_clubs = scenario()
    save(df, cfg.root / "data" / "processed" / "purchases.parquet")
    inter = cfg.interim_dir
    save(matches, inter / "understat_2026_player_matches.parquet")
    save(schedule, inter / "understat_2026_schedule.parquet")
    save(sofa, inter / "sofascore_league_player_seasons.parquet")
    save(understat, inter / "understat_player_seasons.parquet")
    save(tm_clubs, inter / "tm_top5_clubs.parquet")
    cfg.phase5["min_minutes_ranking"] = 270
    perf.run(cfg)
    return cfg


def test_export_writes_meta_ranking_and_evolution(performance_outputs, tmp_path):
    cfg = performance_outputs
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()
    out = export.run(cfg, portfolio)
    assert out == portfolio / "public" / "data" / "overpaid-or-bargain"
    assert sorted(p.name for p in out.iterdir()) == ["evolution.json", "meta.json", "ranking.json"]

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["season"] == "2026/27" and meta["snapshot_date"] == "2026-09-12" and meta["snapshots"] == 1
    assert meta["purchases"] == 24 and meta["min_minutes_ranking"] == 270 and sum(meta["status"].values()) == 24
    # Arsenal jogou 4, o Fulham só 1.
    eng = next(lg for lg in meta["leagues"] if lg["league"] == "Premier League")
    assert eng == {"league": "Premier League", "min_matches": 1, "max_matches": 4, "last_result": "2026-09-12"}

    ranking = json.loads((out / "ranking.json").read_text(encoding="utf-8"))
    table = pd.read_csv(cfg.root / "outputs" / "performance" / "performance_2026.csv")
    assert len(ranking) == meta["ranked"] == table["value_index"].notna().sum() > 0
    assert [r["index"] for r in ranking] == sorted((r["index"] for r in ranking), reverse=True)
    first = ranking[0]
    assert set(first) == {"player", "club", "league", "position", "fee", "fee_pct", "score", "index", "low", "high", "minutes", "confidence", "verdict"}
    assert first["minutes"] == 270 and first["confidence"] == "low" and first["fee"] < 1000  # milhões de euros

    evolution = json.loads((out / "evolution.json").read_text(encoding="utf-8"))
    assert {p["player"] for p in evolution} == {r["player"] for r in ranking}
    assert all(len(p["points"]) == 1 and p["points"][0]["date"] == "2026-09-12" for p in evolution)

    # Os JSON são JSON estrito (sem NaN), para o browser os ler.
    for name in ["meta.json", "ranking.json", "evolution.json"]:
        assert "NaN" not in (out / name).read_text(encoding="utf-8")


def test_evolution_keeps_the_most_expensive_and_every_snapshot():
    table = pd.DataFrame({"summer": 2026, "player_id": [1, 2, 3], "club_id": [10, 20, 30], "fee": [50e6, 20e6, 5e6]})
    rows = []
    for date, indexes in [("2026-09-12", [10.0, None, -5.0]), ("2026-09-19", [12.0, 3.0, -8.0])]:
        for pid, index in zip([1, 2, 3], indexes):
            rows.append({"snapshot_date": date, "summer": 2026, "player_id": pid, "club_id": pid * 10, "player_name": f"P{pid}", "buying_club": "C",
                         "position_group": "ATT", "minutes_after": 900, "confidence": "medium", "value_index": index,
                         "value_index_low": None if index is None else index - 5, "value_index_high": None if index is None else index + 5})
    out = export.evolution(pd.DataFrame(rows), table, n=2)
    assert [p["player"] for p in out] == ["P1", "P2"] and out[0]["fee"] == 50
    assert [pt["date"] for pt in out[0]["points"]] == ["2026-09-12", "2026-09-19"]
    # Sem índice numa fotografia (ainda fora do ranking), esse ponto fica de fora.
    assert [pt["date"] for pt in out[1]["points"]] == ["2026-09-19"]


def test_export_needs_the_portfolio_and_phase5_outputs(cfg, tmp_path):
    with pytest.raises(SystemExit, match="Portfólio não encontrado"):
        export.run(cfg, tmp_path / "nowhere")
    (tmp_path / "portfolio").mkdir()
    with pytest.raises(SystemExit, match="performance"):
        export.run(cfg, tmp_path / "portfolio")
