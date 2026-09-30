"""Pipeline de ponta a ponta com a rede substituída pelas fixtures."""

import pandas as pd
import pytest

from pechincha import pipeline
from pechincha.http import CachedFetcher, FetchError
from pechincha.ingest import tm_dump
from conftest import fixture_json, fixture_text
from test_new_sources import dump_tables

REQUESTS = []  # (url, max_age_hours) de cada pedido, para verificar a cache


def fake_text(self, url, max_age_hours=None, suffix=".html"):
    REQUESTS.append((url, max_age_hours))
    if "/wettbewerb/GB1/" in url:
        return fixture_text("tm_league_transfers.html")
    if "/wettbewerb/" in url:
        return "<html></html>"
    if url.endswith("/profil/spieler/1001"):
        return fixture_text("tm_profile_1001.html")
    raise FetchError(f"404 em {url}")


def fake_json(self, url, max_age_hours=None):
    REQUESTS.append((url, max_age_hours))
    births = {"/player/900": 1046736000, "/player/901": 946684800}  # 2003-03-04, 2000-01-01
    for key, ts in births.items():
        if url.endswith(key):
            return {"player": {"id": int(key.rsplit("/", 1)[1]), "dateOfBirthTimestamp": ts}}
    routes = {
        "unique-tournament/17/seasons": "sofascore_tournament_seasons.json",
        "unique-tournament/17/season/": "sofascore_league_stats.json",
        "transferHistory/list/1001": "tm_transfer_history_1001.json",
        "marketValueDevelopment/graph/1001": "tm_market_values_1001.json",
        "/search/all?": "sofascore_search.json",
        "/player/900/statistics/seasons": "sofascore_seasons.json",
        "/statistics/overall": "sofascore_stats.json",
    }
    for key, name in routes.items():
        if key in url:
            return fixture_json(name)
    raise FetchError(f"404 em {url}")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(CachedFetcher, "get_text", fake_text)
    monkeypatch.setattr(CachedFetcher, "get_json", fake_json)
    monkeypatch.setattr(tm_dump, "load", lambda cfg: dump_tables())
    REQUESTS.clear()


def test_transfers_details_and_sofascore(cfg):
    pipeline.run(cfg, steps=["transfers", "tm_details", "sofascore"], current_only=True)
    out = cfg.interim_dir

    eligible = pd.read_parquet(out / "transfers_eligible.parquet").set_index("player_id")
    # GR (1002), empréstimo (1003) e livre (1004) ficam de fora.
    assert set(eligible.index) == {1001, 1005}
    assert not eligible.loc[1001, "origin_top5"]           # vem do Benfica
    assert eligible.loc[1005, "origin_top5"]               # Chelsea estava nas 5 ligas em 2025/26
    assert eligible.loc[1005, "origin_league_top5"] == "ENG"

    enriched = pd.read_parquet(out / "transfers_enriched.parquet").set_index("player_id")
    assert enriched.loc[1001, "market_value_before"] == 55_000_000
    assert str(enriched.loc[1001, "transfer_date"]) == "2026-07-10"
    assert str(enriched.loc[1001, "date_of_birth"]) == "2003-03-04"
    assert pd.isna(enriched.loc[1005, "market_value_before"])  # sem detalhes: fica vazio, não falha

    matches = pd.read_parquet(out / "sofascore_matches.parquet")
    assert matches.set_index("player_id").loc[1001, "sofascore_id"] == 900
    stats = pd.read_parquet(out / "sofascore_player_seasons.parquet")
    assert set(stats["tournament_id"]) == {238, 7, 345, 26, 454}
    assert stats["expectedGoals"].iloc[0] == 1.6623 and stats["team_name"].iloc[0] == "Sporting CP"


def test_weekly_run_keeps_history(cfg):
    path = cfg.interim_dir / "transfers_eligible.parquet"
    path.parent.mkdir(parents=True)
    old = pd.DataFrame({"season": [2019, 2026], "player_id": [1, 2]})
    old.to_parquet(path)
    pipeline.run(cfg, steps=["transfers"], current_only=True)
    df = pd.read_parquet(path)
    assert 1 in set(df["player_id"])          # 2019 mantém-se
    assert 2 not in set(df["player_id"])      # 2026 foi substituída
    assert set(df.loc[df["season"] == 2026, "player_id"]) == {1001, 1005}


def test_history_uses_public_dataset_and_fetches_only_gaps(cfg):
    pipeline._step_transfers(cfg, [2025])
    pipeline._step_tm_details(cfg, [2025])
    enriched = pd.read_parquet(cfg.interim_dir / "transfers_enriched.parquet").set_index("player_id")
    assert enriched.loc[1001, "details_source"] == "tm_dump"
    assert enriched.loc[1001, "market_value_before"] == 55_000_000   # valor de 1/06/2025, antes da transferência
    assert str(enriched.loc[1001, "date_of_birth"]) == "2003-03-04"
    assert enriched.loc[1005, "details_source"] == "transfermarkt"
    tm = [(u, age) for u, age in REQUESTS if "transfermarkt" in u and "/wettbewerb/" not in u]
    assert tm and all("/1005" in u for u, _ in tm)                   # 1001 não vai ao Transfermarkt
    assert not any("marketValueDevelopment" in u for u, _ in tm)     # histórico já traz o valor
    assert all(age is None for _, age in tm)                         # verão fechado: cache não expira


def test_closed_summer_details_never_expire(cfg):
    assert cfg.summer_closed(2026, today=pd.Timestamp("2026-09-30").date())
    assert not cfg.summer_closed(2026, today=pd.Timestamp("2026-08-15").date())


def test_sofascore_leagues_replaces_fbref(cfg):
    failed = pipeline.run(cfg, steps=["transfers", "sofascore_leagues"], current_only=True)
    assert failed == []
    stats = pd.read_parquet(cfg.interim_dir / "sofascore_league_player_seasons.parquet")
    assert set(stats["league"]) == {"ENG"} and "tackles" in stats
    matches = pd.read_parquet(cfg.interim_dir / "sofascore_league_matches.parquet").set_index("player_id")
    assert matches.loc[1005, "sofascore_id"] == 501                  # Chelsea -> Arsenal
    assert "fbref" not in pipeline.STEPS and "clubelo" not in pipeline.STEPS
