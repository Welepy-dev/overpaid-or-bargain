from datetime import date

from pechincha.ingest import sofascore as ss
from conftest import fixture_json


def test_search_has_no_birth_date_so_profile_is_needed():
    found = ss.parse_search(fixture_json("sofascore_search.json"))
    assert [c["sofascore_id"] for c in found] == [900, 901]
    assert all(c["date_of_birth"] is None for c in found)
    # Excerto real de /player/{id} (Geovany Quenda, nascido a 30/04/2007).
    assert ss.parse_player_birth(fixture_json("sofascore_player.json")) == date(2007, 4, 30)


def test_likely_candidates():
    found = [{"name": n} for n in ["Other Guy", "João Silva", "Joao Pedro", "Silva", "J. Silva"]]
    assert [c["name"] for c in ss.likely_candidates(found, "João Silva")] == ["João Silva", "Joao Pedro", "Silva"]


def test_pick_by_birth_date():
    found = ss.parse_search(fixture_json("sofascore_search.json"))
    found[0]["date_of_birth"], found[1]["date_of_birth"] = date(2003, 3, 4), date(2000, 1, 1)
    assert ss.pick_candidate(found, "Joao Silva", date(2003, 3, 4))["sofascore_id"] == 900


def test_pick_refuses_ambiguous_without_birth_date():
    found = ss.parse_search(fixture_json("sofascore_search.json"))
    assert ss.pick_candidate(found, "João Silva", None) is None


def test_relevant_seasons():
    assert ss.relevant_season_labels(2026) == {"25/26", "2025", "2026"}
    seasons = ss.parse_player_seasons(fixture_json("sofascore_seasons.json"))
    keep = [s for s in seasons if s["season_label"] in ss.relevant_season_labels(2026)]
    # Liga, Champions, Supertaça e as duas competições sub-21; não a 26/27 nem a 24/25.
    assert {(s["tournament_id"], s["season_id"]) for s in keep} == {(238, 77806), (7, 76953), (345, 76982), (26, 72382), (454, 69426)}
    # Só a liga conta: Champions, Supertaça e sub-21 ficam de fora.
    assert {s["tournament_id"] for s in keep if s["is_league"]} == {238}


def test_is_league():
    pt, br, de_am = {"name": "Portugal", "alpha2": "PT"}, {"name": "Brazil", "alpha2": "BR"}, {"name": "Germany Amateur", "alpha2": "DE"}
    for name, cat in [("Liga Portugal Betclic", pt), ("Trendyol Süper Lig", {"alpha2": "TR"}), ("Danish Superliga", {"alpha2": "DK"}),
                      ("Campeonato Brasileiro Série A", br), ("Liga MX, Apertura", {"alpha2": "MX"}), ("Eerste Divisie", {"alpha2": "NL"})]:
        assert ss.is_league(name, cat), name
    for name, cat in [("Taça da Liga", pt), ("Supertaça", pt), ("Copa do Brasil", br), ("Paulista Série A1", br),
                      ("Eurojackpot KNVB Beker", {"alpha2": "NL"}), ("Taça Revelação U23", pt), ("Regionalliga Bayern", de_am),
                      ("UEFA Champions League", {"name": "Europe"}), ("Johan Cruijff Schaal", {"alpha2": "NL"}),
                      ("Franz Beckenbauer Supercup", {"alpha2": "DE"}), ("Serie C, Playoffs", {"alpha2": "IT"})]:
        assert not ss.is_league(name, cat), name


def test_parse_statistics():
    stats = ss.parse_statistics(fixture_json("sofascore_stats.json"))
    assert stats["team_name"] == "Sporting CP" and stats["team_national"] is False
    assert stats["minutesPlayed"] == 873 and stats["expectedGoals"] == 1.6623
    assert "statisticsType" not in stats and "id" not in stats
