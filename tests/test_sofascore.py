from datetime import date

from pechincha.ingest import sofascore as ss
from conftest import fixture_json


def test_search_and_pick_by_birth_date():
    found = ss.parse_search(fixture_json("sofascore_search.json"))
    assert [c["sofascore_id"] for c in found] == [900, 901]
    pick = ss.pick_candidate(found, "Joao Silva", date(2003, 3, 4))
    assert pick["sofascore_id"] == 900


def test_pick_refuses_ambiguous_without_birth_date():
    found = ss.parse_search(fixture_json("sofascore_search.json"))
    assert ss.pick_candidate(found, "João Silva", None) is None


def test_relevant_seasons():
    assert ss.relevant_season_labels(2026) == {"25/26", "2025", "2026"}
    seasons = ss.parse_player_seasons(fixture_json("sofascore_seasons.json"))
    keep = [s for s in seasons if s["season_label"] in ss.relevant_season_labels(2026)]
    assert {(s["tournament_id"], s["season_id"]) for s in keep} == {(238, 77), (7, 88)}
