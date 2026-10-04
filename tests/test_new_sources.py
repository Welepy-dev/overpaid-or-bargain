"""Dataset público do Transfermarkt, Sofascore por liga e ranking UEFA."""

from datetime import date

import pandas as pd

from pechincha.ingest import sofascore as ss
from pechincha.ingest import tm_dump, uefa
from conftest import fixture_json, fixture_text


def dump_tables():
    transfers = pd.DataFrame({
        "player_id": [1001, 1001, 9999],
        "transfer_date": ["2025-07-10", "2023-07-01", "2025-07-02"],
        "transfer_season": ["25/26", "23/24", "25/26"],
        "from_club_id": [294, 2999, 1], "to_club_id": [11, 294, 2],
        "from_club_name": ["Benfica", "Benfica B", "A"], "to_club_name": ["Arsenal", "Benfica", "B"],
        "transfer_fee": [70_000_000.0, None, 1.0], "market_value_in_eur": [60_000_000.0, 1_000_000.0, 1.0],
        "player_name": ["João Silva", "João Silva", "X"],
    })
    valuations = pd.DataFrame({
        "player_id": [1001, 1001, 1001], "date": ["2024-12-15", "2025-06-01", "2025-12-20"],
        "market_value_in_eur": [40_000_000, 55_000_000, 75_000_000], "current_club_name": ["Benfica", "Benfica", "Arsenal"],
        "current_club_id": [294, 294, 11], "player_club_domestic_competition_id": ["PO1", "PO1", "GB1"],
    })
    players = pd.DataFrame({
        "player_id": [1001], "name": ["João Silva"], "date_of_birth": ["2003-03-04 00:00:00"], "height_in_cm": [181.0],
        "foot": ["right"], "position": ["Midfield"], "sub_position": ["Central Midfield"],
        "country_of_citizenship": ["Portugal"], "contract_expiration_date": ["2031-06-30 00:00:00"],
    })
    return {"transfers": transfers, "player_valuations": valuations, "players": players}


def test_dump_same_columns_as_scraper():
    t = dump_tables()
    h = tm_dump.to_history(t["transfers"])
    assert h.loc[0, "date"] == date(2025, 7, 10) and h.loc[0, "market_value_at_transfer"] == 60_000_000
    p = tm_dump.to_profiles(t["players"]).iloc[0]
    assert p["date_of_birth"] == date(2003, 3, 4) and p["height_cm"] == 181 and p["position_detail"] == "Central Midfield"
    v = tm_dump.to_values(t["player_valuations"])
    assert list(v.columns) == ["player_id", "date", "market_value", "club_name"]


def test_dump_coverage_needs_same_club_and_year():
    h = tm_dump.to_history(dump_tables()["transfers"])
    eligible = pd.DataFrame({"player_id": [1001, 1001, 1005], "club_id": [11, 11, 11], "season": [2025, 2024, 2025]})
    assert tm_dump.covered(eligible, h).tolist() == [True, False, False]


def test_league_stats_and_matching():
    seasons = ss.parse_tournament_seasons(fixture_json("sofascore_tournament_seasons.json"))
    assert seasons["25/26"] == 76986
    rows = pd.DataFrame(ss.parse_league_stats(fixture_json("sofascore_league_stats.json"))).assign(league="ENG", season=2025)
    assert rows.loc[0, "tackles"] == 61 and rows.loc[0, "team_name"] == "Chelsea"
    players = pd.DataFrame({
        "player_id": [1005, 1006, 1007], "player_name": ["Rival Mid", "João Silva", "Nobody"], "season": [2026, 2026, 2026],
        "origin_league_top5": ["ENG", "ENG", "ENG"], "other_club_name": ["Chelsea FC", "Arsenal FC", "Chelsea FC"],
    })
    m = ss.match_league_players(players, rows).set_index("player_id")
    assert m.loc[1005, "sofascore_id"] == 501            # último nome, no clube de origem
    assert m.loc[1006, "sofascore_id"] == 777            # nome igual, mudou de clube a meio
    assert m.loc[1007, "match_status"] == "not_found"
    assert "Cole Palmer (502, Chelsea)" in m.loc[1007, "candidates"]          # plantel do clube de origem, para ligar à mão
    # Emprestado a um clube de outra liga das 5: encontra-se pelo nome nas 5 ligas.
    loan = pd.DataFrame([{"league": "GER", "season": 2025, "sofascore_id": 888, "sofascore_name": "Nobody", "team_id": 1, "team_name": "Mainz 05"}])
    assert ss.match_league_players(players, pd.concat([rows, loan])).set_index("player_id").loc[1007, "sofascore_id"] == 888
    m = ss.match_league_players(players, rows, manual={(1007, 2026): 501}).set_index("player_id")
    assert m.loc[1007, "sofascore_id"] == 501 and m.loc[1007, "match_status"] == "manual"


def test_league_stats_url_excludes_goalkeepers():
    url = ss.league_stats_url(17, 76986, 100)
    assert "offset=100" in url and "filters=position.in.D~M~F" in url and "tackles" in url


def test_uefa_ranking():
    coefs = uefa.parse_ranking(fixture_text("uefa_crank2025.html"))
    assert set(coefs["country"]) == {"England", "Italy", "Portugal"}
    assert coefs[(coefs["country"] == "Portugal") & (coefs["season_label"] == "24/25")]["coefficient"].item() == 12.55
    r = uefa.five_year_ranking(coefs, [2025]).set_index("country")
    assert round(r.loc["England", "uefa_5y"], 3) == 115.196 and r.loc["England", "uefa_rank"] == 1
    assert r.loc["Portugal", "seasons_found"] == 5


def test_league_matching_name_variants():
    rows = pd.DataFrame([
        {"league": "ITA", "season": 2022, "sofascore_id": 1, "sofascore_name": "Kim Min-jae", "team_name": "SSC Napoli"},
        {"league": "ITA", "season": 2022, "sofascore_id": 2, "sofascore_name": "Joakim Mæhle", "team_name": "Atalanta"},
        {"league": "ENG", "season": 2022, "sofascore_id": 3, "sofascore_name": "Javier Hernández", "team_name": "West Ham United"},
        {"league": "ESP", "season": 2022, "sofascore_id": 4, "sofascore_name": "Rafinha Alcântara", "team_name": "Celta Vigo"},
        {"league": "ESP", "season": 2022, "sofascore_id": 5, "sofascore_name": "Abner Vinícius", "team_name": "Real Betis"},
        {"league": "ESP", "season": 2022, "sofascore_id": 6, "sofascore_name": "Pedro Porro", "team_name": "Real Betis"},
        {"league": "ESP", "season": 2022, "sofascore_id": 7, "sofascore_name": "Pedro González", "team_name": "Real Betis"},
    ])
    players = pd.DataFrame({
        "player_id": [10, 11, 12, 13, 14, 15],
        "player_name": ["Min-jae Kim", "Joakim Maehle", "Chicharito", "Rafinha", "Abner", "Pedro"],
        "season": [2023] * 6,
        "origin_league_top5": ["ITA", "ITA", "ENG", "ESP", "ESP", "ESP"],
        "other_club_name": ["SSC Napoli", "Atalanta BC", "West Ham United", "FC Barcelona", "Real Betis Balompié", "Real Betis Balompié"],
    })
    m = ss.match_league_players(players, rows).set_index("player_id")
    assert m.loc[10, "sofascore_id"] == 1            # mesmas palavras, outra ordem
    assert m.loc[11, "sofascore_id"] == 2            # 'æ' -> 'ae'
    assert m.loc[12, "sofascore_id"] == 3            # alcunha conhecida
    assert m.loc[13, "sofascore_id"] == 4            # nome contido, emprestado na mesma liga
    assert m.loc[14, "sofascore_id"] == 5            # nome contido, no clube de origem
    assert m.loc[15, "match_status"] == "ambiguous"  # dois 'Pedro' no clube: não se liga


def test_normalize_name_keeps_nordic_letters():
    assert ss.normalize_name("Joakim Mæhle") == "joakim maehle"
    assert ss.normalize_name("Martin Ødegaard") == "martin odegaard"
    assert ss.normalize_name("Łukasz Piszczek") == "lukasz piszczek"
