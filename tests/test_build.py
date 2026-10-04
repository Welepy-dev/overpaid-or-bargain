"""Fase 2 (build): ligações ao Understat, classificações e tabela por compra, sem rede."""

import json

import pandas as pd

from pechincha import build


def understat_rows():
    rows = [
        # league, season, team, player, player_id, minutes
        ("ESP-La Liga", "2021", "Barcelona", "Lionel Messi", 2097, 3000),
        ("ESP-La Liga", "2021", "Barcelona", "Jordi Alba", 2093, 2900),
        ("ESP-La Liga", "2021", "Real Betis", "Emerson", 7430, 2998),
        ("ESP-La Liga", "2021", "Valencia", "Daniel Parejo", 2378, 3049),
        ("ESP-La Liga", "2021", "Valencia", "Daniel Wass", 2400, 2500),
        ("ITA-Serie A", "2021", "Inter", "Lautaro Martínez", 7006, 2700),
        ("ITA-Serie A", "2021", "AC Milan", "Rafael Leão", 7193, 2500),
    ]
    u = pd.DataFrame(rows, columns=["league", "season", "team", "player", "player_id", "minutes"])
    u["league_id"] = u["league"].map({"ESP-La Liga": 4, "ITA-Serie A": 2})
    for c in build.UNDERSTAT_STATS:
        if c not in u:
            u[c] = 1
    return build.understat_seasons(u)


def tm_clubs():
    return pd.DataFrame([
        ("ESP", 2020, 131, "FC Barcelona"), ("ESP", 2020, 150, "Real Betis Balompié"), ("ESP", 2020, 1049, "Valencia CF"),
        ("ITA", 2020, 46, "Inter Milan"), ("ITA", 2020, 5, "AC Milan"),
    ], columns=["league", "season", "club_id", "club_name"])


def purchases():
    base = {"window": "summer", "direction": "in", "fee_known": True, "transfer_type": "fee", "position": "Right-Back",
            "transfer_date": "2021-07-01", "date_of_birth": "1999-01-14", "market_value_before": 1e7}
    rows = [
        # Emerson: comprado duas vezes no verão de 2021 (Barcelona ao Betis, depois Tottenham ao Barcelona).
        dict(base, season=2021, league="ENG", club_id=148, club_name="Tottenham Hotspur", player_id=476344, player_name="Emerson Royal",
             other_club_id=131, other_club_name="FC Barcelona", origin_league_top5="ESP", fee=25e6, transfer_date="2021-08-31"),
        dict(base, season=2021, league="ESP", club_id=131, club_name="FC Barcelona", player_id=476344, player_name="Emerson Royal",
             other_club_id=150, other_club_name="Real Betis Balompié", origin_league_top5="ESP", fee=9e6),
        dict(base, season=2021, league="ESP", club_id=1050, club_name="Villarreal CF", player_id=59561, player_name="Dani Parejo",
             other_club_id=1049, other_club_name="Valencia CF", origin_league_top5="ESP", fee=None, fee_known=False),
        dict(base, season=2021, league="ITA", club_id=5, club_name="AC Milan", player_id=1, player_name="Nobody Played",
             other_club_id=46, other_club_name="Inter Milan", origin_league_top5="ITA", fee=1e6),
    ]
    return pd.DataFrame(rows)


def sofascore():
    links = pd.DataFrame([
        (476344, 2021, 856123, "matched", "Emerson Royal"), (476344, 2021, 856123, "matched", "Emerson Royal"),
        (59561, 2021, 1001, "matched", "Dani Parejo"), (1, 2021, None, "not_found", None),
    ], columns=["player_id", "summer", "sofascore_id", "match_status", "sofascore_name"])
    stats = pd.DataFrame([
        ("ESP", 2020, 856123, "Emerson Royal", "Real Betis", 2992, 7.0),
        ("ESP", 2020, 1001, "Dani Parejo", "Valencia", 3040, 7.2),
    ], columns=["league", "season", "sofascore_id", "sofascore_name", "team_name", "minutesPlayed", "rating"])
    for c in build.SOFASCORE_STATS:
        if c not in stats:
            stats[c] = 10
    return links, stats


def test_club_map_uses_aliases_and_one_team_per_club():
    clubs = build.club_map(tm_clubs(), understat_rows()).set_index("club_name")["understat_team"]
    assert clubs["Inter Milan"] == "Inter"  # sem alias ia para o AC Milan
    assert clubs["AC Milan"] == "AC Milan"
    assert clubs["Real Betis Balompié"] == "Real Betis"


def test_link_understat_tiers():
    links, stats = sofascore()
    clubs = build.club_map(tm_clubs(), understat_rows())
    out = build.link_understat(purchases(), understat_rows(), links, stats, clubs).set_index("player_id")
    assert len(out) == 3  # uma ligação por (verão, jogador), mesmo com duas compras
    # Duas compras: conta a primeira (Barcelona ao Betis), onde o Understat o tem só como "Emerson".
    assert out.loc[476344, "understat_id"] == 7430
    assert out.loc[476344, "match_rule"] == "nested_club"
    assert out.loc[59561, "understat_id"] == 2378  # 'Dani' / 'Daniel' Parejo, pelo último nome no clube
    assert out.loc[59561, "match_rule"] == "last_name_club"
    assert out.loc[1, "match_status"] == "not_found"


def test_link_understat_uses_sofascore_team_for_loans():
    # Só a compra do Tottenham: o clube de origem é o Barcelona, mas jogou emprestado no Betis.
    links, stats = sofascore()
    tottenham = purchases()[lambda d: d["club_name"] == "Tottenham Hotspur"]
    out = build.link_understat(tottenham, understat_rows(), links, stats, build.club_map(tm_clubs(), understat_rows()))
    assert out.iloc[0]["understat_id"] == 7430
    assert out.iloc[0]["match_rule"] == "nested_sofa_team"


def test_manual_understat_link_wins():
    links, stats = sofascore()
    clubs = build.club_map(tm_clubs(), understat_rows())
    out = build.link_understat(purchases(), understat_rows(), links, stats, clubs, manual={(1, 2021): 7006}).set_index("player_id")
    assert out.loc[1, "understat_id"] == 7006
    assert out.loc[1, "match_status"] == "manual"


def test_sofascore_manual_overrides_wrong_link():
    links, stats = sofascore()
    fixed = build.apply_sofascore_manual(links, {(59561, 2021): 856123}, stats)
    row = fixed[fixed["player_id"] == 59561].iloc[0]
    assert row["sofascore_id"] == 856123 and row["match_status"] == "manual"


def test_league_tables(tmp_path):
    def match(pts, scored, missed):
        return {"pts": pts, "scored": scored, "missed": missed}
    data = {"teams": {
        "1": {"id": "1", "title": "Barcelona", "history": [match(3, 2, 0), match(1, 1, 1)]},
        "2": {"id": "2", "title": "Valencia", "history": [match(3, 1, 0), match(1, 0, 0)]},
        "3": {"id": "3", "title": "Real Betis", "history": [match(0, 0, 2), match(0, 0, 1)]},
    }}
    (tmp_path / "league_4_season_2020.json").write_text(json.dumps(data), encoding="utf-8")
    t = build.league_tables(tmp_path, understat_rows()).set_index("understat_team")
    # Mesmos pontos: desempata a diferença de golos.
    assert t.loc["Barcelona", "position"] == 1 and t.loc["Valencia", "position"] == 2 and t.loc["Real Betis", "position"] == 3
    assert t.loc["Barcelona", "teams"] == 3


def test_build_purchases_keeps_double_purchases_and_applies_rules():
    links, stats = sofascore()
    u = understat_rows()
    clubs = build.club_map(tm_clubs(), u)
    ulinks = build.link_understat(purchases(), u, links, stats, clubs)
    tables = pd.DataFrame([("ESP", 2020, "Real Betis", 6, 20, 1.5), ("ESP", 2020, "Barcelona", 3, 20, 2.1), ("ESP", 2020, "Valencia", 13, 20, 1.1),
                           ("ITA", 2020, "Inter", 1, 20, 2.4)],
                          columns=["league", "season", "understat_team", "position", "teams", "ppg"])
    ranking = pd.DataFrame([("Spain", 2021, 2, 100.0), ("Italy", 2021, 3, 90.0)], columns=["country", "summer", "uefa_rank", "uefa_5y"])
    df = build.build_purchases(purchases(), u, stats, links, ulinks, clubs, tables, ranking)

    assert len(df) == 4  # as duas compras do Emerson ficam, sem multiplicar linhas
    emerson = df[df["player_id"] == 476344].set_index("club_name")
    assert emerson["bought_twice_in_summer"].all()
    assert emerson.loc["FC Barcelona", "origin_league_position"] == 6  # comprado ao Betis
    assert emerson.loc["Tottenham Hotspur", "origin_league_position"] == 3  # comprado ao Barcelona
    assert (emerson["us_minutes"] == 2998).all() and (emerson["ss_minutesPlayed"] == 2992).all()
    assert emerson["us_goals_p90"].iloc[0] == 90 / 2998

    rows = df.set_index("player_name")
    assert rows.loc["Dani Parejo", "model_exclusion"] == "unknown_fee"
    assert rows.loc["Nobody Played", "model_exclusion"] == "no_league_minutes_before"
    assert rows.loc["Nobody Played", "origin_uefa_rank"] == 3
    assert df["in_price_model"].sum() == 2 and not df["link_conflict"].any()
