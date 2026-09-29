from datetime import date

import pandas as pd
import pytest

from pechincha.ingest import transfermarkt as tm
from conftest import fixture_json, fixture_text


@pytest.mark.parametrize(
    "text, kind, fee",
    [
        ("€70.00m", "fee", 70_000_000),
        ("€500k", "fee", 500_000),
        ("€750Th.", "fee", 750_000),
        ("£12.5m", "fee", 12_500_000),
        ("Loan fee: €3.00m", "loan", 3_000_000),
        ("loan transfer", "loan", None),
        ("End of loan Jun 30, 2026", "loan_return", None),
        ("free transfer", "free", 0),
        ("?", "unknown", None),
        ("-", "unknown", None),
    ],
)
def test_classify_fee(text, kind, fee):
    out = tm.classify_fee(text)
    assert out["transfer_type"] == kind
    assert out["fee"] == fee


def test_parse_league_transfers():
    df = tm.parse_league_transfers(fixture_text("tm_league_transfers.html"), "ENG", 2026)
    assert list(df.columns) == tm.LEAGUE_TRANSFER_COLUMNS
    arrivals = df[df["direction"] == "in"].set_index("player_id")
    assert set(arrivals.index) == {1001, 1002, 1003, 1004, 1005}
    joao = arrivals.loc[1001]
    assert joao["club_id"] == 11 and joao["club_name"] == "Arsenal FC"
    assert joao["player_name"] == "João Silva"
    assert joao["age"] == 23 and joao["position_short"] == "CF"
    assert joao["market_value_listed"] == 60_000_000
    assert joao["other_club_id"] == 294 and joao["other_club_name"] == "SL Benfica"
    assert joao["other_club_country"] == "Portugal"
    assert joao["fee"] == 70_000_000 and joao["tm_transfer_id"] == 5001
    assert arrivals.loc[1003, "transfer_type"] == "loan"
    out = df[df["direction"] == "out"]
    assert set(out["player_id"]) == {1005, 1006}
    assert set(out["club_id"]) == {11, 631}


def test_transfer_history_and_market_value():
    hist = tm.parse_transfer_history(fixture_json("tm_transfer_history_1001.json"), 1001)
    first = hist.iloc[0]
    assert first["date"] == date(2026, 7, 10)
    assert first["to_club_id"] == 11 and first["from_club_id"] == 294
    assert first["fee"] == 70_000_000 and first["market_value_at_transfer"] == 60_000_000

    values = tm.parse_market_values(fixture_json("tm_market_values_1001.json"), 1001)
    assert len(values) == 3
    mv, when = tm.market_value_before(values, date(2026, 7, 10))
    assert mv == 55_000_000 and when == date(2026, 6, 1)
    assert tm.market_value_before(values, date(2020, 1, 1)) == (None, None)


def test_parse_profile():
    p = tm.parse_profile(fixture_text("tm_profile_1001.html"), 1001)
    assert p["date_of_birth"] == date(2003, 3, 4)
    assert p["contract_expires"] == date(2031, 6, 30)
    assert p["height_cm"] == 186
    assert p["foot"] == "right"
    assert p["position_detail"] == "Attack - Centre-Forward"


def test_league_url():
    lg = tm.League(key="ENG", tm_code="GB1", tm_slug="premier-league", soccerdata="ENG-Premier League")
    assert tm.league_transfers_url(lg, 2026) == (
        "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/plus/?saison_id=2026&s_w=s&leihe=1&intern=0"
    )
