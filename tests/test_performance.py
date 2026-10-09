"""Fase 5 (performance): jogos por equipa, índice custo/desempenho e a execução completa, sem rede."""

import numpy as np
import pandas as pd
import pytest

from pechincha import build
from pechincha.performance import analysis as perf
from pechincha.pipeline import save
from test_eda import purchases as eda_purchases

# Clubes de 2026: (club_id, nome no Transfermarkt, liga, equipa no Understat).
CLUBS = [(1, "Arsenal FC", "ENG", "Arsenal"), (2, "Chelsea FC", "ENG", "Chelsea"), (3, "SSC Napoli", "ITA", "Napoli"), (4, "Olympique Lyon", "FRA", "Lyon")]
LEAGUE = {v: k for k, v in build.UNDERSTAT_LEAGUES.items()}


def schedule_row(game_id, date, lg, home, away, hg, ag, result=True):
    return {"league": LEAGUE[lg], "season": "2627", "game": f"{date} {home}-{away}", "game_id": game_id, "date": f"{date} 15:00:00",
            "home_team": home, "away_team": away, "home_goals": hg, "away_goals": ag, "home_xg": None if hg is None else hg + 0.5, "away_xg": None if ag is None else ag + 0.2,
            "is_result": result, "has_data": result}


SCHEDULE = [
    schedule_row(1, "2026-08-22", "ENG", "Arsenal", "Chelsea", 2, 0),
    schedule_row(2, "2026-08-29", "ENG", "Everton", "Arsenal", 1, 1),
    schedule_row(3, "2026-09-05", "ENG", "Arsenal", "Fulham", 0, 1),
    schedule_row(4, "2026-09-12", "ENG", "Chelsea", "Arsenal", 3, 0),
    schedule_row(5, "2026-08-23", "ITA", "Napoli", "Roma", 1, 0),
    schedule_row(6, "2026-08-23", "FRA", "Lyon", "Lens", 0, 0),
    schedule_row(7, "2026-10-17", "ENG", "Arsenal", "Everton", None, None, result=False),
]


def scenario():
    """Compras sintéticas (histórico da EDA) com cinco casos de 2026 feitos à mão, e as tabelas de 2026/27."""
    df = eda_purchases()
    df["in_phase5"] = df["summer"] == 2026
    df["model_exclusion"] = None
    df["understat_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["sofascore_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    cur = df.index[df["summer"] == 2026]
    for k, i in enumerate(cur):
        club_id, name, lg, _ = CLUBS[k % 4]
        df.loc[i, ["club_id", "club_name", "league", "transfer_date"]] = [club_id, name, lg, "2026-07-01"]
        df.loc[i, "understat_id"] = 9000 + k  # sem jogos: não liga pelo nome a ninguém
    p = {k: df.loc[cur[k]] for k in range(5)}
    # 0: Arsenal, 3 jogos de 90 -> amostra mínima. 1: Chelsea, sem id do Understat (liga pelo nome), 45 minutos.
    # 2: Napoli, mas joga pela Roma. 3: Lyon, sem minutos. 4: Arsenal desde 25/08, antes jogou pelo Chelsea.
    df.loc[cur[0], ["understat_id", "sofascore_id"]] = [1000, 5000]
    df.loc[cur[1], "understat_id"] = pd.NA
    df.loc[cur[2], "understat_id"] = 1002
    df.loc[cur[4], ["understat_id", "sofascore_id", "transfer_date"]] = [1004, 5004, "2026-08-25"]

    def pm(game_id, team, player_id, player, minutes):
        g = next(s for s in SCHEDULE if s["game_id"] == game_id)
        return {"league": g["league"], "season": "2627", "game": g["game"], "team": team, "player": player, "game_id": game_id,
                "player_id": player_id, "position": "FW" if minutes >= 60 else "Sub", "minutes": minutes, "goals": 1, "xg": 0.3,
                "assists": 0, "xa": 0.1, "shots": 2, "key_passes": 1, "xg_chain": 0.5, "xg_buildup": 0.2}

    matches = pd.DataFrame([
        pm(1, "Arsenal", 1000, p[0]["player_name"], 90), pm(2, "Arsenal", 1000, p[0]["player_name"], 90), pm(3, "Arsenal", 1000, p[0]["player_name"], 90),
        pm(4, "Chelsea", 1001, p[1]["player_name"] + " Junior", 45),
        pm(5, "Roma", 1002, p[2]["player_name"], 90),
        pm(1, "Chelsea", 1004, p[4]["player_name"], 90), pm(2, "Arsenal", 1004, p[4]["player_name"], 90), pm(3, "Arsenal", 1004, p[4]["player_name"], 90),
    ])
    stats = {c: 10.0 for c in build.SOFASCORE_STATS}
    sofa = pd.DataFrame([
        {"league": "ENG", "season": 2026, "sofascore_id": 5000, "sofascore_name": p[0]["player_name"], "team_name": "Arsenal", "rating": 7.1, **stats, "minutesPlayed": 270.0},
        {"league": "ENG", "season": 2026, "sofascore_id": 5001, "sofascore_name": p[1]["player_name"], "team_name": "Chelsea", "rating": 6.5, **stats, "minutesPlayed": 45.0},
        {"league": "ENG", "season": 2026, "sofascore_id": 5004, "sofascore_name": p[4]["player_name"], "team_name": "Arsenal", "rating": 6.9, **stats, "minutesPlayed": 270.0},
    ])
    teams = sorted({(s["league"], t) for s in SCHEDULE for t in (s["home_team"], s["away_team"])})
    understat = pd.DataFrame([{"league": lg, "season": "2627", "team": t, "player": "x"} for lg, t in teams])
    tm_clubs = pd.DataFrame([{"league": lg, "season": 2026, "club_id": cid, "club_name": name} for cid, name, lg, _ in CLUBS])
    return df, cur, matches, pd.DataFrame(SCHEDULE), sofa, understat, tm_clubs


def test_team_games_two_sides_and_points():
    g = perf.team_games(pd.DataFrame(SCHEDULE))
    assert len(g) == 12  # 6 jogos com resultado, um por equipa; o jogo por jogar fica de fora
    first = g[g["game_id"] == 1].set_index("team")
    assert first.loc["Arsenal", "points"] == 3 and first.loc["Chelsea", "points"] == 0
    assert first.loc["Arsenal", "goals_against"] == 0 and first.loc["Chelsea", "xg_for"] == pytest.approx(0.2)
    assert g.loc[g["game_id"] == 2, "points"].tolist() == [1, 1]
    prog = perf.season_progress(g).set_index("lg")
    assert prog.loc["ENG", "max_team_matches"] == 4 and prog.loc["ENG", "min_team_matches"] == 1


def test_cost_performance_index_and_fee_percentile():
    df = pd.DataFrame({"position_group": ["ATT"] * 4 + ["DEF"], "fee_known": [True, True, True, False, True], "fee": [1e6, 2e6, 3e6, np.nan, 5e6],
                       "score_after": [80.0, 50.0, 20.0, 60.0, 40.0], "qualified": [True, True, True, True, False]})
    out = perf.cost_performance(df)
    assert out["fee_pct_position"].tolist()[:3] == pytest.approx([16.7, 50.0, 83.3])
    assert out["value_index"].tolist()[:3] == pytest.approx([63.3, 0.0, -63.3])
    # Preço desconhecido ou sem a amostra mínima: sem índice.
    assert out["value_index"].iloc[3:].isna().all()
    assert out["fee_per_score_point"].iloc[0] == pytest.approx(12_000, abs=1_000)


def test_same_name_handles_extra_names():
    assert perf._same_name("Noël Aséko", "Noel Aséko Nkili") == 1.0
    assert perf._same_name("Yannik Engelhardt", "Yannick Engelhardt") > 0.9
    assert perf._same_name("Player 12", "Player 171 Junior") < 0.8


def test_run_statuses_team_results_and_outputs(cfg):
    df, cur, matches, schedule, sofa, understat, tm_clubs = scenario()
    save(df, cfg.root / "data" / "processed" / "purchases.parquet")
    inter = cfg.interim_dir
    save(matches, inter / "understat_2026_player_matches.parquet")
    save(schedule, inter / "understat_2026_schedule.parquet")
    save(sofa, inter / "sofascore_league_player_seasons.parquet")
    save(understat, inter / "understat_player_seasons.parquet")
    save(tm_clubs, inter / "tm_top5_clubs.parquet")
    summary = perf.run(cfg)

    out = cfg.root / "outputs" / "performance"
    charts = sorted(p.stem for p in out.glob("*.html"))
    assert charts == ["01_minutes_after", "02_score_before_after", "03_cost_vs_performance", "04_value_ranking", "05_metric_changes", "06_team_with_without"]
    assert {p.stem for p in out.glob("*.json")} == set(charts) | {"performance_2026", "team_results_2026"}
    for table in ["performance_2026", "team_results_2026", "metric_changes"]:
        assert (out / f"{table}.csv").exists(), table
    assert "Small sample" in (out / "01_minutes_after.html").read_text(encoding="utf-8")
    assert "very small" in (out / "summary.md").read_text(encoding="utf-8")

    res = pd.read_parquet(cfg.root / "data" / "processed" / "performance.parquet")
    assert len(res) == 24 and not res.duplicated(perf.KEY).any()
    r = {k: res[res["player_id"] == df.loc[cur[k], "player_id"]].iloc[0] for k in range(5)}
    assert [r[k]["status"] for k in range(5)] == ["qualified", "under_min_minutes", "playing_elsewhere", "no_minutes", "under_min_minutes"]
    assert (res.drop(index=res.index[res["player_id"].isin([r[k]["player_id"] for k in range(5)])])["status"] == "no_minutes").all()

    # 0: 270 minutos no clube novo (mínimo fixo, a equipa só jogou 4); métricas do Understat e do Sofascore.
    assert r[0]["minutes_after"] == 270 and r[0]["min_minutes_required"] == 270 and r[0]["starts_after"] == 3
    assert r[0]["us_xg_p90_after"] == pytest.approx(0.3) and r[0]["ss_tackles_p90_after"] == pytest.approx(10 * 90 / 270)
    assert r[0]["ss_rating_after"] == 7.1 and pd.notna(r[0]["score_after"]) and pd.notna(r[0]["value_index"])
    # Com ele: vitória, empate, derrota; sem ele: a derrota por 3-0.
    assert (r[0]["matches_with"], r[0]["matches_without"]) == (3, 1)
    assert r[0]["ppg_with"] == pytest.approx(4 / 3) and r[0]["ppg_without"] == 0 and r[0]["gd_per_match_without"] == -3

    # 1: sem id do Understat, ligado pelo nome; o Sofascore também pelo nome.
    assert r[1]["understat_link_2026"] == "name" and r[1]["minutes_after"] == 45 and r[1]["ss_minutesPlayed_after"] == 45
    assert pd.isna(r[1]["value_index"])
    # 2: comprado pelo Napoli, a jogar pela Roma.
    assert r[2]["club_elsewhere"] == "Roma" and r[2]["minutes_after"] == 0
    # 4: chegou a 25/08 (3 jogos do Arsenal desde então); jogou pelo Chelsea antes, por isso sem Sofascore.
    assert r[4]["team_matches_since_arrival"] == 3 and r[4]["minutes_after"] == 180 and r[4]["minutes_elsewhere"] == 90
    assert r[4]["ss_mixed_clubs"] and np.isnan(r[4]["ss_minutesPlayed_after"]) and pd.isna(r[4]["club_elsewhere"])
    assert (r[4]["matches_with"], r[4]["matches_without"]) == (2, 1)

    table = pd.read_csv(out / "performance_2026.csv")
    assert table["status"].iloc[0] == "qualified" and {"us_xg_p90_before", "us_xg_p90_after", "pct_us_xg_p90_after"} <= set(table.columns)
    assert summary["season"].set_index("lg").loc["ENG", "max_team_matches"] == 4
