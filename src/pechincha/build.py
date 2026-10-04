"""Fase 2: limpeza e junção numa tabela por compra.

Só lê o que a Fase 1 deixou em ``data/interim/`` e na cache ``data/raw/``:
não faz nenhum pedido à rede. ``uv run main.py build`` grava:

- ``data/processed/purchases.parquet`` (e ``.csv``): uma linha por compra
  (verão, jogador, clube comprador), com o preço, o perfil do Transfermarkt,
  as estatísticas de liga da época anterior (Understat + Sofascore, totais e
  por 90 minutos), a força do clube e da liga de origem e as regras do modelo.
- ``data/processed/league_tables.parquet``: classificações das 5 ligas por
  época, refeitas a partir dos resultados em cache do Understat.
- ``data/manual/understat_links.csv``: compras que o Understat não liga pelo
  nome; preenche-se ``understat_id`` à mão e o build seguinte usa-o.

A chave de uma compra é (verão, jogador, clube). Há jogadores comprados duas
vezes no mesmo verão por clubes diferentes (opção de compra exercida e venda
logo a seguir, ex.: Cucurella 2019): as duas compras são reais e ficam, com
``bought_twice_in_summer``. As ligações ao Sofascore e ao Understat são do
jogador (verão, jogador), por isso juntam-se depois de tiradas as repetidas.
"""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .config import Config
from .ingest.sofascore import _CLUB_NOISE, normalize_name
from .pipeline import _manual_links, load, save

log = logging.getLogger(__name__)

# Liga do soccerdata/Understat -> código do config; país no ranking UEFA.
UNDERSTAT_LEAGUES = {"ENG-Premier League": "ENG", "ESP-La Liga": "ESP", "ITA-Serie A": "ITA", "GER-Bundesliga": "GER", "FRA-Ligue 1": "FRA"}
UEFA_COUNTRY = {"ENG": "England", "ESP": "Spain", "ITA": "Italy", "GER": "Germany", "FRA": "France"}

# Clubes do Transfermarkt cujo nome não tem palavras em comum com o do Understat.
CLUB_ALIASES = {
    "Inter Milan": "Inter",
    "RCD Espanyol Barcelona": "Espanyol",
    "Borussia Mönchengladbach": "Borussia M.Gladbach",
    "1.FC Köln": "FC Cologne",
    "Stade Rennais FC": "Rennes",
    "Stade Brestois 29": "Brest",
    "1.FC Nuremberg": "Nuernberg",
}

# Equipas do Sofascore -> Understat, quando as palavras não chegam ("M'gladbach" ia para o Dortmund).
SOFASCORE_TEAM_ALIASES = {
    "1. FC Köln": "FC Cologne",
    "1. FC Nürnberg": "Nuernberg",
    "Borussia M'gladbach": "Borussia M.Gladbach",
    "Olympique Lyonnais": "Lyon",
    "Stade Brestois": "Brest",
    "Stade Rennais": "Rennes",
    "Real Racing Club": "Racing Santander",
}

POSITION_GROUP = {
    "Centre-Back": "DEF", "Left-Back": "DEF", "Right-Back": "DEF",
    "Defensive Midfield": "MID", "Central Midfield": "MID", "Attacking Midfield": "MID", "Left Midfield": "MID", "Right Midfield": "MID",
    "Left Winger": "ATT", "Right Winger": "ATT", "Centre-Forward": "ATT", "Second Striker": "ATT",
}

UNDERSTAT_STATS = ["matches", "minutes", "goals", "np_goals", "xg", "np_xg", "assists", "xa", "shots", "key_passes", "xg_chain", "xg_buildup"]
SOFASCORE_STATS = [
    "appearances", "matchesStarted", "minutesPlayed", "goals", "assists", "bigChancesCreated", "keyPasses",
    "accuratePasses", "totalPasses", "accurateFinalThirdPasses", "accurateLongBalls", "totalLongBalls", "accurateCrosses", "totalCross",
    "successfulDribbles", "totalContest", "touches", "possessionLost", "dispossessed", "tackles", "tacklesWon", "interceptions",
    "clearances", "blockedShots", "ballRecovery", "possessionWonAttThird", "dribbledPast", "totalDuelsWon", "groundDuelsWon",
    "aerialDuelsWon", "aerialLost", "duelLost", "errorLeadToShot", "errorLeadToGoal", "fouls", "wasFouled", "yellowCards", "redCards",
]
# Contagens que se mostram também por 90 minutos.
UNDERSTAT_PER90 = ["goals", "np_goals", "xg", "np_xg", "assists", "xa", "shots", "key_passes", "xg_chain", "xg_buildup"]
SOFASCORE_PER90 = [
    "bigChancesCreated", "keyPasses", "accuratePasses", "accurateFinalThirdPasses", "accurateLongBalls", "accurateCrosses",
    "successfulDribbles", "touches", "possessionLost", "tackles", "tacklesWon", "interceptions", "clearances", "blockedShots",
    "ballRecovery", "possessionWonAttThird", "dribbledPast", "totalDuelsWon", "aerialDuelsWon", "fouls", "wasFouled",
]

MANUAL_COLUMNS = ["summer", "player_id", "player_name", "other_club_name", "origin", "match_status", "candidates", "understat_id"]


def run(cfg: Config) -> pd.DataFrame:
    """Constrói e grava a tabela por compra. Devolve-a."""
    inter = cfg.interim_dir
    out = cfg.root / "data" / "processed"
    purchases = load(inter / "transfers_enriched.parquet")
    understat = understat_seasons(load(inter / "understat_player_seasons.parquet"))
    sofa_stats = load(inter / "sofascore_league_player_seasons.parquet")
    sofa_links = apply_sofascore_manual(load(inter / "sofascore_league_matches.parquet"), _manual_links(cfg, "league"), sofa_stats)
    ranking = load(inter / "uefa_country_ranking.parquet")

    clubs = club_map(load(inter / "tm_top5_clubs.parquet"), understat)
    tables = league_tables(cfg.cache_dir / "understat", understat)
    links = link_understat(purchases, understat, sofa_links, sofa_stats, clubs, read_manual_links(cfg))
    df = build_purchases(purchases, understat, sofa_stats, sofa_links, links, clubs, tables, ranking)

    save(tables, out / "league_tables.parquet")
    save(df, out / "purchases.parquet")
    df.to_csv(out / "purchases.csv", index=False)
    write_manual_links(cfg, links, purchases)
    _log_summary(df)
    return df


def apply_sofascore_manual(links: pd.DataFrame, manual: dict[tuple[int, int], int], stats: pd.DataFrame) -> pd.DataFrame:
    """Aplica as ligações à mão de ``sofascore_links.csv`` sem voltar a correr a recolha.

    Também corrige ligações erradas: uma linha preenchida no CSV ganha à do
    passo ``sofascore_leagues`` (ex.: Luis Suárez do Granada, 2022, que o nome
    exato ligava ao do Atlético).
    """
    links = links.copy()
    names = stats.drop_duplicates("sofascore_id").set_index("sofascore_id")["sofascore_name"]
    for (player_id, summer), sofascore_id in manual.items():
        rows = (links["player_id"] == player_id) & (links["summer"] == summer)
        links.loc[rows, ["sofascore_id", "match_status", "sofascore_name"]] = [sofascore_id, "manual", names.get(sofascore_id)]
    return links


# --------------------------------------------------------------------------- Understat


def understat_seasons(u: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ``lg`` (código do config) e ``season_start`` (1819 -> 2018)."""
    u = u.copy()
    u["lg"] = u["league"].map(UNDERSTAT_LEAGUES)
    u["season_start"] = 2000 + u["season"].astype(str).str[:2].astype(int)
    u["name_norm"] = u["player"].fillna("").map(_name)
    return u


def _name(name: str) -> str:
    """Nome normalizado sem apóstrofos ("Stanley N'Soki" -> 'stanley nsoki')."""
    return normalize_name(name).replace("'", "")


def _team_words(name: str) -> set[str]:
    return set(normalize_name(name).split()) - _CLUB_NOISE


def club_map(tm_clubs: pd.DataFrame, understat: pd.DataFrame) -> pd.DataFrame:
    """Clube do Transfermarkt -> equipa do Understat, por liga e época.

    Dentro de cada liga e época, cada clube fica com a equipa com mais palavras
    em comum (``CLUB_ALIASES`` para os que não têm nenhuma) e cada equipa só
    serve um clube.
    """
    teams = understat[["lg", "season_start", "team"]].drop_duplicates()
    out = []
    for (lg, season), g in tm_clubs.groupby(["league", "season"]):
        pool = teams[(teams["lg"] == lg) & (teams["season_start"] == season)]["team"].tolist()
        scores = []
        for club in g.itertuples(index=False):
            alias = CLUB_ALIASES.get(club.club_name)
            wa = _team_words(alias or club.club_name)
            for t in pool:
                wb = _team_words(t)
                score = 2.0 if alias == t else (len(wa & wb) / len(wa | wb) if wa | wb else 0.0)
                if score > 0:
                    scores.append((score, club.club_id, club.club_name, t))
        used_clubs, used_teams = set(), set()
        for score, club_id, club_name, t in sorted(scores, key=lambda x: -x[0]):
            if club_id in used_clubs or t in used_teams:
                continue
            used_clubs.add(club_id)
            used_teams.add(t)
            out.append({"league": lg, "season": season, "club_id": club_id, "club_name": club_name, "understat_team": t})
        for club in g.itertuples(index=False):
            if club.club_id not in used_clubs and pool:
                log.warning("Clube sem equipa no Understat: %s (%s %s)", club.club_name, lg, season)
    return pd.DataFrame(out, columns=["league", "season", "club_id", "club_name", "understat_team"])


def league_tables(understat_dir: Path, understat: pd.DataFrame) -> pd.DataFrame:
    """Classificação de cada liga e época a partir dos jogos em cache do Understat.

    Ordena por pontos por jogo, diferença de golos e golos marcados (a Ligue 1
    de 2019/20 acabou mais cedo e foi decidida por pontos por jogo). Não conta
    desempates por confronto direto nem pontos retirados por castigo.
    """
    league_ids = understat[["league_id", "lg"]].drop_duplicates()
    rows = []
    for league_id, lg in league_ids.itertuples(index=False):
        for season in sorted(understat["season_start"].unique()):
            path = understat_dir / f"league_{league_id}_season_{season}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for team in data.get("teams", {}).values():
                h = team.get("history", [])
                rows.append({
                    "league": lg, "season": int(season), "understat_team": team["title"], "played": len(h),
                    "points": sum(m["pts"] for m in h), "goals_for": sum(m["scored"] for m in h),
                    "goals_against": sum(m["missed"] for m in h),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["goal_diff"] = df["goals_for"] - df["goals_against"]
    df["ppg"] = df["points"] / df["played"].where(df["played"] > 0)
    df = df.sort_values(["league", "season", "ppg", "goal_diff", "goals_for"], ascending=[True, True, False, False, False])
    df["position"] = df.groupby(["league", "season"]).cumcount() + 1
    df["teams"] = df.groupby(["league", "season"])["position"].transform("max")
    return df.reset_index(drop=True)


def link_understat(
    purchases: pd.DataFrame,
    understat: pd.DataFrame,
    sofa_links: pd.DataFrame,
    sofa_stats: pd.DataFrame,
    clubs: pd.DataFrame,
    manual: dict[tuple[int, int], int] | None = None,
) -> pd.DataFrame:
    """Liga cada (verão, jogador) ao jogador do Understat na época anterior.

    Procura com o nome do Transfermarkt e o do Sofascore (quando ligado). Por
    ordem, o primeiro critério com um só jogador:

    1. mesmo nome no clube de origem, na liga de origem, nas 5 ligas;
    2. mesmas palavras noutra ordem, na liga e nas 5 ligas;
    3. mesmo último nome no clube de origem ('Dani Parejo' / 'Daniel Parejo');
    4. um nome contém o outro, no clube de origem ('Matías Soulé' / 'Matìas Soulè Malvano');
    5. na equipa onde o Sofascore o põe nessa época (emprestado, ex.: Emerson no
       Betis): um nome contém o outro, nome quase igual ('Zaydou Youssef'),
       mais palavras do nome em comum ('Amath
       Ndiaye' / 'Amath Diedhiou') ou os mesmos minutos (±1%).

    Os minutos só servem na equipa certa e com margem curta: no clube de origem,
    com ±10%, ligavam jogadores errados (Estupiñán ao Doucouré do Watford).

    Um critério com vários jogadores para a procura (``ambiguous``).
    ``manual`` = {(player_id, verão): understat_id}, de ``understat_links.csv``.
    """
    manual = manual or {}
    sofa = sofa_links.drop_duplicates(["player_id", "summer"]).set_index(["player_id", "summer"])
    sofa_rows = {key: g for key, g in sofa_stats.groupby(["sofascore_id", "season"])}
    team_of = clubs.set_index(["league", "season", "club_id"])["understat_team"]
    by_season = {s: g for s, g in understat.groupby("season_start")}
    # Comprado duas vezes no verão: a primeira compra diz o clube onde jogou antes.
    players = purchases.sort_values("transfer_date", na_position="last").drop_duplicates(["season", "player_id"])
    out = []
    for p in players.itertuples(index=False):
        summer, prev = int(p.season), int(p.season) - 1
        rec = {"player_id": p.player_id, "summer": summer, "player_name": p.player_name, "understat_id": None,
               "understat_name": None, "match_status": "not_found", "match_rule": None, "candidates": None}
        sofa_name, sofa_id = None, None
        if (p.player_id, summer) in sofa.index:
            s = sofa.loc[(p.player_id, summer)]
            sofa_name = s["sofascore_name"] if isinstance(s["sofascore_name"], str) else None
            sofa_id = s["sofascore_id"] if pd.notna(s["sofascore_id"]) else None
        rec["sofascore_linked"] = sofa_id is not None
        if (p.player_id, summer) in manual:
            uid = manual[(p.player_id, summer)]
            row = understat[understat["player_id"] == uid]
            rec.update(understat_id=uid, understat_name=row["player"].iloc[0] if not row.empty else None, match_status="manual", match_rule="manual")
            out.append(rec)
            continue
        everywhere = by_season.get(prev)
        if everywhere is None:
            out.append(rec)
            continue
        pool = everywhere[everywhere["lg"] == p.origin_league_top5]
        other_club = p.other_club_id if pd.notna(p.other_club_id) else -1
        team = team_of.get((p.origin_league_top5, prev, int(other_club)))
        in_club = pool["team"] == team
        names = {_name(p.player_name)} | ({_name(sofa_name)} if sofa_name else set())
        words = [frozenset(n.split()) for n in names]
        lasts = {n.split()[-1] for n in names if n}
        pool_words = pool["name_norm"].map(lambda n: frozenset(n.split()))
        all_words = everywhere["name_norm"].map(lambda n: frozenset(n.split()))
        nested = pool["name_norm"].map(lambda n: bool(n) and any(w and (w <= set(n.split()) or set(n.split()) <= w) for w in words))
        tiers = [
            ("name_club", pool[in_club & pool["name_norm"].isin(names)]),
            ("name_league", pool[pool["name_norm"].isin(names)]),
            ("name_top5", everywhere[everywhere["name_norm"].isin(names)]),
            ("words_league", pool[pool_words.isin(words)]),
            ("words_top5", everywhere[all_words.isin(words)]),
            ("last_name_club", pool[in_club & pool["name_norm"].str.split().str[-1].isin(lasts)]),
            ("nested_club", pool[in_club & nested]),
        ]
        # Onde jogou segundo o Sofascore (pode não ser o clube de origem: empréstimos).
        played = sofa_rows.get((sofa_id, prev), pd.DataFrame()) if sofa_id is not None else pd.DataFrame()
        name_words = {w for n in names for w in n.split() if len(w) >= 3}
        for row in played.itertuples(index=False):
            lg_pool = everywhere[everywhere["lg"] == row.league]
            at = lg_pool[lg_pool["team"] == _closest_team(row.team_name, lg_pool["team"])]
            if at.empty:
                continue
            at_nested = at["name_norm"].map(lambda n: bool(n) and any(w and (w <= set(n.split()) or set(n.split()) <= w) for w in words))
            shared = at["name_norm"].map(lambda n: len(name_words & set(n.split())))
            at_word = (shared > 0) & (shared == shared.max())  # 'Zaydou Youssouf': Zaydou Youssef, não Youssouf Sabaly
            at_minutes = (at["minutes"] - row.minutesPlayed).abs() <= max(0.01 * row.minutesPlayed, 10)
            ratio = at["name_norm"].map(lambda n: max(SequenceMatcher(None, n, x).ratio() for x in names))
            at_similar = (ratio >= 0.85) & (ratio == ratio.max())
            tiers += [("nested_sofa_team", at[at_nested]), ("similar_sofa_team", at[at_similar]),
                      ("word_sofa_team", at[at_word]), ("minutes_sofa_team", at[at_minutes])]
        for rule, cand in tiers:
            ids = cand["player_id"].unique()
            if len(ids) == 1:
                rec.update(understat_id=int(ids[0]), understat_name=cand["player"].iloc[0], match_status="matched", match_rule=rule)
                break
            if len(ids) > 1:
                rec["match_status"] = "ambiguous"
                rec["candidates"] = _describe(cand)
                break
        if rec["match_status"] == "not_found" and team is not None:
            last = pool[pool["name_norm"].str.split().str[-1].isin(lasts)]
            rec["candidates"] = _describe(pd.concat([last, pool[in_club]]).drop_duplicates("player_id")) or None
        out.append(rec)
    return pd.DataFrame(out)


def _closest_team(team: str, teams: pd.Series) -> str | None:
    """Equipa do Understat com mais palavras em comum com a do Sofascore (pelo menos 1/3)."""
    alias = SOFASCORE_TEAM_ALIASES.get(team)
    if alias is not None:
        return alias if (teams == alias).any() else None
    wa = _team_words(team or "")
    best, score = None, 1 / 3
    for t in teams.dropna().unique():
        wb = _team_words(t)
        j = len(wa & wb) / len(wa | wb) if wa | wb else 0.0
        if j >= score:
            best, score = t, j
    return best


def _describe(cand: pd.DataFrame, limit: int = 5) -> str:
    rows = cand.drop_duplicates("player_id").head(limit)
    return "; ".join(f"{r.player} ({r.player_id}, {r.team}, {r.minutes} min)" for r in rows.itertuples())


def _manual_path(cfg: Config) -> Path:
    return cfg.root / "data" / "manual" / "understat_links.csv"


def read_manual_links(cfg: Config) -> dict[tuple[int, int], int]:
    """Ligações preenchidas à mão: {(player_id, verão): understat_id}."""
    path = _manual_path(cfg)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["understat_id"] = pd.to_numeric(df["understat_id"], errors="coerce")
    df = df[df["understat_id"].notna()]
    return {(int(r.player_id), int(r.summer)): int(r.understat_id) for r in df.itertuples()}


def write_manual_links(cfg: Config, links: pd.DataFrame, purchases: pd.DataFrame) -> None:
    """Reescreve o CSV com as ligações à mão e as que ficaram por fazer.

    Só entram as compras que o Sofascore ligou: as outras, em regra, não
    jogaram na liga na época anterior e o Understat também não as tem.
    """
    path = _manual_path(cfg)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=MANUAL_COLUMNS)
    filled = old[pd.to_numeric(old["understat_id"], errors="coerce").notna()]
    info = purchases.drop_duplicates(["season", "player_id"]).rename(columns={"season": "summer", "origin_league_top5": "origin"})
    todo = links[(links["match_status"] != "matched") & (links["match_status"] != "manual") & links["sofascore_linked"].fillna(False).astype(bool)]
    todo = todo.merge(info[["summer", "player_id", "other_club_name", "origin"]], on=["summer", "player_id"], how="left")
    new = todo.reindex(columns=MANUAL_COLUMNS)
    df = pd.concat([filled, new], ignore_index=True).drop_duplicates(["player_id", "summer"]).sort_values(["summer", "player_name"])
    df["understat_id"] = pd.to_numeric(df["understat_id"], errors="coerce").astype("Int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Understat: %d compras por ligar à mão em %s", int(df["understat_id"].isna().sum()), path)


# --------------------------------------------------------------------------- tabela final


def _season_totals(stats: pd.DataFrame, id_col: str, season_col: str, cols: list[str]) -> pd.DataFrame:
    """Soma por jogador e época as linhas de várias ligas (mudou de liga em janeiro)."""
    cols = [c for c in cols if c in stats]
    agg = stats.groupby([id_col, season_col])[cols].sum(min_count=1)
    agg["leagues"] = stats.groupby([id_col, season_col])[stats.columns[0]].size()
    return agg


def build_purchases(
    purchases: pd.DataFrame,
    understat: pd.DataFrame,
    sofa_stats: pd.DataFrame,
    sofa_links: pd.DataFrame,
    understat_links: pd.DataFrame,
    clubs: pd.DataFrame,
    tables: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    df = purchases.drop_duplicates(["season", "player_id", "club_id"]).copy()
    df = df.rename(columns={"season": "summer"})
    df["stats_season"] = df["summer"] - 1
    df["bought_twice_in_summer"] = df.duplicated(["summer", "player_id"], keep=False)
    df["position_group"] = df["position"].map(POSITION_GROUP)
    dates = pd.to_datetime(df["transfer_date"], errors="coerce")
    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age_at_transfer"] = ((dates - dob).dt.days / 365.25).round(2)

    # Ligações do jogador: (verão, jogador), sem repetidos, para não duplicar compras.
    sofa = sofa_links.drop_duplicates(["player_id", "summer"])[["player_id", "summer", "sofascore_id", "match_status"]]
    df = df.merge(sofa.rename(columns={"match_status": "sofascore_match"}), on=["player_id", "summer"], how="left")
    us = understat_links[["player_id", "summer", "understat_id", "match_status", "match_rule"]]
    df = df.merge(us.rename(columns={"match_status": "understat_match", "match_rule": "understat_rule"}), on=["player_id", "summer"], how="left")
    df["sofascore_id"] = pd.to_numeric(df["sofascore_id"], errors="coerce").astype("Int64")
    df["understat_id"] = pd.to_numeric(df["understat_id"], errors="coerce").astype("Int64")

    # Estatísticas de liga da época anterior (todas as ligas das 5 em que jogou).
    ust = _season_totals(understat, "player_id", "season_start", UNDERSTAT_STATS).add_prefix("us_")
    ust.index = ust.index.set_names(["understat_id", "stats_season"])
    df = df.merge(ust.reset_index().astype({"understat_id": "Int64"}), on=["understat_id", "stats_season"], how="left")
    sst = _season_totals(sofa_stats, "sofascore_id", "season", SOFASCORE_STATS).add_prefix("ss_")
    sst.index = sst.index.set_names(["sofascore_id", "stats_season"])
    rating = (sofa_stats.assign(_w=sofa_stats["rating"] * sofa_stats["minutesPlayed"])
              .groupby(["sofascore_id", "season"])[["_w", "minutesPlayed"]].sum())
    sst["ss_rating"] = rating["_w"] / rating["minutesPlayed"].where(rating["minutesPlayed"] > 0)
    df = df.merge(sst.reset_index().astype({"sofascore_id": "Int64"}), on=["sofascore_id", "stats_season"], how="left")

    df["minutes_before"] = df["ss_minutesPlayed"].fillna(df["us_minutes"]).fillna(0)
    # As duas fontes contam minutos de liga: muito diferentes = uma das ligações está errada.
    gap = (df["us_minutes"] - df["ss_minutesPlayed"]).abs()
    df["link_conflict"] = (gap > (0.25 * df[["us_minutes", "ss_minutesPlayed"]].max(axis=1)).clip(lower=200)).fillna(False).astype(bool)
    us90 = 90 / df["us_minutes"].where(df["us_minutes"] > 0)
    ss90 = 90 / df["ss_minutesPlayed"].where(df["ss_minutesPlayed"] > 0)
    per90 = {f"us_{c}_p90": df[f"us_{c}"] * us90 for c in UNDERSTAT_PER90} | {f"ss_{c}_p90": df[f"ss_{c}"] * ss90 for c in SOFASCORE_PER90}
    df = pd.concat([df, pd.DataFrame(per90, index=df.index)], axis=1)
    df["ss_pass_accuracy"] = df["ss_accuratePasses"] / df["ss_totalPasses"].where(df["ss_totalPasses"] > 0)
    df["ss_dribble_success"] = df["ss_successfulDribbles"] / df["ss_totalContest"].where(df["ss_totalContest"] > 0)
    duels = df["ss_totalDuelsWon"] + df["ss_duelLost"]
    df["ss_duels_won_pct"] = df["ss_totalDuelsWon"] / duels.where(duels > 0)

    # Força do clube de origem: posição na liga na época anterior.
    team = clubs.rename(columns={"league": "origin_league_top5", "season": "stats_season", "club_id": "other_club_id"})
    df = df.merge(team[["origin_league_top5", "stats_season", "other_club_id", "understat_team"]], on=["origin_league_top5", "stats_season", "other_club_id"], how="left")
    pos = tables.rename(columns={"league": "origin_league_top5", "season": "stats_season", "position": "origin_league_position",
                                 "teams": "origin_league_teams", "ppg": "origin_ppg"})
    df = df.merge(pos[["origin_league_top5", "stats_season", "understat_team", "origin_league_position", "origin_league_teams", "origin_ppg"]],
                  on=["origin_league_top5", "stats_season", "understat_team"], how="left")
    df = df.rename(columns={"understat_team": "origin_understat_team"})

    # Força da liga de origem: ranking UEFA de países no verão da compra.
    uefa = ranking.rename(columns={"country": "origin_country_uefa", "uefa_rank": "origin_uefa_rank", "uefa_5y": "origin_uefa_5y"})
    df["origin_country_uefa"] = df["origin_league_top5"].map(UEFA_COUNTRY)
    df = df.merge(uefa[["summer", "origin_country_uefa", "origin_uefa_rank", "origin_uefa_5y"]], on=["summer", "origin_country_uefa"], how="left")

    # Inflação do mercado: mediana dos preços conhecidos de cada verão, relativa à do verão mais recente.
    median = df[df["fee_known"]].groupby("summer")["fee"].median()
    df["fee_index"] = df["summer"].map(median / median.loc[median.index.max()])
    df["fee_adjusted"] = df["fee"] / df["fee_index"]

    # Regras do modelo (decisões de 04/10/2026).
    df["no_minutes_before"] = df["minutes_before"] <= 0
    reason = pd.Series(None, index=df.index, dtype=object)
    reason[df["no_minutes_before"]] = "no_league_minutes_before"
    reason[~df["fee_known"].astype(bool)] = "unknown_fee"
    df["model_exclusion"] = reason
    df["in_price_model"] = reason.isna()
    df["in_ranking"] = df["in_price_model"] & (df["summer"] == df["summer"].max())
    df["in_phase5"] = df["summer"] == df["summer"].max()
    return df.sort_values(["summer", "league", "club_name", "player_name"]).reset_index(drop=True)


def _log_summary(df: pd.DataFrame) -> None:
    log.info("Compras: %d (%d no modelo, %d no ranking)", len(df), int(df["in_price_model"].sum()), int(df["in_ranking"].sum()))
    log.info("Excluídas do modelo: %s", df["model_exclusion"].value_counts().to_dict())
    for r in df[df["link_conflict"]].itertuples():
        log.warning("Minutos diferentes no Understat e no Sofascore: %s (%s), %s vs %s", r.player_name, r.summer, r.us_minutes, r.ss_minutesPlayed)
    linked = df["minutes_before"] > 0
    log.info("Com minutos antes: %d; sem Understat: %d; sem posição do clube de origem: %d",
             int(linked.sum()), int((linked & df["understat_id"].isna()).sum()), int(df["origin_league_position"].isna().sum()))
