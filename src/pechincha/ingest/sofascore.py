"""Sofascore: estatísticas por época, por liga inteira (5 ligas) e por jogador (resto).

5 grandes ligas (substitui o FBref, que perdeu os dados Opta em jan/2026):
``/unique-tournament/{liga}/season/{época}/statistics`` devolve todos os
jogadores de campo da liga, 100 por pedido, com as métricas de ``LEAGUE_FIELDS``
(desarmes, interceções, duelos, passes no último terço, dribles...). Cerca de
6 pedidos por liga e época.

Jogadores que chegam de fora das 5 ligas (só esses, não as ligas inteiras):

1. pesquisa pelo nome e confirmação pela data de nascimento (do Transfermarkt);
2. lista de competições/épocas do jogador;
3. estatísticas agregadas do campeonato nacional nas épocas relevantes para o verão
   (só jogos de liga, como nas 5 ligas: sem taças, provas europeias nem seleções).

Os jogadores que não se ligam vão para ``data/manual/sofascore_links.csv``; o
``sofascore_id`` preenchido lá à mão é usado nas execuções seguintes.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date, datetime, timezone
from urllib.parse import quote

import pandas as pd

from ..config import Config
from ..http import BlockedError, BrowserFetcher, CachedFetcher, FetchError
from ..progress import progress
from ._seasons import short_season_label

log = logging.getLogger(__name__)

# api.sofascore.com passou a dar 403 mesmo dentro do browser; a mesma API servida
# pelo www (mesma origem da página aberta) responde.
API = "https://www.sofascore.com/api/v1"
# Candidatos da pesquisa cujo perfil se abre para comparar a data de nascimento.
MAX_CANDIDATES = 3
# Competições de um país que não são o campeonato: taças, supertaças, formação, estaduais...
_NOT_LEAGUE = re.compile(
    r"cup|copa|coupe|coppa|pokal|beker|kup|puchar|ta[cç]a|troph|trofeo|schaal|shield"
    r"|super ?cup|supercop|superta|superpuchar|s[üu]per kupa|u\d\d\b|primavera|next gen|juvenil|revela"
    r"|all.?star|play.?off|feminin|women|copinha|premier league 2"
    r"|paulista|carioca|mineiro|ga[uú]cho|catarinense|baian|pernambucano|paranaense|goiano|cearense",
    re.IGNORECASE,
)


def fetcher(cfg: Config) -> BrowserFetcher:
    # A API devolve 403 a clientes que não são browsers; os pedidos vão por um Chromium.
    # Abre-se o robots.txt (leve) só para ficar na origem www; a página inicial pode não acabar de carregar.
    h = cfg.http
    return BrowserFetcher(
        cfg.cache_dir, "sofascore", home="https://www.sofascore.com/robots.txt",
        min_interval=cfg.min_interval("sofascore"), timeout=h["timeout_seconds"], max_retries=h["max_retries"],
    )


# Letras que o NFKD não decompõe e que se perderiam ao passar para ASCII ('Mæhle' -> 'mhle').
_TRANSLIT = str.maketrans({"æ": "ae", "Æ": "Ae", "ø": "o", "Ø": "O", "ß": "ss", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
                           "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "œ": "oe", "Œ": "Oe", "ı": "i"})


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", (name or "").translate(_TRANSLIT)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").replace(".", " ").split())


def relevant_season_labels(summer: int) -> set[str]:
    """Etiquetas de época do Sofascore com dados anteriores ao verão ``summer``.

    Ligas europeias usam '25/26'; ligas de ano civil (Brasil, MLS...) usam
    '2025' e '2026' (esta última só com a parte jogada até à transferência).
    """
    yy = summer % 100
    return {f"{(yy - 1) % 100:02d}/{yy:02d}", str(summer - 1), str(summer)}


def parse_search(data: dict) -> list[dict]:
    out = []
    for r in data.get("results", []) or []:
        if r.get("type") not in (None, "player"):
            continue
        e = r.get("entity") or {}
        if not e.get("id"):
            continue
        dob = e.get("dateOfBirthTimestamp")
        out.append(
            {
                "sofascore_id": e["id"],
                "name": e.get("name"),
                "team": (e.get("team") or {}).get("name"),
                "date_of_birth": datetime.fromtimestamp(dob, tz=timezone.utc).date() if dob is not None else None,
            }
        )
    return out


def parse_player_birth(data: dict) -> date | None:
    """Data de nascimento de ``/player/{id}`` (a pesquisa não a inclui)."""
    ts = (data.get("player") or {}).get("dateOfBirthTimestamp")
    return datetime.fromtimestamp(ts, tz=timezone.utc).date() if ts is not None else None


def parse_statistics(data: dict) -> dict:
    """Estatísticas de uma competição/época, só valores simples, mais a equipa.

    ``team_national`` separa jogos por seleções (sub-21 incluídas) dos de clube.
    """
    stats = {k: v for k, v in (data.get("statistics") or {}).items() if not isinstance(v, (dict, list))}
    stats.pop("id", None)
    team = data.get("team") or {}
    return {"team_name": team.get("name"), "team_national": team.get("national"), **stats}


def likely_candidates(candidates: list[dict], name: str) -> list[dict]:
    """Os primeiros candidatos com alguma palavra do nome em comum (cada um custa um pedido)."""
    words = set(normalize_name(name).split())
    same = [c for c in candidates if words & set(normalize_name(c["name"] or "").split())]
    return (same or candidates)[:MAX_CANDIDATES]


def pick_candidate(candidates: list[dict], name: str, dob: date | None, club: str | None = None) -> dict | None:
    """Escolhe o jogador certo entre os resultados da pesquisa.

    Com data de nascimento, exige que coincida. Sem ela, só aceita um nome
    exatamente igual e único (para não ligar jogadores errados).
    """
    if dob is not None:
        same = [c for c in candidates if c["date_of_birth"] == dob]
        if len(same) == 1:
            return same[0]
        if len(same) > 1 and club:
            by_club = [c for c in same if c["team"] and normalize_name(club) in normalize_name(c["team"])]
            if len(by_club) == 1:
                return by_club[0]
        return None
    exact = [c for c in candidates if normalize_name(c["name"]) == normalize_name(name)]
    return exact[0] if len(exact) == 1 else None


def is_league(tournament_name: str | None, category: dict | None) -> bool:
    """Campeonato nacional de clubes (de um país, não amador, não taça)."""
    category = category or {}
    if not category.get("alpha2") or "amateur" in (category.get("name") or "").lower():
        return False  # provas internacionais (UEFA, seleções, Libertadores) e amadoras
    return not _NOT_LEAGUE.search(tournament_name or "")


def parse_player_seasons(data: dict) -> list[dict]:
    out = []
    for block in data.get("uniqueTournamentSeasons", []) or []:
        ut = block.get("uniqueTournament") or {}
        league = is_league(ut.get("name"), ut.get("category"))
        for s in block.get("seasons", []) or []:
            out.append(
                {
                    "tournament_id": ut.get("id"),
                    "tournament_name": ut.get("name"),
                    "is_league": league,
                    "season_id": s.get("id"),
                    "season_label": s.get("year"),
                }
            )
    return out


def fetch_players(
    cfg: Config, players: pd.DataFrame, f: CachedFetcher | None = None, manual: dict[tuple[int, int], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``players`` precisa de: player_id (Transfermarkt), player_name, date_of_birth, other_club_name, season.

    ``manual`` são ligações feitas à mão, {(player_id, verão): sofascore_id}.
    Devolve (correspondências, estatísticas). As correspondências registam
    também os jogadores não encontrados, com os candidatos, para ligação manual.
    """
    own = f is None
    f = f or fetcher(cfg)
    try:
        return _fetch_players(cfg, players, f, manual or {})
    finally:
        if own:
            f.close()


def _fetch_players(cfg: Config, players: pd.DataFrame, f: CachedFetcher, manual: dict[tuple[int, int], int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches, stats = [], []
    # Primeiro os jogadores sem pesquisa em cache: se houver um bloqueio, o que falta avança antes.
    cached = players["player_name"].map(lambda n: f.cache_path(f"{API}/search/all?q={quote(n)}&page=0", ".json").exists())
    players = players.iloc[cached.argsort(kind="stable")]
    for p in progress(players.itertuples(index=False), "Sofascore jogadores", total=len(players)):
        dob = p.date_of_birth if isinstance(p.date_of_birth, date) else None
        live = cfg.http["live_max_age_hours"] if p.season >= cfg.season else None
        rec = {"player_id": p.player_id, "summer": p.season, "player_name": p.player_name, "sofascore_id": None, "match_status": "not_found"}
        try:
            if (p.player_id, p.season) in manual:
                rec.update(sofascore_id=manual[(p.player_id, p.season)], match_status="manual")
            else:
                found = parse_search(f.get_json(f"{API}/search/all?q={quote(p.player_name)}&page=0"))
                if dob is not None:
                    for c in likely_candidates(found, p.player_name):
                        if c["date_of_birth"] is None:
                            c["date_of_birth"] = parse_player_birth(f.get_json(f"{API}/player/{c['sofascore_id']}"))
                cand = pick_candidate(found, p.player_name, dob, p.other_club_name)
                if cand is None:
                    rec["match_status"] = "ambiguous" if found else "not_found"
                    rec["candidates"] = describe_candidates(found)
                    continue
                rec.update(sofascore_id=cand["sofascore_id"], sofascore_name=cand["name"], match_status="matched")
            labels = relevant_season_labels(p.season)
            seasons = parse_player_seasons(f.get_json(f"{API}/player/{rec['sofascore_id']}/statistics/seasons", max_age_hours=live))
            for s in (s for s in seasons if s["season_label"] in labels and s["is_league"]):
                url = f"{API}/player/{rec['sofascore_id']}/unique-tournament/{s['tournament_id']}/season/{s['season_id']}/statistics/overall"
                try:
                    body = f.get_json(url, max_age_hours=live)
                except FetchError:
                    continue
                stats.append({"player_id": p.player_id, "sofascore_id": rec["sofascore_id"], "summer": p.season, **s, **parse_statistics(body)})
        except BlockedError:
            raise
        except FetchError as exc:
            log.warning("Sofascore falhou para %s: %s", p.player_name, exc)
            rec["match_status"] = "error"
        finally:
            matches.append(rec)
    return pd.DataFrame(matches), pd.DataFrame(stats)


def describe_candidates(candidates: list[dict], limit: int = 5) -> str:
    """Candidatos em texto para quem liga à mão: 'Nome (id, nascimento, equipa) | ...'."""
    parts = []
    for c in candidates[:limit]:
        extra = ", ".join(str(v) for v in (c.get("sofascore_id"), c.get("date_of_birth"), c.get("team")) if v is not None)
        parts.append(f"{c.get('name')} ({extra})")
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Ligas inteiras (5 grandes ligas)
# --------------------------------------------------------------------------- #

# Métricas pedidas por jogador; os nomes são os da API (lista do ScraperFC).
LEAGUE_FIELDS = (
    "appearances", "matchesStarted", "minutesPlayed", "rating",
    "goals", "assists", "expectedGoals", "expectedAssists", "totalShots", "shotsOnTarget",
    "bigChancesCreated", "keyPasses", "totalAttemptAssist",
    "accuratePasses", "totalPasses", "accurateFinalThirdPasses", "accurateOppositionHalfPasses",
    "accurateLongBalls", "totalLongBalls", "accurateCrosses", "totalCross",
    "successfulDribbles", "totalContest", "touches", "possessionLost", "dispossessed",
    "tackles", "tacklesWon", "interceptions", "clearances", "outfielderBlocks", "blockedShots",
    "ballRecovery", "possessionWonAttThird", "dribbledPast",
    "totalDuelsWon", "groundDuelsWon", "aerialDuelsWon", "aerialLost", "duelLost",
    "errorLeadToShot", "errorLeadToGoal", "fouls", "wasFouled", "yellowCards", "redCards",
)
# Defesas, médios e avançados (sem guarda-redes, fora do âmbito).
OUTFIELD_FILTER = "position.in.D~M~F"


def parse_tournament_seasons(data: dict) -> dict[str, int]:
    """``/unique-tournament/{id}/seasons`` -> {'25/26': id da época}."""
    return {s["year"]: s["id"] for s in data.get("seasons", []) or [] if s.get("year") and s.get("id")}


def league_stats_url(tournament_id: int, season_id: int, offset: int) -> str:
    return (
        f"{API}/unique-tournament/{tournament_id}/season/{season_id}/statistics"
        f"?limit=100&offset={offset}&accumulation=total&fields={'%2C'.join(LEAGUE_FIELDS)}&filters={OUTFIELD_FILTER}"
    )


def parse_league_stats(data: dict) -> list[dict]:
    rows = []
    for r in data.get("results", []) or []:
        player, team = r.get("player") or {}, r.get("team") or {}
        stats = {k: v for k, v in r.items() if k not in ("player", "team") and not isinstance(v, (dict, list))}
        rows.append(
            {"sofascore_id": player.get("id"), "sofascore_name": player.get("name"),
             "team_id": team.get("id"), "team_name": team.get("name"), **stats}
        )
    return rows


def fetch_league_seasons(cfg: Config, seasons: list[int], f: CachedFetcher | None = None) -> pd.DataFrame:
    """Uma linha por jogador de campo, liga e época (``season`` = ano de início)."""
    own = f is None
    f = f or fetcher(cfg)
    rows = []
    try:
        leagues = [lg for lg in cfg.leagues if lg.sofascore]
        for lg in progress(leagues, "Sofascore ligas"):
            try:
                ids = parse_tournament_seasons(f.get_json(f"{API}/unique-tournament/{lg.sofascore}/seasons", max_age_hours=cfg.http["live_max_age_hours"]))
            except BlockedError:
                raise
            except FetchError as exc:
                log.warning("Sofascore sem épocas de %s: %s", lg.key, exc)
                continue
            for season in seasons:
                season_id = ids.get(short_season_label(season))
                if season_id is None:
                    log.warning("Sofascore sem a época %s de %s", short_season_label(season), lg.key)
                    continue
                live = cfg.http["live_max_age_hours"] if season >= cfg.season else None
                offset = 0
                while True:
                    body = f.get_json(league_stats_url(lg.sofascore, season_id, offset), max_age_hours=live)
                    for r in parse_league_stats(body):
                        rows.append({"league": lg.key, "season": season, "sofascore_season_id": season_id, **r})
                    if body.get("page", 1) >= body.get("pages", 0):
                        break
                    offset += 100
    finally:
        if own:
            f.close()
    return pd.DataFrame(rows)


def match_league_players(players: pd.DataFrame, league_stats: pd.DataFrame, manual: dict[tuple[int, int], int] | None = None) -> pd.DataFrame:
    """Liga cada compra vinda das 5 ligas ao jogador do Sofascore na época anterior.

    ``players`` precisa de: player_id, player_name, season (verão), origin_league_top5,
    other_club_name. Por ordem, o primeiro critério com um só jogador:

    1. mesmo nome: no clube de origem, na liga (mudou de clube em janeiro), nas 5
       ligas (emprestado a outro clube das 5 ligas);
    2. mesmas palavras noutra ordem ('Min-jae Kim' / 'Kim Min-jae'), nos mesmos sítios;
    3. último nome igual no clube de origem;
    4. um nome contém todas as palavras do outro ('Abner' / 'Abner Vinícius'), no
       clube de origem e depois na liga.

    Alcunhas conhecidas (``NAME_ALIASES``) trocam-se antes pelo nome do Sofascore.
    Um critério com vários jogadores para a procura (``ambiguous``). O resto fica
    ``not_found``/``ambiguous``, com os candidatos, para ligação manual;
    ``manual`` = {(player_id, verão): sofascore_id}.
    """
    manual = manual or {}
    pools = {key: g for key, g in league_stats.groupby(["league", "season"])} if not league_stats.empty else {}
    seasons = {key: g for key, g in league_stats.groupby("season")} if not league_stats.empty else {}
    out = []
    for p in players.itertuples(index=False):
        rec = {"player_id": p.player_id, "summer": p.season, "player_name": p.player_name, "sofascore_id": None, "match_status": "not_found"}
        pool = pools.get((p.origin_league_top5, p.season - 1))
        if (p.player_id, p.season) in manual:
            rec.update(sofascore_id=manual[(p.player_id, p.season)], match_status="manual")
        elif pool is not None:
            names = pool["sofascore_name"].fillna("").map(normalize_name)
            club = normalize_name(p.other_club_name or "")
            team = _closest_team(club, pool["team_name"])  # 'Paris FC' não é o 'Paris Saint-Germain'
            in_club = pool["team_name"] == team
            name = normalize_name(p.player_name)
            name = NAME_ALIASES.get(name, name)
            last = name.split()[-1] if name else None
            words = frozenset(name.split())
            everywhere = seasons.get(p.season - 1, pool)
            all_names = everywhere["sofascore_name"].fillna("").map(normalize_name)
            same_words, all_same_words = names.map(lambda n: frozenset(n.split()) == words), all_names.map(lambda n: frozenset(n.split()) == words)
            nested = names.map(lambda n: bool(n) and bool(words) and (words <= set(n.split()) or set(n.split()) <= words))
            tiers = (
                pool[in_club & (names == name)], pool[names == name], everywhere[all_names == name],
                pool[in_club & same_words], pool[same_words], everywhere[all_same_words],
                pool[in_club & (names.str.split().str[-1] == last)],
                pool[in_club & nested], pool[nested],
            )
            for cand in tiers:
                ids = cand["sofascore_id"].unique()
                if len(ids) == 1:
                    rec.update(sofascore_id=int(ids[0]), sofascore_name=cand["sofascore_name"].iloc[0], match_status="matched")
                    break
                if len(ids) > 1:
                    rec["match_status"] = "ambiguous"
                    break
            if rec["match_status"] != "matched":
                # Mesmo último nome na liga primeiro, depois o plantel do clube de origem.
                near = pd.concat([pool[names.str.split().str[-1] == last], pool[in_club]])
                near = near.drop_duplicates("sofascore_id")
                rec["candidates"] = describe_candidates(
                    [{"name": r.sofascore_name, "sofascore_id": r.sofascore_id, "team": r.team_name} for r in near.itertuples()]
                )
        out.append(rec)
    return pd.DataFrame(out)


# Alcunhas do Transfermarkt -> nome no Sofascore (normalizados), quando não há palavras em comum.
NAME_ALIASES = {
    "chicharito": "javier hernandez",
}

_CLUB_NOISE = {"fc", "cf", "ac", "as", "ssc", "afc", "sc", "sv", "vfb", "vfl", "tsg", "rc", "ogc", "de", "club", "calcio", "1", "04", "05", "1899", "1846", "1907", "1909"}


def _closest_team(club: str, teams: pd.Series) -> str | None:
    """Equipa do Sofascore com mais palavras em comum com o nome do Transfermarkt."""
    wa = set(club.split()) - _CLUB_NOISE
    best, score = None, 0.0
    for t in teams.dropna().unique():
        wb = set(normalize_name(t).split()) - _CLUB_NOISE
        j = len(wa & wb) / len(wa | wb) if wa | wb else 0.0
        if j > score:
            best, score = t, j
    return best

