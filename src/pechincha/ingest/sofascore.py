"""Sofascore: estatísticas por época, por liga inteira (5 ligas) e por jogador (resto).

5 grandes ligas (substitui o FBref, que perdeu os dados Opta em jan/2026):
``/unique-tournament/{liga}/season/{época}/statistics`` devolve todos os
jogadores de campo da liga, 100 por pedido, com as métricas de ``LEAGUE_FIELDS``
(desarmes, interceções, duelos, passes no último terço, dribles...). Cerca de
6 pedidos por liga e época.

Jogadores que chegam de fora das 5 ligas (só esses, não as ligas inteiras):

1. pesquisa pelo nome e confirmação pela data de nascimento (do Transfermarkt);
2. lista de competições/épocas do jogador;
3. estatísticas agregadas de cada competição nas épocas relevantes para o verão.
"""

from __future__ import annotations

import logging
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


def fetcher(cfg: Config) -> BrowserFetcher:
    # A API devolve 403 a clientes que não são browsers; os pedidos vão por um Chromium.
    # Abre-se o robots.txt (leve) só para ficar na origem www; a página inicial pode não acabar de carregar.
    h = cfg.http
    return BrowserFetcher(
        cfg.cache_dir, "sofascore", home="https://www.sofascore.com/robots.txt",
        min_interval=cfg.min_interval("sofascore"), timeout=h["timeout_seconds"], max_retries=h["max_retries"],
    )


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
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


def parse_player_seasons(data: dict) -> list[dict]:
    out = []
    for block in data.get("uniqueTournamentSeasons", []) or []:
        ut = block.get("uniqueTournament") or {}
        for s in block.get("seasons", []) or []:
            out.append(
                {
                    "tournament_id": ut.get("id"),
                    "tournament_name": ut.get("name"),
                    "season_id": s.get("id"),
                    "season_label": s.get("year"),
                }
            )
    return out


def fetch_players(cfg: Config, players: pd.DataFrame, f: CachedFetcher | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``players`` precisa de: player_id (Transfermarkt), player_name, date_of_birth, other_club_name, season.

    Devolve (correspondências, estatísticas). As correspondências registam
    também os jogadores não encontrados, para revisão manual na Fase 2.
    """
    own = f is None
    f = f or fetcher(cfg)
    try:
        return _fetch_players(cfg, players, f)
    finally:
        if own:
            f.close()


def _fetch_players(cfg: Config, players: pd.DataFrame, f: CachedFetcher) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches, stats = [], []
    for p in progress(players.itertuples(index=False), "Sofascore jogadores", total=len(players)):
        dob = p.date_of_birth if isinstance(p.date_of_birth, date) else None
        live = cfg.http["live_max_age_hours"] if p.season >= cfg.season else None
        rec = {"player_id": p.player_id, "summer": p.season, "player_name": p.player_name, "sofascore_id": None, "match_status": "not_found"}
        try:
            found = parse_search(f.get_json(f"{API}/search/all?q={quote(p.player_name)}&page=0"))
            if dob is not None:
                for c in likely_candidates(found, p.player_name):
                    if c["date_of_birth"] is None:
                        c["date_of_birth"] = parse_player_birth(f.get_json(f"{API}/player/{c['sofascore_id']}"))
            cand = pick_candidate(found, p.player_name, dob, p.other_club_name)
            if cand is None:
                rec["match_status"] = "ambiguous" if found else "not_found"
                continue
            rec.update(sofascore_id=cand["sofascore_id"], sofascore_name=cand["name"], match_status="matched")
            labels = relevant_season_labels(p.season)
            seasons = parse_player_seasons(f.get_json(f"{API}/player/{cand['sofascore_id']}/statistics/seasons", max_age_hours=live))
            for s in (s for s in seasons if s["season_label"] in labels):
                url = f"{API}/player/{cand['sofascore_id']}/unique-tournament/{s['tournament_id']}/season/{s['season_id']}/statistics/overall"
                try:
                    body = f.get_json(url, max_age_hours=live)
                except FetchError:
                    continue
                stats.append({"player_id": p.player_id, "sofascore_id": cand["sofascore_id"], "summer": p.season, **s, **parse_statistics(body)})
        except BlockedError:
            raise
        except FetchError as exc:
            log.warning("Sofascore falhou para %s: %s", p.player_name, exc)
            rec["match_status"] = "error"
        finally:
            matches.append(rec)
    return pd.DataFrame(matches), pd.DataFrame(stats)


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


def match_league_players(players: pd.DataFrame, league_stats: pd.DataFrame) -> pd.DataFrame:
    """Liga cada compra vinda das 5 ligas ao jogador do Sofascore na época anterior.

    ``players`` precisa de: player_id, player_name, season (verão), origin_league_top5,
    other_club_name. Por ordem: mesmo nome no clube de origem; mesmo nome, único na
    liga (jogadores que mudaram de clube em janeiro); último nome igual e único no
    clube de origem. O resto fica ``not_found``/``ambiguous`` para revisão na Fase 2.
    """
    pools = {key: g for key, g in league_stats.groupby(["league", "season"])} if not league_stats.empty else {}
    out = []
    for p in players.itertuples(index=False):
        rec = {"player_id": p.player_id, "summer": p.season, "player_name": p.player_name, "sofascore_id": None, "match_status": "not_found"}
        pool = pools.get((p.origin_league_top5, p.season - 1))
        if pool is not None:
            names = pool["sofascore_name"].fillna("").map(normalize_name)
            club = normalize_name(p.other_club_name or "")
            in_club = pool["team_name"].fillna("").map(lambda t: _same_club(club, normalize_name(t)))
            name = normalize_name(p.player_name)
            last = name.split()[-1] if name else None
            for cand in (pool[in_club & (names == name)], pool[names == name], pool[in_club & (names.str.split().str[-1] == last)]):
                ids = cand["sofascore_id"].unique()
                if len(ids) == 1:
                    rec.update(sofascore_id=int(ids[0]), sofascore_name=cand["sofascore_name"].iloc[0], match_status="matched")
                    break
                if len(ids) > 1:
                    rec["match_status"] = "ambiguous"
                    break
        out.append(rec)
    return pd.DataFrame(out)


_CLUB_NOISE = {"fc", "cf", "ac", "as", "ssc", "afc", "sc", "sv", "vfb", "vfl", "tsg", "rc", "ogc", "de", "club", "calcio", "1", "04", "05", "1899", "1846", "1907", "1909"}


def _same_club(a: str, b: str) -> bool:
    """Nomes de clube do Transfermarkt e do Sofascore ('Chelsea FC' / 'Chelsea')."""
    wa, wb = set(a.split()) - _CLUB_NOISE, set(b.split()) - _CLUB_NOISE
    return bool(wa and wb and (wa <= wb or wb <= wa or len(wa & wb) >= 2))
