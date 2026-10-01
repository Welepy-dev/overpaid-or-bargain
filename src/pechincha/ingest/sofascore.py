"""Sofascore: estatísticas por época dos jogadores que chegam de fora das 5 ligas.

Recolhem-se só as páginas desses jogadores (não as ligas inteiras):

1. pesquisa pelo nome e confirmação pela data de nascimento (do Transfermarkt);
2. lista de competições/épocas do jogador;
3. estatísticas agregadas de cada competição nas épocas relevantes para o verão.

Também serve de fonte de reserva para métricas de progressão e defesa que o
FBref deixou de publicar.
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
