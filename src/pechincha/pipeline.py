"""Fase 1: orquestração da recolha.

Passos (cada um grava em ``data/interim/`` e pode correr sozinho):

- ``transfers``   Transfermarkt: movimentos das 5 ligas, clubes de cada época,
                  transferências elegíveis (compras, sem empréstimos nem GR).
- ``tm_details``  Transfermarkt: histórico, valor de mercado antes da
                  transferência e perfil dos jogadores elegíveis.
- ``understat``   Estatísticas por época (verões anteriores) e jogo a jogo (época corrente).
- ``fbref``       Estatísticas por época das 5 ligas (standard, shooting, playing_time, misc).
- ``clubelo``     Força dos clubes a 1 de julho de cada verão.
- ``sofascore``   Jogadores que chegam de fora das 5 ligas.

``current_only=True`` limita tudo à época corrente: é a execução semanal.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Config
from .ingest import clubelo, fbref, sofascore, transfermarkt, understat

log = logging.getLogger(__name__)

STEPS = ["transfers", "tm_details", "understat", "fbref", "clubelo", "sofascore"]

GOALKEEPER = {"goalkeeper", "gk", "guarda-redes"}


def summers(cfg: Config, current_only: bool) -> list[int]:
    return [cfg.season] if current_only else cfg.all_seasons


def select_eligible(moves: pd.DataFrame, top5_clubs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compras das 5 ligas no verão, segundo as regras do config.

    Usa só as entradas ("in"), para cada transferência aparecer uma vez mesmo
    quando os dois clubes são das 5 ligas. ``origin_top5`` indica se o clube
    de origem estava nas 5 ligas na época anterior.
    """
    rules = cfg.eligibility
    df = moves[(moves["direction"] == "in") & (moves["window"] == "summer")].copy()
    excluded = set()
    if rules.get("exclude_loans", True):
        excluded.add("loan")
    if rules.get("exclude_returns", True):
        excluded.add("loan_return")
    if rules.get("exclude_free_transfers", True):
        excluded.add("free")
    df = df[~df["transfer_type"].isin(excluded)]
    if rules.get("exclude_goalkeepers", True):
        pos = df["position"].fillna("").str.lower()
        short = df["position_short"].fillna("").str.lower()
        df = df[~(pos.isin(GOALKEEPER) | short.isin(GOALKEEPER))]

    prev = top5_clubs.assign(season=top5_clubs["season"] + 1)[["season", "club_id", "league"]]
    prev = prev.rename(columns={"club_id": "other_club_id", "league": "origin_league_top5"})
    df = df.merge(prev, on=["season", "other_club_id"], how="left")
    df["origin_top5"] = df["origin_league_top5"].notna()
    # Transferências com valor desconhecido ficam, marcadas, para tentar outra fonte na Fase 2.
    df["fee_known"] = df["fee"].notna()
    return df.drop_duplicates(["season", "player_id", "club_id"]).reset_index(drop=True)


def enrich_with_details(eligible: pd.DataFrame, history: pd.DataFrame, values: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta data da transferência, valor de mercado anterior e dados do perfil."""
    df = eligible.copy()
    if not history.empty:
        h = history.dropna(subset=["date"]).copy()
        h["season"] = h["date"].map(lambda d: d.year)  # verão de 2026 -> transferências datadas de 2026
        h = h.rename(columns={"to_club_id": "club_id", "date": "transfer_date", "fee": "fee_history", "fee_raw": "fee_raw_history"})
        h = h.sort_values("transfer_date").drop_duplicates(["player_id", "club_id", "season"], keep="last")
        df = df.merge(
            h[["player_id", "club_id", "season", "transfer_date", "market_value_at_transfer", "fee_history", "fee_raw_history"]],
            on=["player_id", "club_id", "season"], how="left",
        )
    else:
        df["transfer_date"] = None
    if not values.empty:
        mv_before, mv_date = [], []
        grouped = {pid: g for pid, g in values.dropna(subset=["date"]).groupby("player_id")}
        for pid, when in zip(df["player_id"], df["transfer_date"]):
            v, d = transfermarkt.market_value_before(grouped.get(pid, values.iloc[0:0]), when if pd.notna(when) else None)
            mv_before.append(v)
            mv_date.append(d)
        df["market_value_before"] = mv_before
        df["market_value_before_date"] = mv_date
    if not profiles.empty:
        cols = [c for c in ["player_id", "date_of_birth", "height_cm", "foot", "position_detail", "citizenship", "contract_expires"] if c in profiles]
        df = df.merge(profiles[cols].drop_duplicates("player_id"), on="player_id", how="left")
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for c in df.columns:  # colunas mistas (ex.: datas e None) gravadas como texto
        if df[c].dtype == object:
            kinds = {type(v) for v in df[c].dropna()}
            if len(kinds) > 1:
                df[c] = df[c].astype(str)
    df.to_parquet(path, index=False)
    log.info("Gravado %s (%d linhas)", path.name, len(df))


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe: corre antes o passo que o cria.")
    return pd.read_parquet(path)


def run(cfg: Config, steps: list[str] | None = None, current_only: bool = False) -> list[str]:
    """Corre os passos pedidos. Um passo que falha não impede os seguintes.

    Devolve a lista de passos que falharam (vazia se tudo correu bem).
    """
    years = summers(cfg, current_only)
    failed = []
    for step in steps or STEPS:
        log.info("Passo %s (verões %s)", step, years)
        try:
            _STEP_FUNCS[step](cfg, years)
        except Exception:
            log.exception("Passo %s falhou", step)
            failed.append(step)
    return failed


def _step_transfers(cfg: Config, years: list[int]) -> None:
    out = cfg.interim_dir
    # A época anterior ao primeiro verão só serve para saber que clubes eram das 5 ligas.
    pages = sorted(set(years) | {min(years) - 1})
    moves = transfermarkt.fetch_league_transfers(cfg, pages)
    clubs = moves[["league", "season", "club_id", "club_name"]].drop_duplicates()
    eligible = select_eligible(moves[moves["season"].isin(years)], clubs, cfg)
    _merge_save(moves, out / "tm_moves.parquet", "season", pages)
    _merge_save(clubs, out / "tm_top5_clubs.parquet", "season", pages)
    _merge_save(eligible, out / "transfers_eligible.parquet", "season", years)


def _step_tm_details(cfg: Config, years: list[int]) -> None:
    out = cfg.interim_dir
    eligible = load(out / "transfers_eligible.parquet")
    eligible = eligible[eligible["season"].isin(years)]
    history, values, profiles = transfermarkt.fetch_player_details(cfg, eligible["player_id"].tolist())
    _merge_save(history, out / "tm_transfer_history.parquet", "player_id", None)
    _merge_save(values, out / "tm_market_values.parquet", "player_id", None)
    _merge_save(profiles, out / "tm_profiles.parquet", "player_id", None)
    enriched = enrich_with_details(eligible, history, values, profiles)
    _merge_save(enriched, out / "transfers_enriched.parquet", "season", years)


def _stat_seasons(cfg: Config, years: list[int]) -> list[int]:
    # Verão s usa a época s-1; a época corrente entra para a Fase 5.
    return sorted({s - 1 for s in years} | {cfg.season})


def _step_understat(cfg: Config, years: list[int]) -> None:
    out = cfg.interim_dir
    _merge_save(understat.fetch_player_seasons(cfg, _stat_seasons(cfg, years)), out / "understat_player_seasons.parquet", "season", None)
    for name, df in understat.fetch_current_season_matches(cfg).items():
        save(df, out / f"understat_{cfg.season}_{name}.parquet")


def _step_fbref(cfg: Config, years: list[int]) -> None:
    for stat, df in fbref.fetch_player_seasons(cfg, _stat_seasons(cfg, years)).items():
        _merge_save(df, cfg.interim_dir / f"fbref_player_seasons_{stat}.parquet", "season", None)


def _step_clubelo(cfg: Config, years: list[int]) -> None:
    _merge_save(clubelo.fetch_ratings(cfg, years), cfg.interim_dir / "clubelo_summers.parquet", "summer", years)


def _step_sofascore(cfg: Config, years: list[int]) -> None:
    out = cfg.interim_dir
    src = out / "transfers_enriched.parquet"
    base = load(src if src.exists() else out / "transfers_eligible.parquet")
    todo = base[(~base["origin_top5"].astype(bool)) & base["season"].isin(years)].copy()
    if "date_of_birth" not in todo:
        todo["date_of_birth"] = None
    todo["date_of_birth"] = pd.to_datetime(todo["date_of_birth"], errors="coerce").dt.date
    cols = ["player_id", "player_name", "date_of_birth", "other_club_name", "season"]
    matches, stats = sofascore.fetch_players(cfg, todo[cols].drop_duplicates())
    _merge_save(matches, out / "sofascore_matches.parquet", "summer", years)
    _merge_save(stats, out / "sofascore_player_seasons.parquet", "summer", years)


_STEP_FUNCS = {
    "transfers": _step_transfers,
    "tm_details": _step_tm_details,
    "understat": _step_understat,
    "fbref": _step_fbref,
    "clubelo": _step_clubelo,
    "sofascore": _step_sofascore,
}


def _merge_save(new: pd.DataFrame, path: Path, key: str, replaced: list[int] | None) -> None:
    """Substitui só as épocas recolhidas agora, mantendo as restantes do ficheiro.

    Assim a execução semanal (só época corrente) não apaga o histórico.
    """
    if path.exists() and key in new.columns:
        old = pd.read_parquet(path)
        keys = set(replaced) if replaced is not None else set(new[key].unique())
        if key in old.columns:
            old = old[~old[key].isin(keys)]
            new = pd.concat([old, new], ignore_index=True)
    save(new, path)
