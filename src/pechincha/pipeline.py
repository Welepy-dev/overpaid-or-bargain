"""Fase 1: orquestração da recolha.

Passos (cada um grava em ``data/interim/`` e pode correr sozinho):

- ``transfers``   Transfermarkt: movimentos das 5 ligas, clubes de cada época,
                  transferências elegíveis (compras, sem empréstimos nem GR).
- ``tm_details``  Histórico, valor de mercado antes da transferência e perfil dos
                  jogadores elegíveis: dataset público nos verões até
                  ``tm_dump.last_summer``, Transfermarkt para o resto.
- ``understat``   Estatísticas por época (verões anteriores) e jogo a jogo (época corrente).
- ``sofascore_leagues``  Sofascore: métricas defensivas e de posse de todos os
                  jogadores de campo das 5 ligas, por época (substitui o FBref).
- ``uefa``        Coeficientes UEFA por país (força das ligas de origem).
- ``sofascore``   Jogadores que chegam de fora das 5 ligas.

Opcional (fora da execução por omissão): ``clubelo``, força dos clubes a 1 de
julho; a API do ClubElo passou a exigir registo em setembro de 2026.

``current_only=True`` limita tudo à época corrente: é a execução semanal.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Config
from .http import FetchError
from .ingest import clubelo, sofascore, tm_dump, transfermarkt, uefa, understat
from .progress import progress

log = logging.getLogger(__name__)

STEPS = ["transfers", "tm_details", "understat", "sofascore_leagues", "uefa", "sofascore"]
OPTIONAL_STEPS = ["clubelo"]

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
    if "market_value_at_transfer" in df:
        # Sem histórico de valores (jogadores antigos pedidos só com o histórico de
        # transferências): usa-se o valor que o Transfermarkt mostra na transferência.
        if "market_value_before" not in df:
            df["market_value_before"] = None
            df["market_value_before_date"] = None
        df["market_value_before"] = pd.to_numeric(df["market_value_before"], errors="coerce").fillna(df["market_value_at_transfer"])
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
    steps = steps or STEPS
    for step in progress(steps, "Pipeline (passos)"):
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
    eligible = eligible[eligible["season"].isin(years)].reset_index(drop=True)
    parts = []  # (histórico, valores, perfis, origem)

    # Verões cobertos pelo dataset público: só os jogadores em falta vão ao Transfermarkt.
    last_dump = cfg.tm_dump.get("last_summer")
    in_dump_years = eligible["season"] <= last_dump if last_dump else pd.Series(False, index=eligible.index)
    from_dump = pd.Series(False, index=eligible.index)
    if in_dump_years.any():
        try:
            mask, h, v, p = tm_dump.details_for(cfg, eligible[in_dump_years])
            from_dump.loc[mask[mask].index] = True
            parts.append((h, v, p, "tm_dump"))
        except (FetchError, OSError, KeyError, ValueError) as exc:
            # Sem o dataset (download falhou, colunas mudaram), tudo vai ao Transfermarkt.
            log.warning("Dataset público indisponível (%s); a usar só o Transfermarkt", exc)

    f = transfermarkt.fetcher(cfg)
    try:
        gaps = eligible[in_dump_years & ~from_dump]
        if not gaps.empty:
            # Verões antigos: o histórico já traz o valor na data da transferência.
            h, v, p = transfermarkt.fetch_player_details(cfg, gaps["player_id"].tolist(), f, closed=True, with_values=False)
            parts.append((h, v, p, "transfermarkt"))
        recent = eligible[~in_dump_years]
        if not recent.empty:
            closed = all(cfg.summer_closed(s) for s in recent["season"].unique())
            h, v, p = transfermarkt.fetch_player_details(cfg, recent["player_id"].tolist(), f, closed=closed)
            parts.append((h, v, p, "transfermarkt"))
    finally:
        f.close()

    history = _concat([h.assign(details_source=src) for h, _, _, src in parts])
    values = _concat([v for _, v, _, _ in parts])
    profiles = _concat([p.assign(details_source=src) if not p.empty else p for _, _, p, src in parts])
    _merge_save(history, out / "tm_transfer_history.parquet", "player_id", None)
    _merge_save(values, out / "tm_market_values.parquet", "player_id", None)
    _merge_save(profiles, out / "tm_profiles.parquet", "player_id", None)
    enriched = enrich_with_details(eligible, history, values, profiles)
    enriched["details_source"] = from_dump.map({True: "tm_dump", False: "transfermarkt"})
    _merge_save(enriched, out / "transfers_enriched.parquet", "season", years)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [x for x in frames if not x.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _stat_seasons(cfg: Config, years: list[int]) -> list[int]:
    # Verão s usa a época s-1; a época corrente entra para a Fase 5.
    return sorted({s - 1 for s in years} | {cfg.season})


def _step_understat(cfg: Config, years: list[int]) -> None:
    out = cfg.interim_dir
    _merge_save(understat.fetch_player_seasons(cfg, _stat_seasons(cfg, years)), out / "understat_player_seasons.parquet", "season", None)
    for name, df in understat.fetch_current_season_matches(cfg).items():
        save(df, out / f"understat_{cfg.season}_{name}.parquet")


def _step_sofascore_leagues(cfg: Config, years: list[int]) -> None:
    out = cfg.interim_dir
    stats = sofascore.fetch_league_seasons(cfg, _stat_seasons(cfg, years))
    _merge_save(stats, out / "sofascore_league_player_seasons.parquet", "season", _stat_seasons(cfg, years))
    base = load(out / "transfers_eligible.parquet")
    todo = base[base["origin_top5"].astype(bool) & base["season"].isin(years)]
    cols = ["player_id", "player_name", "season", "origin_league_top5", "other_club_name"]
    matches = sofascore.match_league_players(todo[cols].drop_duplicates(), stats)
    if not matches.empty:
        log.info("Sofascore (ligas): %d de %d compras ligadas", int((matches["match_status"] == "matched").sum()), len(matches))
    _merge_save(matches.assign(source="league"), out / "sofascore_league_matches.parquet", "summer", years)


def _step_uefa(cfg: Config, years: list[int]) -> None:
    out = cfg.interim_dir
    coefs = uefa.fetch_coefficients(cfg, years)
    _merge_save(coefs, out / "uefa_country_coefficients.parquet", "season_label", None)
    _merge_save(uefa.five_year_ranking(coefs, years), out / "uefa_country_ranking.parquet", "summer", years)


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
    "sofascore_leagues": _step_sofascore_leagues,
    "uefa": _step_uefa,
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
