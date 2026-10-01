"""Dataset público transfermarkt-datasets (dcaribou, CC0): histórico sem raspar o Transfermarkt.

Três tabelas em CSV comprimido (``{base_url}/{tabela}.csv.gz``):

- ``transfers``: data, clubes, valor pago e valor de mercado na altura;
- ``player_valuations``: histórico de valores de mercado;
- ``players``: data de nascimento, altura, pé, posição, nacionalidade.

Os IDs são os do Transfermarkt, por isso cruzam diretamente com as páginas de
transferências por liga. As atualizações do dataset pararam em julho de 2026:
serve para os verões até ``tm_dump.last_summer``; o resto vem do Transfermarkt.

As funções ``to_*`` devolvem as mesmas colunas que os ``parse_*`` do módulo
``transfermarkt``, para o resto do pipeline não saber de onde vieram os dados.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from ..config import Config
from ..http import USER_AGENT, FetchError

log = logging.getLogger(__name__)

TABLES = ("transfers", "player_valuations", "players")


def download(cfg: Config, tables: tuple[str, ...] = TABLES) -> dict[str, Path]:
    """Descarrega as tabelas uma vez; o dataset está congelado, por isso a cópia não expira.

    Para voltar a descarregar (se o dataset recomeçar a ser atualizado), apagar
    ``data/raw/tm_dump/``.
    """
    base = cfg.tm_dump["base_url"].rstrip("/")
    folder = cfg.cache_dir / "tm_dump"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for table in tables:
        path = folder / f"{table}.csv.gz"
        if not path.exists():
            url = f"{base}/{table}.csv.gz"
            log.info("A descarregar %s", url)
            tmp = path.with_suffix(".part")
            try:
                with requests.get(url, stream=True, timeout=cfg.http["timeout_seconds"], headers={"User-Agent": USER_AGENT}) as resp:
                    if resp.status_code != 200:
                        raise FetchError(f"HTTP {resp.status_code} em {url}")
                    with tmp.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
            except requests.RequestException as exc:
                raise FetchError(f"{url}: {exc}") from exc
            tmp.replace(path)
        paths[table] = path
    return paths


def load(cfg: Config) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path, compression="gzip", low_memory=False) for name, path in download(cfg).items()}


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """Primeira coluna que existir (os nomes mudaram entre versões do dataset)."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def to_history(transfers: pd.DataFrame) -> pd.DataFrame:
    """Mesmas colunas que ``transfermarkt.parse_transfer_history``."""
    fee = pd.to_numeric(_col(transfers, "transfer_fee"), errors="coerce")
    return pd.DataFrame(
        {
            "player_id": transfers["player_id"].astype("int64"),
            "date": _dates(transfers["transfer_date"]),
            "season_label": _col(transfers, "transfer_season"),
            "from_club_id": pd.to_numeric(_col(transfers, "from_club_id"), errors="coerce").astype("Int64"),
            "from_club_name": _col(transfers, "from_club_name"),
            "to_club_id": pd.to_numeric(_col(transfers, "to_club_id"), errors="coerce").astype("Int64"),
            "to_club_name": _col(transfers, "to_club_name"),
            "market_value_at_transfer": pd.to_numeric(_col(transfers, "market_value_in_eur"), errors="coerce"),
            "upcoming": False,
            # O dataset não distingue empréstimos: o tipo vem sempre das páginas por liga.
            "transfer_type": None,
            "fee": fee,
            "fee_currency": fee.notna().map({True: "EUR", False: None}),
            "fee_raw": None,
        }
    )


def to_values(valuations: pd.DataFrame) -> pd.DataFrame:
    """Mesmas colunas que ``transfermarkt.parse_market_values``."""
    return pd.DataFrame(
        {
            "player_id": valuations["player_id"].astype("int64"),
            "date": _dates(valuations["date"]),
            "market_value": pd.to_numeric(_col(valuations, "market_value_in_eur"), errors="coerce"),
            "club_name": _col(valuations, "current_club_name"),
        }
    )


def to_profiles(players: pd.DataFrame) -> pd.DataFrame:
    """Mesmas colunas que ``transfermarkt.parse_profile``.

    ``contract_expires`` é o contrato atual (como no perfil do Transfermarkt), não o da altura.
    """
    return pd.DataFrame(
        {
            "player_id": players["player_id"].astype("int64"),
            "date_of_birth": _dates(_col(players, "date_of_birth")),
            "height_cm": pd.to_numeric(_col(players, "height_in_cm", "height_cm"), errors="coerce").astype("Int64"),
            "foot": _col(players, "foot"),
            "position_detail": _col(players, "sub_position", "position"),
            "citizenship": _col(players, "country_of_citizenship"),
            "contract_expires": _dates(_col(players, "contract_expiration_date")),
        }
    )


def covered(eligible: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """Linhas de ``eligible`` cuja transferência (jogador, clube de destino, ano) está no dataset."""
    h = history.dropna(subset=["date", "to_club_id"])
    keys = set(zip(h["player_id"].astype(int), h["to_club_id"].astype(int), h["date"].map(lambda d: d.year)))
    return pd.Series(
        [(int(p), int(c), int(s)) in keys for p, c, s in zip(eligible["player_id"], eligible["club_id"], eligible["season"])],
        index=eligible.index,
    )


def details_for(cfg: Config, eligible: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detalhes do dataset para ``eligible``.

    Devolve (máscara das linhas cobertas, histórico, valores, perfis), só dos jogadores cobertos.
    """
    raw = load(cfg)
    ids = set(eligible["player_id"].astype(int))
    history = to_history(raw["transfers"][raw["transfers"]["player_id"].isin(ids)])
    mask = covered(eligible, history)
    keep = set(eligible.loc[mask, "player_id"].astype(int))
    values = to_values(raw["player_valuations"][raw["player_valuations"]["player_id"].isin(keep)])
    profiles = to_profiles(raw["players"][raw["players"]["player_id"].isin(keep)])
    history = history[history["player_id"].isin(keep)].reset_index(drop=True)
    log.info("Dataset público: %d de %d transferências cobertas", int(mask.sum()), len(eligible))
    return mask, history, values.reset_index(drop=True), profiles.reset_index(drop=True)
