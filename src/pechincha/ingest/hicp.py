"""IPCH da zona euro (BCE): inflação geral, para pôr o gasto total em euros de 2026.

Série ``ICP.M.U2.N.000000.4.INX`` do BCE (índice mensal do IPCH, zona euro,
todos os itens). Só serve a EDA: o ``fee_index`` do ``build`` corrige pela
subida das transferências, que no gasto total apagava o próprio crescimento.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from ..config import Config
from ..http import CachedFetcher

log = logging.getLogger(__name__)

DEFAULT_URL = "https://data-api.ecb.europa.eu/service/data/ICP/M.U2.N.000000.4.INX?format=csvdata&startPeriod=2015-01"
SUMMER_MONTHS = (6, 7, 8)
COLUMNS = ["period", "year", "month", "hicp"]


def parse_csv(text: str) -> pd.DataFrame:
    """Linhas (period, year, month, hicp) do CSV do BCE (colunas TIME_PERIOD e OBS_VALUE)."""
    raw = pd.read_csv(io.StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    raw = raw.dropna(subset=["OBS_VALUE"])
    period = pd.to_datetime(raw["TIME_PERIOD"], format="%Y-%m")
    df = pd.DataFrame({"period": raw["TIME_PERIOD"], "year": period.dt.year, "month": period.dt.month,
                       "hicp": raw["OBS_VALUE"].astype(float)})
    return df.sort_values("period").reset_index(drop=True)[COLUMNS]


def fetch_monthly(cfg: Config, f: CachedFetcher | None = None) -> pd.DataFrame:
    """Série mensal completa (pedida de novo quando a cópia em cache tem mais de ``live_max_age_hours``)."""
    f = f or CachedFetcher(cfg.cache_dir, "hicp", min_interval=cfg.min_interval("hicp"),
                           timeout=cfg.http["timeout_seconds"], max_retries=cfg.http["max_retries"])
    url = cfg.hicp.get("url", DEFAULT_URL)
    return parse_csv(f.get_text(url, max_age_hours=cfg.http["live_max_age_hours"], suffix=".csv"))


def summer_deflator(monthly: pd.DataFrame, summers: list[int]) -> pd.Series:
    """Multiplicador por verão que põe euros desse verão em euros do último verão.

    O índice de cada verão é a média de junho a agosto. Um verão ainda sem
    esses meses publicados usa o último mês disponível.
    """
    level = {}
    latest = monthly.sort_values("period").iloc[-1]
    for s in summers:
        months = monthly[(monthly["year"] == s) & monthly["month"].isin(SUMMER_MONTHS)]
        if len(months):
            level[s] = months["hicp"].mean()
        elif s > latest["year"] or (s == latest["year"] and latest["month"] < SUMMER_MONTHS[0]):
            level[s] = latest["hicp"]
    level = pd.Series(level, dtype=float).reindex(summers)
    return level[max(summers)] / level
