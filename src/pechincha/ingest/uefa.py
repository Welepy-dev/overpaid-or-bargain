"""Coeficientes UEFA por país (kassiesa.net): força das ligas de origem.

Substitui o ClubElo, cuja API passou a exigir registo em setembro de 2026.
Cada página ``crank{ano}`` tem os coeficientes das últimas 5 épocas de cada
país. Juntam-se as páginas numa tabela longa (país, época, coeficiente) e, para
cada verão ``s``, soma-se as 5 épocas até ``(s-1)/s``: é o ranking que estava
em vigor quando a transferência foi feita.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
from bs4 import BeautifulSoup

from ..config import Config
from ..http import CachedFetcher, FetchError
from ._seasons import short_season_label

log = logging.getLogger(__name__)

_SEASON = re.compile(r"^\d{2}/\d{2}$")


def parse_ranking(html: str) -> pd.DataFrame:
    """Linhas (country, season_label, coefficient) de uma página ``crank``."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for table in soup.find_all("table"):
        header = None
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if header is None:
                if any(c.lower() == "country" for c in cells) and any(_SEASON.match(c) for c in cells):
                    header = [c.lower() for c in cells]
                continue
            if len(cells) != len(header):
                cells = [c for c in cells if c]  # célula da bandeira, sem texto
            if len(cells) != len(header):
                continue
            country = cells[header.index("country")]
            if not country:
                continue
            for label, value in zip(header, cells):
                if _SEASON.match(label):
                    try:
                        coef = float(value)
                    except ValueError:
                        continue
                    rows.append({"country": country, "season_label": label, "coefficient": coef})
        if rows:
            break
    return pd.DataFrame(rows, columns=["country", "season_label", "coefficient"])


def fetch_coefficients(cfg: Config, summers: list[int], f: CachedFetcher | None = None) -> pd.DataFrame:
    """Coeficientes por época de cada país, das páginas precisas para ``summers``.

    Para cada época fica o valor da página mais recente (as páginas antigas podem
    ter sido guardadas a meio dessa época).
    """
    f = f or CachedFetcher(cfg.cache_dir, "uefa", min_interval=cfg.min_interval("uefa"),
                           timeout=cfg.http["timeout_seconds"], max_retries=cfg.http["max_retries"])
    years = sorted({y for s in summers for y in (s, s + 1)})
    frames = []
    for year in years:
        live = cfg.http["live_max_age_hours"] if year >= cfg.season else None
        try:
            df = parse_ranking(f.get_text(cfg.uefa["ranking_url"].format(year=year), max_age_hours=live))
        except FetchError as exc:
            log.warning("Ranking UEFA %s indisponível: %s", year, exc)
            continue
        frames.append(df.assign(page_year=year))
    if not frames:
        return pd.DataFrame(columns=["country", "season_label", "coefficient", "page_year"])
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("page_year").drop_duplicates(["country", "season_label"], keep="last").reset_index(drop=True)


def five_year_ranking(coefs: pd.DataFrame, summers: list[int]) -> pd.DataFrame:
    """Soma das 5 épocas até ``(s-1)/s`` para cada verão ``s`` (uma linha por país e verão)."""
    out = []
    for s in summers:
        labels = [short_season_label(y) for y in range(s - 5, s)]
        sub = coefs[coefs["season_label"].isin(labels)]
        g = sub.groupby("country").agg(uefa_5y=("coefficient", "sum"), seasons_found=("season_label", "nunique")).reset_index()
        g["summer"] = s
        g["uefa_rank"] = g["uefa_5y"].rank(ascending=False, method="min").astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
