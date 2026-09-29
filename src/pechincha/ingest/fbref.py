"""FBref (dados Opta) via soccerdata: métricas defensivas e de posse.

Desde 2025 o FBref deixou de publicar as tabelas avançadas (defense, passing,
possession). O soccerdata 1.9 só lê ``standard``, ``shooting``,
``playing_time`` e ``misc``; é aí que estão as métricas defensivas que restam
(interceções, desarmes ganhos, duelos aéreos, recuperações). Progressão e
restantes métricas defensivas vêm do Sofascore (ver ``sofascore.py``).

O FBref exige um browser (Chromium, via seleniumbase); o caminho pode ser
indicado com a variável de ambiente ``PECHINCHA_BROWSER``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import soccerdata as sd

from ..config import Config
from ._seasons import soccerdata_season

log = logging.getLogger(__name__)

STAT_TYPES = ["standard", "shooting", "playing_time", "misc"]


def fetch_player_seasons(cfg: Config, seasons: list[int]) -> dict[str, pd.DataFrame]:
    browser = os.environ.get("PECHINCHA_BROWSER")
    reader = sd.FBref(
        leagues="Big 5 European Leagues Combined",
        seasons=[soccerdata_season(s) for s in seasons],
        data_dir=cfg.cache_dir / "fbref",
        path_to_browser=Path(browser) if browser else None,
        headless=True,
    )
    out = {}
    for stat in STAT_TYPES:
        log.info("FBref %s", stat)
        df = reader.read_player_season_stats(stat_type=stat)
        df.columns = ["_".join(c for c in col if c).strip("_") if isinstance(col, tuple) else col for col in df.columns]
        out[stat] = df.reset_index()
    return out
