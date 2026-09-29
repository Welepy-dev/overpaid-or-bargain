"""FBref (dados Opta) via soccerdata: métricas defensivas e de posse.

Desde 2025 o FBref deixou de publicar as tabelas avançadas (defense, passing,
possession). O soccerdata 1.9 só lê ``standard``, ``shooting``,
``playing_time`` e ``misc``; é aí que estão as métricas defensivas que restam
(interceções, desarmes ganhos, duelos aéreos, recuperações). Progressão e
restantes métricas defensivas vêm do Sofascore (ver ``sofascore.py``).

O FBref exige um browser (Chromium, via seleniumbase), encontrado por
``http.find_browser`` ou indicado com a variável ``PECHINCHA_BROWSER``. Se o Cloudflare
mostrar um captcha, correr com ``PECHINCHA_HEADLESS=0`` (abre uma janela).
"""

from __future__ import annotations

import logging

import pandas as pd
import soccerdata as sd
import soccerdata.fbref as sd_fbref

from ..config import Config
from ..http import find_browser, headless
from ._seasons import soccerdata_season

log = logging.getLogger(__name__)

STAT_TYPES = ["standard", "shooting", "playing_time", "misc"]

# O FBref passou a escrever "Bundesliga" (sem "Fußball-") na coluna Comp; sem
# isto o soccerdata 1.9 deixa a liga vazia nas linhas da Bundesliga.
sd_fbref.BIG_FIVE_DICT.setdefault("Bundesliga", "GER-Bundesliga")


def fetch_player_seasons(cfg: Config, seasons: list[int]) -> dict[str, pd.DataFrame]:
    reader = sd.FBref(
        leagues="Big 5 European Leagues Combined",
        seasons=[soccerdata_season(s) for s in seasons],
        data_dir=cfg.cache_dir / "fbref",
        path_to_browser=find_browser(),  # texto: o seleniumbase não aceita Path
        # O Cloudflare do FBref costuma bloquear o modo headless; com
        # PECHINCHA_HEADLESS=0 abre-se uma janela e o soccerdata resolve o captcha.
        headless=headless(),
    )
    out = {}
    for stat in STAT_TYPES:
        log.info("FBref %s", stat)
        df = reader.read_player_season_stats(stat_type=stat)
        df.columns = ["_".join(c for c in col if c).strip("_") if isinstance(col, tuple) else col for col in df.columns]
        out[stat] = df.reset_index()
    return out
