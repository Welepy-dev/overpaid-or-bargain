"""Understat via soccerdata: estatísticas por época e, na época corrente, jogo a jogo.

Cobre só as 5 grandes ligas. As estatísticas "pré-transferência" de um verão
``s`` são as da época ``s-1`` (ex.: verão de 2026 -> época 2025/26).
"""

from __future__ import annotations

import logging

import pandas as pd
import soccerdata as sd

from ..config import Config
from ._seasons import soccerdata_season

log = logging.getLogger(__name__)


def _reader(cfg: Config, seasons: list[int]) -> sd.Understat:
    return sd.Understat(
        leagues=[lg.soccerdata for lg in cfg.leagues],
        seasons=[soccerdata_season(s) for s in seasons],
        data_dir=cfg.cache_dir / "understat",
    )


def fetch_player_seasons(cfg: Config, seasons: list[int]) -> pd.DataFrame:
    """xG, xA, xGChain, xGBuildup, remates, passes-chave, minutos por jogador e época."""
    return _reader(cfg, seasons).read_player_season_stats().reset_index()


def fetch_current_season_matches(cfg: Config) -> dict[str, pd.DataFrame]:
    """Época corrente jogo a jogo, para a Fase 5 (antes/depois e on/off)."""
    reader = _reader(cfg, [cfg.season])
    return {
        "schedule": reader.read_schedule().reset_index(),
        "player_matches": reader.read_player_match_stats().reset_index(),
        "shots": reader.read_shot_events().reset_index(),
    }
