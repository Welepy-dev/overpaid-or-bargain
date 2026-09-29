"""ClubElo via soccerdata: força de todos os clubes europeus numa data.

Usa-se 1 de julho de cada verão. A força de uma liga é a média do Elo dos seus
clubes (calculada na Fase 2), o que serve para ajustar métricas de ligas
fora das 5 grandes.
"""

from __future__ import annotations

import pandas as pd
import soccerdata as sd

from ..config import Config


def fetch_ratings(cfg: Config, seasons: list[int]) -> pd.DataFrame:
    reader = sd.ClubElo(data_dir=cfg.cache_dir / "clubelo")
    frames = []
    for s in seasons:
        when = f"{s}-{cfg.clubelo_reference_month_day}"
        df = reader.read_by_date(when).reset_index()
        df["summer"] = s
        df["reference_date"] = when
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
