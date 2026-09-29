"""Leitura do config.yaml na raiz do projeto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class League:
    key: str
    tm_code: str
    tm_slug: str
    soccerdata: str


@dataclass(frozen=True)
class Config:
    season: int
    history_seasons: list[int]
    leagues: list[League]
    eligibility: dict
    clubelo_reference_month_day: str
    http: dict
    root: Path = ROOT

    @property
    def cache_dir(self) -> Path:
        return self.root / self.http["cache_dir"]

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def all_seasons(self) -> list[int]:
        return sorted(set(self.history_seasons) | {self.season})


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    leagues = [League(key=k, **v) for k, v in raw["leagues"].items()]
    return Config(
        season=raw["season"],
        history_seasons=raw["history_seasons"],
        leagues=leagues,
        eligibility=raw["eligibility"],
        clubelo_reference_month_day=raw["clubelo_reference_month_day"],
        http=raw["http"],
        root=path.parent,
    )
