"""Leitura do config.yaml na raiz do projeto."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class League:
    key: str
    tm_code: str
    tm_slug: str
    soccerdata: str
    sofascore: int | None = None


@dataclass(frozen=True)
class Config:
    season: int
    history_seasons: list[int]
    leagues: list[League]
    eligibility: dict
    clubelo_reference_month_day: str
    http: dict
    summer_window_close_month_day: str = "09-01"
    tm_dump: dict = field(default_factory=dict)
    uefa: dict = field(default_factory=dict)
    root: Path = ROOT

    @property
    def cache_dir(self) -> Path:
        return self.root / self.http["cache_dir"]

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim"

    def min_interval(self, source: str) -> float:
        """Intervalo entre pedidos de uma fonte (``http.min_interval_by_source`` ou o geral)."""
        return self.http.get("min_interval_by_source", {}).get(source, self.http["min_interval_seconds"])

    @property
    def all_seasons(self) -> list[int]:
        return sorted(set(self.history_seasons) | {self.season})

    def summer_closed(self, summer: int, today: date | None = None) -> bool:
        """A janela de verão ``summer`` já fechou (os dados dessas transferências não mudam)."""
        month, day = (int(x) for x in self.summer_window_close_month_day.split("-"))
        return (today or date.today()) > date(summer, month, day)


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
        summer_window_close_month_day=raw.get("summer_window_close_month_day", "09-01"),
        tm_dump=raw.get("tm_dump", {}),
        uefa=raw.get("uefa", {}),
        root=path.parent,
    )
