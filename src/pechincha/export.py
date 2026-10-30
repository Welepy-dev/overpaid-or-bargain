"""Dados da Fase 5 para o artigo do portfólio (``uv run main.py export``).

Só lê ``outputs/performance`` (a tabela, ``history.csv`` e ``season_progress.csv``
que o ``performance`` grava) e escreve três JSON pequenos, só com dados, em
``<portfolio>/public/data/overpaid-or-bargain/``. Os gráficos são desenhados no
portfólio, com as cores do site; aqui só se decide o que entra:

- ``meta.json``: data da fotografia, jornadas por liga e contagens por estado.
- ``ranking.json``: todas as compras no ranking (índice custo/desempenho e intervalo de 80%).
- ``evolution.json``: o índice das 15 compras mais caras no ranking, uma entrada por fotografia semanal.

A execução semanal (``scripts/weekly.sh``) corre-o depois do ``performance`` e
faz commit e push do portfólio, que volta a publicar o site.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config
from .eda.analysis import KEY, LEAGUE_NAMES
from .performance.analysis import DEFAULT_MIN_MINUTES_RANKING, EVOLUTION_PLAYERS, SEASON

log = logging.getLogger(__name__)

PORTFOLIO_DATA = Path("public") / "data" / "overpaid-or-bargain"


def _clean(value):
    """Valor pronto para JSON: vazios do pandas passam a ``None`` e os tipos do numpy a nativos."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(_clean(data), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


def ranking(table: pd.DataFrame) -> list[dict]:
    """Uma entrada por compra no ranking (com índice), do melhor índice para o pior. Valores em milhões de euros."""
    v = table[table["value_index"].notna()].sort_values("value_index", ascending=False)
    return [{
        "player": r.player_name, "club": r.buying_club, "league": r.buying_league, "position": r.position_group,
        "fee": round(r.fee / 1e6, 2), "fee_pct": r.fee_pct_position, "score": r.score_shrunk,
        "index": r.value_index, "low": r.value_index_low, "high": r.value_index_high,
        "minutes": int(r.minutes_after), "confidence": r.confidence, "verdict": r.verdict,
    } for r in v.itertuples()]


def evolution(history: pd.DataFrame, table: pd.DataFrame, n: int = EVOLUTION_PLAYERS) -> list[dict]:
    """O índice das ``n`` compras mais caras no ranking da última fotografia, em cada fotografia (como o gráfico 07)."""
    h = history.merge(table[KEY + ["fee"]], on=KEY, how="left")
    h = h[h["value_index"].notna()]
    if h.empty:
        return []
    h["snapshot_date"] = h["snapshot_date"].astype(str)
    latest = h[h["snapshot_date"] == h["snapshot_date"].max()]
    keys = latest.nlargest(n, "fee")[KEY]
    h = h.merge(keys, on=KEY).sort_values("snapshot_date")
    out = []
    for _, g in h.groupby(KEY, sort=False):
        r = g.iloc[-1]
        out.append({
            "player": r["player_name"], "club": r["buying_club"], "position": r["position_group"], "fee": round(r["fee"] / 1e6, 2),
            "points": [{"date": p.snapshot_date, "index": p.value_index, "low": p.value_index_low, "high": p.value_index_high,
                        "minutes": int(p.minutes_after), "confidence": p.confidence} for p in g.itertuples()],
        })
    return sorted(out, key=lambda p: -p["fee"])


def meta(table: pd.DataFrame, history: pd.DataFrame, season: pd.DataFrame, min_minutes_ranking: int) -> dict:
    """Data da fotografia, jornadas por liga e contagens, para o artigo dizer de quando são os números."""
    return {
        "season": f"{SEASON}/{(SEASON + 1) % 100:02d}",
        "snapshot_date": str(history["snapshot_date"].astype(str).max()),
        "snapshots": int(history["snapshot_date"].nunique()),
        "exported": date.today().isoformat(),
        "purchases": len(table),
        "ranked": int(table["value_index"].notna().sum()),
        "min_minutes_ranking": int(min_minutes_ranking),
        "status": {k: int(v) for k, v in table["status"].value_counts().items()},
        "leagues": [{"league": LEAGUE_NAMES.get(r.lg, r.lg), "min_matches": int(r.min_team_matches), "max_matches": int(r.max_team_matches),
                     "last_result": str(r.last_result)} for r in season.itertuples()],
    }


def run(cfg: Config, portfolio: Path | None = None) -> Path:
    """Escreve os três JSON em ``<portfolio>/public/data/overpaid-or-bargain`` e devolve essa pasta."""
    portfolio = Path(portfolio) if portfolio else cfg.root.parent / "portfolio"
    if not portfolio.is_dir():
        raise SystemExit(f"Portfólio não encontrado em {portfolio}: clonar ao lado deste repositório ou passar --portfolio")
    src = cfg.root / "outputs" / "performance"
    missing = [f for f in ["performance_2026.csv", "history.csv", "season_progress.csv"] if not (src / f).exists()]
    if missing:
        raise SystemExit(f"Faltam {', '.join(missing)} em {src}: correr primeiro `uv run main.py performance`")
    table = pd.read_csv(src / "performance_2026.csv")
    history = pd.read_csv(src / "history.csv")
    season = pd.read_csv(src / "season_progress.csv")

    out = portfolio / PORTFOLIO_DATA
    out.mkdir(parents=True, exist_ok=True)
    rows = ranking(table)
    tracks = evolution(history, table)
    _write(out / "meta.json", meta(table, history, season, cfg.phase5.get("min_minutes_ranking", DEFAULT_MIN_MINUTES_RANKING)))
    _write(out / "ranking.json", rows)
    _write(out / "evolution.json", tracks)
    log.info("Exportação: %d compras no ranking, %d na evolução, em %s", len(rows), len(tracks), out)
    return out
