"""Linha de comandos: ``uv run main.py collect [--steps ...] [--current-only] [--no-refresh]``, ``build``, ``eda``, ``model`` e ``performance``."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from .config import load_config
from .pipeline import OPTIONAL_STEPS, STEPS, run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pechincha", description="Sobrepago ou pechincha? Pipeline de dados.")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Fase 1: recolha de dados")
    collect.add_argument("--steps", nargs="+", choices=STEPS + OPTIONAL_STEPS, default=STEPS, help="passos a correr (por omissão, todos menos sofascore e clubelo)")
    collect.add_argument("--current-only", action="store_true", help="só a época corrente (execução semanal)")
    collect.add_argument("--no-refresh", action="store_true", help="não renovar a cache da época corrente: o que já está em cache lê-se de lá")
    sub.add_parser("build", help="Fase 2: tabela por compra (offline, só lê data/interim e a cache)")
    sub.add_parser("eda", help="Fase 3: análise exploratória (offline, só lê data/processed)")
    sub.add_parser("model", help="Fase 4: modelo de preço justo e ranking (offline, só lê data/processed)")
    sub.add_parser("performance", help="Fase 5: desempenho no clube novo em 2026/27 (offline, lê data/processed e data/interim)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if getattr(args, "no_refresh", False):
        cfg = replace(cfg, http={**cfg.http, "live_max_age_hours": None})
    if args.command == "collect":
        failed = run(cfg, steps=args.steps, current_only=args.current_only)
        if failed:
            raise SystemExit(f"Passos com erro: {', '.join(failed)} (ver o log acima)")
    elif args.command == "build":
        from . import build

        build.run(cfg)
    elif args.command == "eda":
        from . import eda

        eda.run(cfg)
    elif args.command == "model":
        from . import model

        model.run(cfg)
    elif args.command == "performance":
        from . import performance

        performance.run(cfg)
