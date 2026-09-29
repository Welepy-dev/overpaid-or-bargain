"""Linha de comandos: ``uv run main.py collect [--steps ...] [--current-only]``."""

from __future__ import annotations

import argparse
import logging

from .config import load_config
from .pipeline import STEPS, run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pechincha", description="Sobrepago ou pechincha? Pipeline de dados.")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Fase 1: recolha de dados")
    collect.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS, help="passos a correr (por omissão, todos)")
    collect.add_argument("--current-only", action="store_true", help="só a época corrente (execução semanal)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if args.command == "collect":
        failed = run(cfg, steps=args.steps, current_only=args.current_only)
        if failed:
            raise SystemExit(f"Passos com erro: {', '.join(failed)} (ver o log acima)")
