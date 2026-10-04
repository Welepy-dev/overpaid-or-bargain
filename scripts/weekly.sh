#!/usr/bin/env bash
# Execução semanal: só as estatísticas da época 2026/27 (Understat e Sofascore por liga).
# Transferências e detalhes do Transfermarkt ficam de fora: a janela fechou e
# repeti-los seriam ~450 pedidos ao Transfermarkt, com risco de bloqueio.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
log="logs/weekly_$(date +%Y%m%d_%H%M).log"
start=$(date +%s)
marker=$(mktemp)
trap 'rm -f "$marker"' EXIT
status=0
uv run main.py collect --current-only --steps understat sofascore_leagues >>"$log" 2>&1 || status=$?
# Pedidos feitos = páginas novas ou renovadas na cache.
{
  echo "---"
  echo "Duração: $(( $(date +%s) - start )) s; estado: $status"
  for d in data/raw/*/; do
    n=$(find "$d" -type f -newer "$marker" | wc -l)
    [ "$n" -gt 0 ] && echo "Pedidos $(basename "$d"): $n"
  done
} >>"$log"
exit "$status"
