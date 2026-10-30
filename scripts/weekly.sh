#!/usr/bin/env bash
# Execução semanal: só as estatísticas da época 2026/27 (Understat e Sofascore por liga),
# e depois a Fase 5 (desempenho no clube novo), offline, com o que ficou em cache.
# Transferências e detalhes do Transfermarkt ficam de fora: a janela fechou e
# repeti-los seriam ~450 pedidos ao Transfermarkt, com risco de bloqueio.
# No fim exporta os dados da Fase 5 para o portfólio (clone em ../portfolio, ou
# PORTFOLIO_DIR) e faz commit e push deles, para os gráficos do artigo se atualizarem.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
log="logs/weekly_$(date +%Y%m%d_%H%M).log"
start=$(date +%s)
marker=$(mktemp)
trap 'rm -f "$marker"' EXIT
status=0
uv run main.py collect --current-only --steps understat sofascore_leagues >>"$log" 2>&1 || status=$?
# Mesmo que um passo da recolha falhe, a Fase 5 corre com o que está em cache.
perf=0
uv run main.py performance >>"$log" 2>&1 || perf=$?
[ "$perf" -ne 0 ] && status=$perf

# Portfólio: só com a Fase 5 em dia, no branch main e sem conflitos com o remoto.
portfolio="${PORTFOLIO_DIR:-../portfolio}"
data="public/data/overpaid-or-bargain"
publish() {
  if [ ! -d "$portfolio/.git" ]; then
    echo "Portfólio: $portfolio não é um clone git; exportação ignorada"
    return 1
  fi
  local branch
  branch=$(git -C "$portfolio" rev-parse --abbrev-ref HEAD)
  if [ "$branch" != "main" ]; then
    echo "Portfólio: está no branch $branch, não em main; exportação ignorada"
    return 1
  fi
  git -C "$portfolio" pull --ff-only --quiet origin main || { echo "Portfólio: pull falhou"; return 1; }
  uv run main.py export --portfolio "$portfolio" || return 1
  git -C "$portfolio" add -- "$data"
  if git -C "$portfolio" diff --cached --quiet -- "$data"; then
    echo "Portfólio: dados iguais aos da semana passada, nada a publicar"
    return 0
  fi
  git -C "$portfolio" commit --quiet -m "Overpaid or bargain: weekly data $(date +%Y-%m-%d)" -- "$data" || return 1
  git -C "$portfolio" push --quiet origin main || { echo "Portfólio: push falhou (o commit fica local para a próxima semana)"; return 1; }
  echo "Portfólio: dados publicados"
}
if [ "$perf" -eq 0 ]; then
  publish >>"$log" 2>&1 || status=$?
else
  echo "Portfólio: a Fase 5 falhou, exportação ignorada" >>"$log"
fi
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
