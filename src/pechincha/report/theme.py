"""Tema e exportação dos gráficos Plotly (formato da Fase 6).

Cada gráfico sai em dois ficheiros, para o website de portfolio:
``<nome>.html`` (fragmento para iframe, Plotly pelo CDN) e ``<nome>.json``
(para ``Plotly.newPlot``).
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# Paleta categórica (ordem fixa: azul, laranja, verde-água; nos gráficos de
# pontos usam-se no máximo as três primeiras).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
POSITION_COLORS = {"ATT": SERIES[0], "MID": SERIES[1], "DEF": SERIES[2]}
POSITION_LABELS = {"ATT": "Attackers", "MID": "Midfielders", "DEF": "Defenders"}
# Divergente azul <-> vermelho com cinzento no meio (correlações).
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ee8a89"], [1.0, "#a3201f"]]
ACCENT = SERIES[0]
MUTED = "#8a8984"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"

TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=TEXT),
        title=dict(font=dict(size=17), x=0, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY)),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY)),
        hoverlabel=dict(bgcolor="white", font=dict(color=TEXT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=80, b=60),
        bargap=0.25,
    )
)
pio.templates["pechincha"] = TEMPLATE


def style(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    """Aplica o tema, o título e um subtítulo (onde vai a amostra)."""
    text = f"{title}<br><sup style='color:{TEXT_SECONDARY}'>{subtitle}</sup>" if subtitle else title
    fig.update_layout(template="pechincha", title_text=text)
    return fig


def export(fig: go.Figure, out_dir: Path, name: str) -> None:
    """Grava ``name.html`` (fragmento, Plotly pelo CDN) e ``name.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_dir / f"{name}.html", include_plotlyjs="cdn", full_html=False)
    (out_dir / f"{name}.json").write_text(pio.to_json(fig), encoding="utf-8")
