"""Fase 3: análise exploratória da tabela por compra.

Só lê ``data/processed/purchases.parquet`` (do ``build``): não faz pedidos à
rede. ``uv run main.py eda`` grava:

- ``data/processed/percentiles.parquet``: percentis por posição das métricas
  por 90 (e rácios) da época anterior, uma linha por compra, para a Fase 4.
- ``outputs/eda/*.html`` e ``*.json``: gráficos Plotly no formato da Fase 6.
- ``outputs/eda/*.csv``: as tabelas por trás dos gráficos.
- ``outputs/eda/summary.md``: os números principais, sempre com a amostra.

Lê também ``data/interim/hicp_monthly.parquet`` (passo ``hicp`` do ``collect``),
se existir, para o gasto por verão em euros de 2026 (gráfico 03b).

Percentis: cada compra é comparada com as compras do modelo de preço da mesma
posição (ATT, MID, DEF), de todos os verões juntos, com pelo menos
``REFERENCE_MIN_MINUTES`` minutos de liga na época anterior. Não há posição
para todos os jogadores das ligas no Sofascore, por isso a referência são os
jogadores comprados, não a liga inteira. Compras com menos minutos também
recebem percentil, marcadas com ``low_minutes`` (o por 90 é instável).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import Config
from ..ingest.hicp import summer_deflator
from ..pipeline import load, save
from ..report.theme import ACCENT, DIVERGING, MUTED, POSITION_COLORS, POSITION_LABELS, TEXT_SECONDARY, export, log_axis, style

log = logging.getLogger(__name__)

REFERENCE_MIN_MINUTES = 900
MINUTE_THRESHOLDS = [1, 450, 900, 1350, 1800]
POSITIONS = ["ATT", "MID", "DEF"]
LEAGUE_NAMES = {"ENG": "Premier League", "ESP": "LaLiga", "ITA": "Serie A", "GER": "Bundesliga", "FRA": "Ligue 1"}

# Métricas com percentil: todas as por 90 e os rácios. ``ballRecovery`` fica de
# fora (falta em 60% das compras do modelo).
RATIOS = ["ss_pass_accuracy", "ss_dribble_success", "ss_duels_won_pct", "ss_rating"]
SKIP = {"ss_ballRecovery_p90"}
# Métricas em que menos é melhor: o percentil é invertido (100 = melhor).
LOWER_IS_BETTER = {"ss_possessionLost_p90", "ss_dribbledPast_p90", "ss_fouls_p90"}

METRIC_LABELS = {
    "us_goals_p90": "Goals", "us_np_goals_p90": "NP goals", "us_xg_p90": "xG", "us_np_xg_p90": "NP xG",
    "us_assists_p90": "Assists", "us_xa_p90": "xA", "us_shots_p90": "Shots", "us_key_passes_p90": "Key passes",
    "us_xg_chain_p90": "xGChain", "us_xg_buildup_p90": "xGBuildup", "ss_bigChancesCreated_p90": "Big chances created",
    "ss_keyPasses_p90": "Key passes (Sofascore)", "ss_accuratePasses_p90": "Accurate passes",
    "ss_accurateFinalThirdPasses_p90": "Final-third passes", "ss_accurateLongBalls_p90": "Accurate long balls",
    "ss_accurateCrosses_p90": "Accurate crosses", "ss_successfulDribbles_p90": "Dribbles", "ss_touches_p90": "Touches",
    "ss_possessionLost_p90": "Possession lost (inv.)", "ss_tackles_p90": "Tackles", "ss_tacklesWon_p90": "Tackles won",
    "ss_interceptions_p90": "Interceptions", "ss_clearances_p90": "Clearances", "ss_blockedShots_p90": "Blocked shots",
    "ss_possessionWonAttThird_p90": "Possession won (att. third)", "ss_dribbledPast_p90": "Dribbled past (inv.)",
    "ss_totalDuelsWon_p90": "Duels won", "ss_aerialDuelsWon_p90": "Aerials won", "ss_fouls_p90": "Fouls (inv.)",
    "ss_wasFouled_p90": "Fouled", "ss_pass_accuracy": "Pass accuracy", "ss_dribble_success": "Dribble success",
    "ss_duels_won_pct": "Duels won %", "ss_rating": "Sofascore rating",
}

# Perfil (radar) por posição: as métricas que descrevem o papel.
PROFILE_METRICS = {
    "ATT": ["us_np_xg_p90", "us_shots_p90", "us_xa_p90", "us_key_passes_p90", "ss_bigChancesCreated_p90",
            "ss_successfulDribbles_p90", "us_xg_chain_p90", "ss_possessionWonAttThird_p90"],
    "MID": ["us_xa_p90", "us_key_passes_p90", "us_xg_chain_p90", "us_xg_buildup_p90", "ss_accuratePasses_p90",
            "ss_accurateFinalThirdPasses_p90", "ss_successfulDribbles_p90", "ss_tackles_p90", "ss_interceptions_p90"],
    "DEF": ["ss_tackles_p90", "ss_interceptions_p90", "ss_clearances_p90", "ss_aerialDuelsWon_p90", "ss_duels_won_pct",
            "ss_accuratePasses_p90", "ss_accurateLongBalls_p90", "us_xg_buildup_p90", "ss_dribbledPast_p90"],
}

# Features candidatas da Fase 4 (além dos percentis) para as correlações.
CONTEXT_FEATURES = {
    "log_mv_adjusted": "Market value before (log, adj.)", "age_at_transfer": "Age", "minutes_before": "Minutes before",
    "origin_league_position": "Origin league position", "origin_ppg": "Origin points per game",
    "origin_uefa_5y": "Origin league UEFA 5y coef.", "height_cm": "Height",
}
HEATMAP_METRICS = [
    "us_np_xg_p90", "us_shots_p90", "us_xa_p90", "us_key_passes_p90", "us_xg_chain_p90", "us_xg_buildup_p90",
    "ss_keyPasses_p90", "ss_bigChancesCreated_p90", "ss_accuratePasses_p90", "ss_accurateFinalThirdPasses_p90",
    "ss_touches_p90", "ss_successfulDribbles_p90", "ss_tackles_p90", "ss_interceptions_p90", "ss_clearances_p90",
    "ss_aerialDuelsWon_p90", "ss_rating",
]
REDUNDANT_RHO = 0.8


def run(cfg: Config) -> dict:
    """Corre a análise exploratória e grava tabelas e gráficos. Devolve o resumo."""
    df = prepare(load(cfg.root / "data" / "processed" / "purchases.parquet"))
    out = cfg.root / "outputs" / "eda"
    out.mkdir(parents=True, exist_ok=True)

    metrics = percentile_metrics(df)
    pct = percentiles(df, metrics)
    save(pct, cfg.root / "data" / "processed" / "percentiles.parquet")
    df = df.merge(pct.drop(columns=["player_name", "position_group", "minutes_before"]), on=KEY, how="left")

    summary: dict = {"n": len(df), "n_model": int(df["in_price_model"].sum()), "n_2026": int((df["summer"] == SEASON).sum())}
    hicp_path = cfg.interim_dir / "hicp_monthly.parquet"
    hicp = load(hicp_path) if hicp_path.exists() else None
    if hicp is None:
        log.warning("Sem %s: corre `collect --steps hicp` para o gráfico 03b (gasto em euros de 2026)", hicp_path)
    summary["fees"] = fee_charts(df, out, hicp)
    summary["ages"] = age_charts(df, out)
    summary["minutes"] = minutes_charts(df, out)
    summary["drivers"] = driver_charts(df, out)
    summary["spend_2026"] = spend_2026_charts(df, out)
    summary["profiles"] = profile_charts(df, out)
    summary["features"] = feature_charts(df, metrics, out)
    summary["reference_n"] = pct.groupby("position_group")["reference_n"].first().to_dict()
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    log.info("Fase 3: gráficos e tabelas em %s", out)
    return summary


KEY = ["summer", "player_id", "club_id"]
SEASON = 2026


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas derivadas usadas em vários gráficos."""
    df = df.copy()
    df["fee_m"] = df["fee"] / 1e6
    df["fee_adjusted_m"] = df["fee_adjusted"] / 1e6
    df["log_fee_adjusted"] = np.log(df["fee_adjusted"].where(df["fee_adjusted"] > 0))
    df["mv_adjusted"] = df["market_value_before"] / df["fee_index"]
    df["log_mv_adjusted"] = np.log(df["mv_adjusted"].where(df["mv_adjusted"] > 0))
    transfer = pd.to_datetime(df["transfer_date"], errors="coerce")
    expires = pd.to_datetime(df["contract_expires"], errors="coerce")
    df["contract_years"] = (expires - transfer).dt.days / 365.25
    return df


# --------------------------------------------------------------------------- percentis


def percentile_metrics(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_p90") and c not in SKIP] + RATIOS


def percentiles(df: pd.DataFrame, metrics: list[str], min_minutes: int = REFERENCE_MIN_MINUTES) -> pd.DataFrame:
    """Percentil (0–100, 100 = melhor) de cada compra contra a referência da sua posição.

    Referência: compras do modelo de preço da mesma posição, todos os verões,
    com ``min_minutes`` ou mais na época anterior. Empates contam metade
    (percentil "médio"). Sem minutos na época anterior, o percentil fica vazio.
    """
    minutes = df["minutes_before"].fillna(0)
    reference = df["in_price_model"].astype(bool) & (minutes >= min_minutes)
    out = df[KEY + ["player_name", "position_group", "minutes_before"]].copy()
    out["low_minutes"] = minutes < min_minutes
    out["reference_n"] = df["position_group"].map(df[reference].groupby("position_group").size()).fillna(0).astype(int)
    for m in metrics:
        col = pd.Series(np.nan, index=df.index)
        for pos, g in df.groupby("position_group"):
            ref = np.sort(df.loc[reference & (df["position_group"] == pos), m].dropna().to_numpy())
            values = g[m].where(minutes.loc[g.index] > 0)
            if not len(ref):
                continue
            ok = values.notna()
            v = values[ok].to_numpy()
            p = (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / 2 / len(ref) * 100
            col.loc[values[ok].index] = 100 - p if m in LOWER_IS_BETTER else p
        out[f"pct_{m}"] = col.round(1)
    return out


# --------------------------------------------------------------------------- preços


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """ρ de Spearman sem o scipy (ordens + Pearson), só nas linhas com os dois valores."""
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank())


def fee_charts(df: pd.DataFrame, out: Path, hicp: pd.DataFrame | None = None) -> dict:
    known = df[df["fee_known"] & (df["fee"] > 0)]
    log_fee = np.log10(known["fee"])
    fig = go.Figure(go.Histogram(x=log_fee, xbins=dict(size=0.1), marker=dict(color=ACCENT, line=dict(width=1, color="white")),
                                 hovertemplate="%{y} purchases<extra></extra>"))
    ticks = [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8]
    fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[_money(t) for t in ticks], title="Fixed fee (log scale)")
    fig.update_yaxes(title="Purchases")
    style(fig, "Transfer fees are heavily right-skewed", f"Top-5 to top-5 purchases with a known fee, summers 2019–2026 (n={len(known):,}). "
          f"Median {_money(known['fee'].median())}, mean {_money(known['fee'].mean())}.")
    export(fig, out, "01_fee_distribution")

    by = known.groupby("summer").agg(n=("fee", "size"), median_fee_m=("fee_m", "median"), mean_fee_m=("fee_m", "mean"),
                                     p90_fee_m=("fee_m", lambda s: s.quantile(0.9)), total_fee_m=("fee_m", "sum"),
                                     fee_index=("fee_index", "first")).reset_index()
    if hicp is not None and len(hicp):
        by["hicp_deflator"] = by["summer"].map(summer_deflator(hicp, by["summer"].tolist()))
        by["total_fee_real_m"] = by["total_fee_m"] * by["hicp_deflator"]
    by.round(2).to_csv(out / "fees_by_summer.csv", index=False)

    fig = go.Figure()
    for summer, g in known.groupby("summer"):
        fig.add_trace(go.Box(y=g["fee_m"], name=f"{summer}<br>n={len(g)}", marker=dict(color=ACCENT, size=4), line=dict(width=1.5),
                             boxpoints="outliers", text=g["player_name"], hovertemplate="%{text}: €%{y:.1f}m<extra></extra>", showlegend=False))
    fig.update_yaxes(**log_axis("Fixed fee (€m, log scale)"))
    style(fig, "Fees per summer", f"Median fee rose from {_money(by['median_fee_m'].iloc[0] * 1e6)} in {by['summer'].iloc[0]} "
          f"to {_money(by['median_fee_m'].iloc[-1] * 1e6)} in {by['summer'].iloc[-1]}; known fees only.")
    export(fig, out, "02_fee_by_summer")

    fig = go.Figure(go.Bar(x=by["summer"].astype(str), y=by["total_fee_m"], marker=dict(color=ACCENT, cornerradius=4),
                           customdata=by["n"], hovertemplate="%{x}: €%{y:,.0f}m over %{customdata} purchases<extra></extra>",
                           text=[f"€{v / 1000:.2f}bn" for v in by["total_fee_m"]], textposition="outside", textfont=dict(color=TEXT_SECONDARY)))
    fig.update_yaxes(title="Total fixed fees (€m)")
    style(fig, "Spending on top-5 to top-5 purchases per summer", "Sum of known fixed fees; loans, free transfers and goalkeepers excluded.")
    export(fig, out, "03_spend_by_summer")

    real = None
    if "total_fee_real_m" in by and by["total_fee_real_m"].notna().any():
        real = by.dropna(subset=["total_fee_real_m"])
        base = int(by["summer"].max())
        x = real["summer"].astype(str)
        fig = go.Figure(go.Bar(x=x, y=real["total_fee_real_m"], name=f"In {base} euros", marker=dict(color=ACCENT, cornerradius=4),
                               customdata=np.c_[real["n"], real["total_fee_m"]],
                               hovertemplate=f"%{{x}}: €%{{y:,.0f}}m in {base} euros (€%{{customdata[1]:,.0f}}m at the time) "
                                             "over %{customdata[0]} purchases<extra></extra>",
                               text=[f"€{v / 1000:.2f}bn" for v in real["total_fee_real_m"]], textposition="outside",
                               textfont=dict(color=TEXT_SECONDARY)))
        fig.add_trace(go.Scatter(x=x, y=real["total_fee_m"], name="Nominal (euros of each summer)", mode="markers",
                                 marker=dict(color=MUTED, size=9, symbol="line-ew", line=dict(width=3, color=MUTED)),
                                 hovertemplate="%{x}: €%{y:,.0f}m nominal<extra></extra>"))
        fig.update_yaxes(title=f"Total fixed fees (€m, {base} prices)")
        fig.update_layout(legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
        style(fig, f"Spending on top-5 to top-5 purchases per summer, in {base} euros",
              "Sum of known fixed fees deflated by euro-area HICP (ECB, June–August average of each summer); "
              "grey marks are the nominal totals. Loans, free transfers and goalkeepers excluded.")
        export(fig, out, "03b_spend_by_summer_real")
    first, last = by.iloc[0], by.iloc[-1]
    return {"n_known": len(known), "median": known["fee"].median(), "mean": known["fee"].mean(), "skew_log": float(log_fee.skew()),
            "skew": float(known["fee"].skew()), "median_first": (int(first["summer"]), first["median_fee_m"]),
            "median_last": (int(last["summer"]), last["median_fee_m"]), "total_last": last["total_fee_m"], "by_summer": by,
            "real": real}


def age_charts(df: pd.DataFrame, out: Path) -> dict:
    """Idade dos jogadores comprados em cada verão (todas as compras, com ou sem valor conhecido)."""
    a = df.dropna(subset=["age_at_transfer"])
    by = a.groupby("summer")["age_at_transfer"].agg(n="size", median_age="median", mean_age="mean",
                                                      under_23=lambda s: (s < 23).mean(), over_28=lambda s: (s >= 28).mean()).reset_index()
    by.round(3).to_csv(out / "ages_by_summer.csv", index=False)

    fig = go.Figure()
    for summer, g in a.groupby("summer"):
        fig.add_trace(go.Box(y=g["age_at_transfer"], name=f"{summer}<br>n={len(g)}", marker=dict(color=ACCENT, size=4), line=dict(width=1.5),
                             boxpoints="outliers", text=g["player_name"], hovertemplate="%{text}: %{y:.1f} years<extra></extra>", showlegend=False))
    fig.update_yaxes(title="Age at transfer (years)")
    first, last = by.iloc[0], by.iloc[-1]
    style(fig, "Age of top-5 to top-5 purchases per summer",
          f"All purchases (n={len(a):,}), goalkeepers and loans excluded. Median age {first['median_age']:.1f} in {int(first['summer'])} "
          f"and {last['median_age']:.1f} in {int(last['summer'])}; {last['under_23']:.0%} under 23 in {int(last['summer'])}.")
    export(fig, out, "06b_age_by_summer")
    return {"n": len(a), "median": float(a["age_at_transfer"].median()), "by_summer": by}


def _money(v: float) -> str:
    if v >= 1e9:
        return f"€{v / 1e9:.2f}bn"
    if v >= 1e6:
        return f"€{v / 1e6:.0f}m" if v >= 1e7 else f"€{v / 1e6:.1f}m".replace(".0m", "m")
    return f"€{v / 1e3:.0f}k"


# --------------------------------------------------------------------------- minutos


def minutes_thresholds(df: pd.DataFrame, thresholds: list[int] = MINUTE_THRESHOLDS) -> pd.DataFrame:
    """Tamanho do modelo de preço e do ranking de 2026 com cada mínimo de minutos."""
    eligible = df[df["fee_known"].astype(bool)]
    rows = []
    for t in thresholds:
        keep = eligible[eligible["minutes_before"].fillna(0) >= t]
        hist = keep[keep["summer"] < SEASON]
        row = {"min_minutes": t, "price_model": len(keep), "train_2019_2025": len(hist), "ranking_2026": int((keep["summer"] == SEASON).sum())}
        for pos in POSITIONS:
            row[f"ranking_2026_{pos}"] = int(((keep["summer"] == SEASON) & (keep["position_group"] == pos)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def minutes_charts(df: pd.DataFrame, out: Path) -> dict:
    table = minutes_thresholds(df)
    table.to_csv(out / "minutes_thresholds.csv", index=False)
    played = df[df["fee_known"] & (df["minutes_before"].fillna(0) > 0)]
    fig = go.Figure(go.Histogram(x=played["minutes_before"], xbins=dict(start=0, size=180), marker=dict(color=ACCENT, line=dict(width=1, color="white")),
                                 hovertemplate="%{x} min: %{y} purchases<extra></extra>"))
    under = int((played["minutes_before"] < REFERENCE_MIN_MINUTES).sum())
    fig.add_vline(x=REFERENCE_MIN_MINUTES, line=dict(color=MUTED, dash="dash", width=1.5),
                  annotation_text=f"{REFERENCE_MIN_MINUTES} min: {under} purchases below", annotation_position="top right")
    fig.update_xaxes(title="League minutes in the season before")
    fig.update_yaxes(title="Purchases")
    style(fig, "How much the bought players played the season before",
          f"Known fee and at least one league minute (n={len(played):,}). Purchases under {REFERENCE_MIN_MINUTES} minutes stay out of the price model.")
    export(fig, out, "04_minutes_before")
    return {"table": table, "under": under, "n_played": len(played)}


# --------------------------------------------------------------------------- o que explica o preço


def driver_charts(df: pd.DataFrame, out: Path) -> dict:
    m = df[df["in_price_model"].astype(bool)]
    res: dict = {"n_model": len(m)}

    # Preço vs. valor de mercado (log-log).
    both = m.dropna(subset=["fee_adjusted", "mv_adjusted"])
    both = both[both["mv_adjusted"] > 0]
    fig = go.Figure()
    for pos in POSITIONS:
        g = both[both["position_group"] == pos]
        fig.add_trace(go.Scattergl(x=g["mv_adjusted"] / 1e6, y=g["fee_adjusted_m"], mode="markers", name=POSITION_LABELS[pos],
                                   marker=dict(color=POSITION_COLORS[pos], size=7, opacity=0.65, line=dict(width=1, color="white")),
                                   text=g["player_name"] + " (" + g["summer"].astype(str) + ", " + g["club_name"] + ")",
                                   hovertemplate="%{text}<br>Fee €%{y:.1f}m · market value €%{x:.1f}m<extra></extra>"))
    lo, hi = both[["mv_adjusted", "fee_adjusted"]].min().min() / 1e6, both[["mv_adjusted", "fee_adjusted"]].max().max() / 1e6
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dot", width=1.5), name="Fee = market value", hoverinfo="skip"))
    fig.update_xaxes(**log_axis("Transfermarkt value before the transfer (€m, 2026 prices)"))
    fig.update_yaxes(**log_axis("Fixed fee (€m, 2026 prices)"))
    rho = _spearman(both["fee_adjusted"], both["mv_adjusted"])
    premium = (both["fee_adjusted"] / both["mv_adjusted"]).median()
    style(fig, "Fees follow market value closely", f"Price-model purchases with a market value (n={len(both):,}). Spearman ρ = {rho:.2f}; "
          f"median fee is {premium:.2f}× the market value.")
    export(fig, out, "05_fee_vs_market_value")
    res["mv"] = {"n": len(both), "rho": rho, "premium": premium, "missing": int(m["mv_adjusted"].isna().sum())}

    # Preço vs. idade, com a mediana por idade.
    a = m.dropna(subset=["age_at_transfer", "fee_adjusted"])
    fig = go.Figure()
    for pos in POSITIONS:
        g = a[a["position_group"] == pos]
        fig.add_trace(go.Scattergl(x=g["age_at_transfer"], y=g["fee_adjusted_m"], mode="markers", name=POSITION_LABELS[pos],
                                   marker=dict(color=POSITION_COLORS[pos], size=7, opacity=0.55, line=dict(width=1, color="white")),
                                   text=g["player_name"] + " (" + g["summer"].astype(str) + ")",
                                   hovertemplate="%{text}<br>Age %{x:.1f} · €%{y:.1f}m<extra></extra>"))
    ages = a.assign(age=a["age_at_transfer"].round().clip(18, 33)).groupby("age")["fee_adjusted_m"].agg(["median", "size"]).reset_index()
    fig.add_trace(go.Scatter(x=ages["age"], y=ages["median"], mode="lines+markers", name="Median by age", line=dict(color="#0b0b0b", width=2),
                             marker=dict(size=8), customdata=ages["size"], hovertemplate="Age %{x}: median €%{y:.1f}m (n=%{customdata})<extra></extra>"))
    fig.update_xaxes(title="Age at transfer (≤18 and ≥33 pooled in the median)")
    fig.update_yaxes(**log_axis("Fixed fee (€m, 2026 prices)"))
    peak = ages.loc[ages["median"].idxmax()]
    style(fig, "Fee by age", f"Price-model purchases (n={len(a):,}). The median peaks at {int(peak['age'])} (€{peak['median']:.1f}m, n={int(peak['size'])}).")
    export(fig, out, "06_fee_vs_age")
    ages.to_csv(out / "fee_by_age.csv", index=False)
    res["age"] = {"peak": int(peak["age"]), "peak_median": peak["median"], "rho": _spearman(a["fee_adjusted"], a["age_at_transfer"])}

    # Preço por liga de origem e posição na tabela.
    fig = go.Figure()
    leagues = m.groupby("origin_league_top5")["fee_adjusted_m"].median().sort_values(ascending=False).index
    for lg in leagues:
        g = m[m["origin_league_top5"] == lg]
        fig.add_trace(go.Box(y=g["fee_adjusted_m"], name=f"{LEAGUE_NAMES[lg]}<br>n={len(g)}", marker=dict(color=ACCENT, size=4), line=dict(width=1.5),
                             text=g["player_name"], hovertemplate="%{text}: €%{y:.1f}m<extra></extra>", showlegend=False))
    fig.update_yaxes(**log_axis("Fixed fee (€m, 2026 prices)"))
    style(fig, "Fee by origin league", "Price-model purchases, all summers; ordered by median fee.")
    export(fig, out, "07_fee_by_origin_league")
    by_league = m.groupby("origin_league_top5")["fee_adjusted_m"].agg(["size", "median"]).sort_values("median", ascending=False)
    res["origin_league"] = by_league

    bins = [0, 4, 8, 12, 16, 20]
    labels = ["1st–4th", "5th–8th", "9th–12th", "13th–16th", "17th–20th"]
    p = m.dropna(subset=["origin_league_position"]).copy()
    p["band"] = pd.cut(p["origin_league_position"], bins, labels=labels)
    fig = go.Figure()
    for band in labels:
        g = p[p["band"] == band]
        fig.add_trace(go.Box(y=g["fee_adjusted_m"], name=f"{band}<br>n={len(g)}", marker=dict(color=ACCENT, size=4), line=dict(width=1.5),
                             text=g["player_name"] + " (" + g["other_club_name"] + ")", hovertemplate="%{text}: €%{y:.1f}m<extra></extra>", showlegend=False))
    fig.update_xaxes(title="Origin club's league position the season before")
    fig.update_yaxes(**log_axis("Fixed fee (€m, 2026 prices)"))
    rho_pos = _spearman(p["fee_adjusted"], p["origin_league_position"])
    style(fig, "Players from stronger clubs cost more", f"Price-model purchases with a known origin position (n={len(p):,}). Spearman ρ = {rho_pos:.2f}.")
    export(fig, out, "08_fee_by_origin_position")
    res["origin_position"] = {"rho": rho_pos, "medians": p.groupby("band", observed=True)["fee_adjusted_m"].median()}

    # Contrato: a data de fim é a de hoje, não a do momento da transferência.
    c = df[(df["summer"] == SEASON) & df["fee_known"] & (df["fee"] > 0)].dropna(subset=["contract_years"])
    fig = go.Figure()
    for pos in POSITIONS:
        g = c[c["position_group"] == pos]
        fig.add_trace(go.Scatter(x=g["contract_years"], y=g["fee_m"], mode="markers", name=POSITION_LABELS[pos],
                                 marker=dict(color=POSITION_COLORS[pos], size=9, opacity=0.7, line=dict(width=1, color="white")),
                                 text=g["player_name"] + " (" + g["club_name"] + ")", hovertemplate="%{text}<br>%{x:.1f} years · €%{y:.1f}m<extra></extra>"))
    fig.update_xaxes(title="Length of the new contract (years)")
    fig.update_yaxes(**log_axis("Fixed fee (€m)"))
    style(fig, "Summer 2026: fee vs. length of the new contract", f"Summer 2026 purchases (n={len(c)}). Transfermarkt only keeps the current contract, "
          "so years left at the selling club are not available.")
    export(fig, out, "09_fee_vs_contract_2026")
    res["contract"] = {"n": len(c), "rho": _spearman(c["fee"], c["contract_years"])}
    return res


# --------------------------------------------------------------------------- verão de 2026


def spend_2026_charts(df: pd.DataFrame, out: Path) -> dict:
    s = df[(df["summer"] == SEASON) & df["fee_known"]].copy()
    by_league = s.groupby("league").agg(n=("fee", "size"), total_m=("fee_m", "sum"), median_m=("fee_m", "median")).sort_values("total_m")
    fig = go.Figure(go.Bar(y=[LEAGUE_NAMES[lg] for lg in by_league.index], x=by_league["total_m"], orientation="h", marker=dict(color=ACCENT, cornerradius=4),
                           customdata=np.stack([by_league["n"], by_league["median_m"]], axis=1),
                           text=[f"€{v:,.0f}m · {n} buys" for v, n in zip(by_league["total_m"], by_league["n"])], textposition="outside",
                           textfont=dict(color=TEXT_SECONDARY), hovertemplate="%{y}: €%{x:,.0f}m, %{customdata[0]} purchases, median €%{customdata[1]:.1f}m<extra></extra>"))
    fig.update_xaxes(title="Fixed fees spent (€m)")
    fig.update_yaxes(automargin=True)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(margin=dict(r=140))
    total = s["fee_m"].sum()
    share = by_league["total_m"].get("ENG", 0) / total
    style(fig, "Summer 2026 spending by buying league", f"{len(s)} top-5 to top-5 purchases, €{total / 1000:.2f}bn in fixed fees; the Premier League spent {share:.0%}.")
    export(fig, out, "10_spend_2026_by_league")
    by_league.round(2).to_csv(out / "spend_2026_by_league.csv")

    by_club = s.groupby(["club_name", "league"]).agg(n=("fee", "size"), total_m=("fee_m", "sum")).reset_index().sort_values("total_m", ascending=False)
    by_club.round(2).to_csv(out / "spend_2026_by_club.csv", index=False)
    top = by_club.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(y=top["club_name"], x=top["total_m"], orientation="h", marker=dict(color=ACCENT, cornerradius=4),
                           text=[f"€{v:,.0f}m · {n}" for v, n in zip(top["total_m"], top["n"])], textposition="outside", textfont=dict(color=TEXT_SECONDARY),
                           customdata=top["n"], hovertemplate="%{y}: €%{x:,.0f}m over %{customdata} purchases<extra></extra>"))
    fig.update_xaxes(title="Fixed fees spent (€m)")
    style(fig, "Summer 2026: the 15 biggest spenders", "Purchases from other top-5 clubs only; label shows total and number of purchases.")
    fig.update_yaxes(automargin=True)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(height=560, margin=dict(r=100))
    export(fig, out, "11_spend_2026_by_club")

    cols = ["player_name", "position", "age_at_transfer", "other_club_name", "club_name", "fee_m", "market_value_before", "minutes_before"]
    ranked = s.sort_values("fee", ascending=False)
    ranked[cols].assign(market_value_before_m=lambda x: x["market_value_before"] / 1e6).drop(columns="market_value_before").round(2) \
        .to_csv(out / "purchases_2026_by_fee.csv", index=False)
    t = ranked.head(20).iloc[::-1]
    names = t["player_name"] + " (" + t["club_name"] + ")"
    fig = go.Figure()
    for _, r in t.iterrows():
        if pd.notna(r["market_value_before"]):
            fig.add_trace(go.Scatter(x=[r["market_value_before"] / 1e6, r["fee_m"]], y=[f"{r['player_name']} ({r['club_name']})"] * 2, mode="lines",
                                     line=dict(color="#c9c8c3", width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t["market_value_before"] / 1e6, y=names, mode="markers", name="Market value before", marker=dict(color=MUTED, size=10),
                             hovertemplate="%{y}<br>Market value €%{x:.0f}m<extra></extra>"))
    fig.add_trace(go.Scatter(x=t["fee_m"], y=names, mode="markers", name="Fixed fee", marker=dict(color=ACCENT, size=11, line=dict(width=2, color="white")),
                             customdata=t["other_club_name"], hovertemplate="%{y}<br>Fee €%{x:.1f}m from %{customdata}<extra></extra>"))
    fig.update_xaxes(title="€m")
    no_mv = t.loc[t["market_value_before"].isna(), "player_name"].iloc[::-1].tolist()
    note = f" No market value for {', '.join(no_mv)}." if no_mv else ""
    style(fig, "Summer 2026: the 20 most expensive purchases", f"Fixed fee against the Transfermarkt value just before the move.{note}")
    fig.update_layout(height=640, margin=dict(l=280))
    export(fig, out, "12_top_purchases_2026")

    cheapest = s[s["fee"] > 0].sort_values("fee").iloc[0]
    priciest = ranked.iloc[0]
    mv = s.dropna(subset=["market_value_before"])
    return {"n": len(s), "total_m": total, "by_league": by_league, "top_clubs": by_club.head(5),
            "priciest": (priciest["player_name"], priciest["club_name"], priciest["fee_m"]),
            "cheapest": (cheapest["player_name"], cheapest["club_name"], cheapest["fee_m"]),
            "above_mv": int((mv["fee"] > mv["market_value_before"]).sum()), "n_mv": len(mv), "missing_mv": int(s["market_value_before"].isna().sum())}


# --------------------------------------------------------------------------- perfis


def profile_charts(df: pd.DataFrame, out: Path, n_players: int = 3) -> dict:
    """Radar dos percentis das compras mais caras de 2026, por posição (3 jogadores por gráfico)."""
    s = df[(df["summer"] == SEASON) & df["in_ranking"].astype(bool)]
    res = {}
    for i, pos in enumerate(POSITIONS):
        metrics = PROFILE_METRICS[pos]
        theta = [METRIC_LABELS[m] for m in metrics] + [METRIC_LABELS[metrics[0]]]
        top = s[s["position_group"] == pos].sort_values("fee", ascending=False).head(n_players)
        fig = go.Figure()
        for j, (_, r) in enumerate(top.iterrows()):
            r_vals = [r[f"pct_{m}"] for m in metrics]
            color = list(POSITION_COLORS.values())[j]
            fig.add_trace(go.Scatterpolar(r=r_vals + r_vals[:1], theta=theta, name=f"{r['player_name']} (€{r['fee_m']:.0f}m, {int(r['minutes_before'])} min)",
                                          line=dict(color=color, width=2), fill="toself", fillcolor=_alpha(color, 0.12), marker=dict(size=6),
                                          hovertemplate="%{theta}: %{r:.0f}th percentile<extra>" + r["player_name"] + "</extra>"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], gridcolor="#e6e5e1"),
                                     angularaxis=dict(gridcolor="#e6e5e1"), bgcolor="#fcfcfb"), height=560)
        ref_n = int(df.loc[df["position_group"] == pos, "reference_n"].iloc[0])
        style(fig, f"Summer 2026's most expensive {POSITION_LABELS[pos].lower()}: profile the season before",
              f"Percentile against {ref_n} bought {POSITION_LABELS[pos].lower()} with ≥{REFERENCE_MIN_MINUTES} league minutes (2019–2026). 100 = best.")
        export(fig, out, f"13_profile_2026_{pos}")
        res[pos] = top[["player_name", "fee_m", "minutes_before"]]
    return res


def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# --------------------------------------------------------------------------- features da Fase 4


def redundant_pairs(df: pd.DataFrame, metrics: list[str], threshold: float = REDUNDANT_RHO) -> pd.DataFrame:
    """Pares de métricas com |ρ de Spearman| >= ``threshold`` em alguma posição."""
    rows = []
    for pos, g in df.groupby("position_group"):
        corr = g[metrics].corr(method="spearman")
        for i, a in enumerate(metrics):
            for b in metrics[i + 1:]:
                rows.append({"metric_a": a, "metric_b": b, "position_group": pos, "rho": corr.loc[a, b]})
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["metric_a", "metric_b"], columns="position_group", values="rho").reset_index()
    wide["max_abs_rho"] = wide[POSITIONS].abs().max(axis=1)
    return wide[wide["max_abs_rho"] >= threshold].sort_values("max_abs_rho", ascending=False).round(3)


def feature_charts(df: pd.DataFrame, metrics: list[str], out: Path) -> dict:
    m = df[df["in_price_model"].astype(bool) & (df["minutes_before"].fillna(0) >= REFERENCE_MIN_MINUTES)]
    target = "log_fee_adjusted"

    # Correlação de cada feature com o preço, dentro de cada posição (percentis) e no total (contexto).
    rows = []
    for f, label in CONTEXT_FEATURES.items():
        rows.append({"feature": f, "label": label, "kind": "context", "rho": _spearman(m[target], m[f]), "n": int(m[[target, f]].dropna().shape[0])})
    for metric in metrics:
        f = f"pct_{metric}"
        r = {"feature": f, "label": METRIC_LABELS[metric], "kind": "percentile", "rho": _spearman(m[target], m[f]), "n": int(m[[target, f]].dropna().shape[0])}
        for pos, g in m.groupby("position_group"):
            r[f"rho_{pos}"] = _spearman(g[target], g[f])
        rows.append(r)
    corr = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
    corr.round(3).to_csv(out / "feature_correlations.csv", index=False)

    top = corr.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(y=top["label"], x=top["rho"], orientation="h", marker=dict(color=np.where(top["rho"] >= 0, DIVERGING[-1][1], DIVERGING[0][1]), cornerradius=4),
                           customdata=top["n"], hovertemplate="%{y}: ρ = %{x:.2f} (n=%{customdata})<extra></extra>"))
    fig.update_xaxes(title="Spearman ρ with log(fee, 2026 prices)", range=[-1, 1], zeroline=True, zerolinecolor=MUTED)
    style(fig, "What moves with the fee: the 20 strongest features", f"Price-model purchases with ≥{REFERENCE_MIN_MINUTES} minutes before (n={len(m):,}). "
          "Red = higher with higher fees, blue = lower (as in the heatmap).")
    fig.update_layout(height=620, margin=dict(l=240))
    export(fig, out, "14_feature_correlations")

    # Redundância entre métricas (valores por 90, dentro de cada posição).
    pairs = redundant_pairs(m, metrics)
    pairs.to_csv(out / "redundant_metric_pairs.csv", index=False)

    cols = ["log_fee_adjusted", "log_mv_adjusted", "age_at_transfer", "minutes_before", "origin_league_position", "origin_uefa_5y"] + HEATMAP_METRICS
    labels = ["Fee (log)"] + [CONTEXT_FEATURES[c] for c in cols[1:6]] + [METRIC_LABELS[c] for c in HEATMAP_METRICS]
    mat = m[cols].corr(method="spearman")
    fig = go.Figure(go.Heatmap(z=mat.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale=DIVERGING, xgap=2, ygap=2,
                               hovertemplate="%{y} × %{x}: ρ = %{z:.2f}<extra></extra>", colorbar=dict(title="ρ", thickness=12)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=-45, showgrid=False)
    style(fig, "Correlations between candidate features", f"Spearman ρ, price-model purchases with ≥{REFERENCE_MIN_MINUTES} minutes (n={len(m):,}), all positions pooled.")
    fig.update_layout(height=820, width=900, margin=dict(l=220, b=200))
    export(fig, out, "15_feature_heatmap")
    return {"n": len(m), "corr": corr, "pairs": pairs}


# --------------------------------------------------------------------------- resumo


def summary_markdown(s: dict) -> str:
    f, mn, d, sp, ft = s["fees"], s["minutes"], s["drivers"], s["spend_2026"], s["features"]
    lines = [
        "# Phase 3: exploratory analysis (summary)",
        "",
        f"Generated by `uv run main.py eda` from `data/processed/purchases.parquet` ({s['n']:,} purchases; {s['n_model']:,} in the price model; "
        f"{s['n_2026']} in summer 2026). Charts are in this folder as `.html` (iframe fragment) and `.json` (Plotly).",
        "",
        "## Fees",
        f"- Known fees: n={f['n_known']:,}. Median {_money(f['median'])}, mean {_money(f['mean'])}; skewness {f['skew']:.1f} in euros and "
        f"{f['skew_log']:.2f} in log10, so the model works on log(fee).",
        f"- Median fee went from €{f['median_first'][1]:.1f}m ({f['median_first'][0]}) to €{f['median_last'][1]:.1f}m ({f['median_last'][0]}); "
        "`fee_adjusted` puts every summer in 2026 prices.",
    ]
    if f["real"] is not None:
        r = f["real"]
        peak = r.loc[r["total_fee_real_m"].idxmax()]
        lines.append(f"- Total spending in {int(r['summer'].max())} euros (euro-area HICP, chart 03b): "
                     + ", ".join(f"{int(x.summer)} €{x.total_fee_real_m / 1000:.2f}bn" for x in r.itertuples())
                     + f". Peak: {int(peak['summer'])}.")
    else:
        lines.append("- No HICP file yet (`uv run main.py collect --steps hicp`), so chart 03b was skipped.")
    lines += [
        "",
        "## Ages (chart 06b)",
        f"- Median age at transfer {s['ages']['median']:.1f} (n={s['ages']['n']:,}). By summer (median, share under 23): "
        + ", ".join(f"{int(r.summer)} {r.median_age:.1f} ({r.under_23:.0%})" for r in s["ages"]["by_summer"].itertuples()) + ".",
        "",
        "## Minutes the season before",
        f"The price model needs at least {REFERENCE_MIN_MINUTES} league minutes the season before (decision of 5 Oct 2026, `price_model.min_minutes_before`). "
        f"{mn['under']} of {mn['n_played']:,} purchases with a known fee and some minutes played fall under it; they stay in the data for phase 5.",
        "",
        "| Minimum minutes | Price model | Training 2019–2025 | Ranking 2026 | ATT | MID | DEF |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in mn["table"].itertuples():
        lines.append(f"| {r.min_minutes} | {r.price_model} | {r.train_2019_2025} | {r.ranking_2026} | {r.ranking_2026_ATT} | {r.ranking_2026_MID} | {r.ranking_2026_DEF} |")
    mv, age = d["mv"], d["age"]
    lines += [
        "",
        "## What goes with the fee (price model, 2026 prices)",
        f"- **Market value:** Spearman ρ = {mv['rho']:.2f} (n={mv['n']:,}); the median fee is {mv['premium']:.2f}× the Transfermarkt value. "
        f"{mv['missing']} model purchases have no market value before the transfer.",
        f"- **Age:** ρ = {age['rho']:.2f}; the median fee peaks at age {age['peak']} (€{age['peak_median']:.1f}m).",
        f"- **Origin club position:** ρ = {d['origin_position']['rho']:.2f} (a lower position number, a stronger club, goes with a higher fee). "
        "Medians: " + ", ".join(f"{k} €{v:.1f}m" for k, v in d["origin_position"]["medians"].items()) + ".",
        "- **Origin league** (median, n): " + ", ".join(f"{LEAGUE_NAMES[k]} €{r['median']:.1f}m ({int(r['size'])})" for k, r in d["origin_league"].iterrows()) + ".",
        f"- **Contract:** Transfermarkt keeps only the current contract, so years left at the selling club are unknown for every summer. "
        f"For 2026 the new contract length gives ρ = {d['contract']['rho']:.2f} with the fee (n={d['contract']['n']}); it is not a pre-transfer feature.",
        "",
        "## Summer 2026",
        f"- {sp['n']} purchases with a known fee, €{sp['total_m'] / 1000:.2f}bn in fixed fees.",
        "- By buying league: " + ", ".join(f"{LEAGUE_NAMES[k]} €{r['total_m']:,.0f}m ({int(r['n'])})" for k, r in sp["by_league"].iloc[::-1].iterrows()) + ".",
        "- Biggest spenders: " + ", ".join(f"{r.club_name} €{r.total_m:,.0f}m ({r.n})" for r in sp["top_clubs"].itertuples()) + ".",
        f"- Most expensive: {sp['priciest'][0]} to {sp['priciest'][1]}, €{sp['priciest'][2]:.1f}m. Cheapest paid fee: {sp['cheapest'][0]} to {sp['cheapest'][1]}, €{sp['cheapest'][2]:.2f}m.",
        f"- {sp['above_mv']} of {sp['n_mv']} purchases with a market value cost more than it ({sp['missing_mv']} have no market value).",
        "",
        "## Percentiles and profiles",
        "Each purchase is ranked against bought players of the same position (ATT/MID/DEF), all summers pooled, with at least "
        f"{REFERENCE_MIN_MINUTES} league minutes the season before: reference sizes "
        + ", ".join(f"{k} {v}" for k, v in sorted(s["reference_n"].items())) + ". "
        "It is not the whole league: Sofascore has no position for every player. Purchases under 900 minutes also get a percentile, flagged `low_minutes`. "
        "Possession lost, dribbled past and fouls are inverted (100 = best). Stored in `data/processed/percentiles.parquet`.",
        "",
    ]
    for pos, t in s["profiles"].items():
        lines.append(f"- Radar {pos}: " + ", ".join(f"{r.player_name} (€{r.fee_m:.0f}m, {int(r.minutes_before)} min)" for r in t.itertuples()) + ".")
    corr = ft["corr"]
    pct = corr[corr["kind"] == "percentile"].head(8)
    lines += [
        "",
        f"## Features for phase 4 (n={ft['n']:,}: price model, ≥{REFERENCE_MIN_MINUTES} minutes)",
        "Strongest percentile features (Spearman ρ with log fee, pooled; per position in `feature_correlations.csv`): "
        + ", ".join(f"{r.label} {r.rho:.2f}" for r in pct.itertuples()) + ".",
        "",
        f"{len(ft['pairs'])} pairs of metrics have |ρ| ≥ {REDUNDANT_RHO} in at least one position (`redundant_metric_pairs.csv`); keep one of each:",
    ]
    for r in ft["pairs"].head(15).itertuples():
        lines.append(f"- {METRIC_LABELS[r.metric_a]} ~ {METRIC_LABELS[r.metric_b]}: max |ρ| {r.max_abs_rho:.2f}")
    lines += [
        "",
        "## Limits",
        "- Correlations are bivariate and descriptive; market value already carries most of the information on performance.",
        "- Percentiles compare bought players only, pooled over 2019–2026 (per 90 drift between seasons is ignored).",
        "- Sofascore ball recoveries are missing for 60% of the model and are left out.",
    ]
    return "\n".join(lines) + "\n"
