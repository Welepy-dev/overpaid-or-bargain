"""Fase 4: preço justo das compras do verão de 2026.

Só lê ``data/processed`` (``purchases.parquet`` do ``build`` e
``percentiles.parquet`` do ``eda``): não faz pedidos à rede, por isso volta a
correr depois de cada atualização semanal. ``uv run main.py model`` grava:

- ``data/processed/fair_price.parquet``: uma linha por compra do modelo, com o
  preço previsto, o intervalo de 80% e (no verão corrente) a parte de cada
  grupo de features. Nos
  verões 2019–2025 a previsão é fora da amostra (o verão fica fora do treino).
- ``outputs/model/*.html`` e ``*.json``: gráficos Plotly no formato da Fase 6.
- ``outputs/model/*.csv`` e ``ranking_2026.json``: ranking e tabelas do modelo.
- ``outputs/model/summary.md``: os números principais, sempre com a amostra.

Modelo: ridge sobre log(preço fixo em preços de 2026), treinado nas compras de
2019–2025 do modelo de preço (≥900 minutos de liga na época anterior, preço
conhecido). O alpha escolhe-se deixando um verão de fora de cada vez. O
intervalo de 80% vem dos erros fora da amostra (quantis 10% e 90%): uma
compra é "sobrepaga" acima dele e "pechincha" abaixo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import Config
from ..eda.analysis import KEY, LEAGUE_NAMES, METRIC_LABELS, POSITIONS, _money, prepare
from ..pipeline import load, save
from ..report.theme import ACCENT, DIVERGING, MUTED, POSITION_COLORS, POSITION_LABELS, TEXT_SECONDARY, export, log_axis, style
from .core import Ridge, choose_alpha, out_of_fold, scores, summer_folds

log = logging.getLogger(__name__)

SEASON = 2026
ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
INTERVAL = (0.10, 0.90)
REFERENCE_AGE = 24  # centro do termo quadrático da idade

# Percentis (por posição, do eda) usados no modelo. Um de cada par redundante
# de ``outputs/eda/redundant_metric_pairs.csv`` (|ρ| ≥ 0.8): golos/xG sem
# penáltis, xA em vez de passes-chave e grandes ocasiões, desarmes em vez de
# desarmes ganhos, passes certos em vez de toques e passes no último terço,
# xGChain e xGBuildup ficam os dois (só se repetem nos defesas).
PERFORMANCE = [
    "ss_rating", "us_np_xg_p90", "us_np_goals_p90", "us_xa_p90", "us_xg_chain_p90", "us_xg_buildup_p90",
    "ss_accuratePasses_p90", "ss_accurateLongBalls_p90", "ss_successfulDribbles_p90", "ss_possessionLost_p90",
    "ss_possessionWonAttThird_p90", "ss_tackles_p90", "ss_interceptions_p90", "ss_clearances_p90",
    "ss_aerialDuelsWon_p90", "ss_duels_won_pct",
]

# Lista explícita de features, por grupo (o grupo dá a decomposição do preço).
FEATURES: dict[str, list[str]] = {
    "market_value": ["log_mv_adjusted"],
    "age": ["age_at_transfer", "age_sq"],
    "minutes": ["minutes_k"],
    "origin": ["origin_rel_position", "origin_ENG", "origin_ESP", "origin_ITA", "origin_GER"],
    "position": ["pos_ATT", "pos_MID"],
    "buyer_league": ["buyer_ENG", "buyer_ESP", "buyer_ITA", "buyer_GER"],
    "performance": [f"pct_{m}" for m in PERFORMANCE],
}
GROUP_LABELS = {"market_value": "Market value", "age": "Age", "minutes": "Minutes played", "origin": "Origin club & league",
                "position": "Position", "buyer_league": "Buying league", "performance": "Performance percentiles"}
FEATURE_LABELS = {
    "log_mv_adjusted": "Market value before (log)", "age_at_transfer": "Age", "age_sq": f"Age, distance from {REFERENCE_AGE} (squared)",
    "minutes_k": "League minutes the season before", "origin_rel_position": "Origin club's league position (0 = 1st)",
    **{f"origin_{k}": f"Bought from {v}" for k, v in LEAGUE_NAMES.items()}, **{f"buyer_{k}": f"Bought by {v} club" for k, v in LEAGUE_NAMES.items()},
    "pos_ATT": "Attacker", "pos_MID": "Midfielder", **{f"pct_{m}": f"{METRIC_LABELS[m]} percentile" for m in PERFORMANCE},
}

# Variantes comparadas na validação (a última é o modelo de preço justo).
VARIANTS = {
    "Market value only": ["market_value"],
    "Profile without market value": ["age", "minutes", "origin", "position", "buyer_league", "performance"],
    "Market value + context": ["market_value", "age", "minutes", "origin", "position", "buyer_league"],
    "Fair-price model": list(FEATURES),
}
FINAL = "Fair-price model"

VERDICT_COLORS = {"Overpaid": DIVERGING[-1][1], "Fair": MUTED, "Bargain": DIVERGING[0][1]}


def run(cfg: Config) -> dict:
    """Treina o modelo, prevê o verão corrente e grava tabelas e gráficos. Devolve o resumo."""
    processed = cfg.root / "data" / "processed"
    df = features(load(processed / "purchases.parquet"), load(processed / "percentiles.parquet"))
    out = cfg.root / "outputs" / "model"
    out.mkdir(parents=True, exist_ok=True)

    m = df[df["in_price_model"].astype(bool)].reset_index(drop=True)
    train = m[m["summer"] < SEASON].reset_index(drop=True)
    current = m[m["summer"] == SEASON].reset_index(drop=True)
    if len(train) == 0 or len(current) == 0:
        raise ValueError(f"Sem compras para treinar ({len(train)}) ou para prever ({len(current)}).")

    comparison, alpha_search = compare_variants(train)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    alpha_search.to_csv(out / "alpha_search.csv", index=False)
    alpha = float(comparison.set_index("variant").loc[FINAL, "alpha"])
    cols = columns(list(FEATURES))

    y = train["log_fee_adjusted"]
    oof = out_of_fold(train[cols], y, train["summer"], alpha)
    resid = y.to_numpy() - oof
    band = np.quantile(resid, INTERVAL)
    model = Ridge(alpha).fit(train[cols], y)

    history = predictions(train, oof, train[cols], model, band)
    ranking = predictions(current, model.predict(current[cols]), current[cols], model, band)
    save(pd.concat([history, ranking], ignore_index=True), cfg.root / "data" / "processed" / "fair_price.parquet")

    coefs = coefficient_table(model)
    coefs.to_csv(out / "coefficients.csv", index=False)
    by_summer = pd.DataFrame([{"summer": s, **scores(y[test], oof[test])} for s, _, test in summer_folds(train["summer"])])
    by_summer.round(3).to_csv(out / "cv_by_summer.csv", index=False)
    table = ranking_table(ranking)
    table.to_csv(out / "ranking_2026.csv", index=False)
    (out / "ranking_2026.json").write_text(table.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")

    summary = {"n_train": len(train), "n_rank": len(current), "alpha": alpha, "band": band, "comparison": comparison,
               "by_summer": by_summer, "coefs": coefs, "ranking": table, "excluded_2026": excluded_counts(df)}
    summary["groups"] = group_importance(model, train[cols])
    charts(history, ranking, comparison, coefs, summary, out)
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    counts = table["verdict"].value_counts().to_dict()
    log.info("Fase 4: %d compras de treino, %d no ranking de %d (%s); resultados em %s", len(train), len(current), SEASON, counts, out)
    return summary


# --------------------------------------------------------------------------- features


def features(purchases: pd.DataFrame, percentiles: pd.DataFrame) -> pd.DataFrame:
    """Tabela por compra com as colunas da lista ``FEATURES`` (e o alvo ``log_fee_adjusted``)."""
    purchases = purchases.copy()
    # Sem valor antes da transferência, usa o valor da lista de transferências do
    # Transfermarkt (igual ao de antes em todas as compras que têm os dois).
    listed = purchases["market_value_before"].isna() & purchases["market_value_listed"].notna()
    purchases.loc[listed, "market_value_before"] = purchases.loc[listed, "market_value_listed"]
    purchases["market_value_from_listing"] = listed
    df = prepare(purchases)
    pct = percentiles[KEY + [f"pct_{m}" for m in PERFORMANCE]]
    df = df.merge(pct, on=KEY, how="left")
    df["age_sq"] = (df["age_at_transfer"] - REFERENCE_AGE) ** 2
    df["minutes_k"] = df["minutes_before"] / 1000
    teams = df["origin_league_teams"].where(df["origin_league_teams"] > 1)
    df["origin_rel_position"] = (df["origin_league_position"] - 1) / (teams - 1)
    # Ligas como indicadores; a Ligue 1 (e o defesa, nas posições) é a referência.
    for k in ["ENG", "ESP", "ITA", "GER"]:
        df[f"origin_{k}"] = (df["origin_league_top5"] == k).astype(float)
        df[f"buyer_{k}"] = (df["league"] == k).astype(float)
    for pos in ["ATT", "MID"]:
        df[f"pos_{pos}"] = (df["position_group"] == pos).astype(float)
    df["market_value_missing"] = df["mv_adjusted"].isna()
    return df


def columns(groups: list[str]) -> list[str]:
    return [c for g in groups for c in FEATURES[g]]


def compare_variants(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erro fora da amostra (um verão de fora de cada vez) de cada variante, com o melhor alpha."""
    y, groups = train["log_fee_adjusted"], train["summer"]
    rows, searches = [], []
    for name, gs in VARIANTS.items():
        alpha, search = choose_alpha(train[columns(gs)], y, groups, ALPHAS)
        searches.append(search.assign(variant=name))
        best = search[search["alpha"] == alpha].iloc[0].to_dict()
        rows.append({"variant": name, "features": len(columns(gs)), **best})
    mv = train["log_mv_adjusted"].notna()
    naive = {"variant": "Fee = market value (no model)", "features": 0, "alpha": np.nan,
             **scores(y[mv], train.loc[mv, "log_mv_adjusted"])}
    table = pd.DataFrame([naive] + rows)
    table[["n", "features"]] = table[["n", "features"]].astype(int)
    return table, pd.concat(searches, ignore_index=True)


# --------------------------------------------------------------------------- previsões


def predictions(rows: pd.DataFrame, pred_log: np.ndarray, X: pd.DataFrame, model: Ridge, band: np.ndarray) -> pd.DataFrame:
    """Preço justo, intervalo de 80%, veredicto e parte de cada grupo de features (em log)."""
    out = rows[KEY + ["player_name", "club_name", "league", "other_club_name", "origin_league_top5", "position_group", "position",
                      "age_at_transfer", "minutes_before", "market_value_before", "market_value_from_listing", "market_value_missing",
                      "bought_twice_in_summer", "fee", "fee_index", "fee_adjusted"]].copy()
    pred_log = np.asarray(pred_log, float)
    # Em preços de 2026; o preço justo volta aos euros do verão com o índice.
    out["fair_price"] = np.exp(pred_log) * out["fee_index"]
    out["fair_low"] = np.exp(pred_log + band[0]) * out["fee_index"]
    out["fair_high"] = np.exp(pred_log + band[1]) * out["fee_index"]
    out["fee_vs_fair"] = out["fee"] / out["fair_price"]
    out["verdict"] = np.select([out["fee"] > out["fair_high"], out["fee"] < out["fair_low"]], ["Overpaid", "Bargain"], "Fair")
    out["out_of_sample"] = rows["summer"].to_numpy() < SEASON
    # Partes de cada grupo (modelo final): só batem com a previsão no verão corrente,
    # porque nos outros a previsão vem do modelo que não viu esse verão.
    contrib = model.contributions(X)
    for g, cols in FEATURES.items():
        out[f"contrib_{g}"] = np.where(out["out_of_sample"], np.nan, contrib[cols].sum(axis=1).to_numpy())
    out["baseline_price"] = np.exp(model.intercept)
    return out


def ranking_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Ranking do verão corrente: do mais sobrepago (preço / preço justo maior) à maior pechincha."""
    t = pred.sort_values("fee_vs_fair", ascending=False).reset_index(drop=True)
    t.insert(0, "rank", range(1, len(t) + 1))
    t["origin_league"] = t["origin_league_top5"].map(LEAGUE_NAMES)
    t["buying_league"] = t["league"].map(LEAGUE_NAMES)
    keep = ["rank", "player_name", "club_name", "buying_league", "other_club_name", "origin_league", "position_group", "position",
            "age_at_transfer", "minutes_before", "market_value_before", "fee", "fair_price", "fair_low", "fair_high", "fee_vs_fair",
            "verdict", "market_value_from_listing", "market_value_missing", "bought_twice_in_summer", "summer", "player_id", "club_id"]
    keep += [c for c in t.columns if c.startswith("contrib_")]
    t = t[keep].rename(columns={"club_name": "buying_club", "other_club_name": "origin_club", "age_at_transfer": "age"})
    for c in ["fair_price", "fair_low", "fair_high"]:
        t[c] = t[c].round(-4)
    return t.round({"age": 1, "fee_vs_fair": 3, **{c: 3 for c in t.columns if c.startswith("contrib_")}})


def coefficient_table(model: Ridge) -> pd.DataFrame:
    """Efeito de +1 desvio-padrão de cada feature no preço (multiplicador), com o grupo."""
    c = model.coefficients
    group = {col: g for g, cols in FEATURES.items() for col in cols}
    t = pd.DataFrame({"feature": c.index, "label": [FEATURE_LABELS[f] for f in c.index], "group": [group[f] for f in c.index],
                      "coef_std_log": c.to_numpy(), "price_multiplier_per_sd": np.exp(c.to_numpy()), "feature_sd": model.std[c.index].to_numpy()})
    return t.reindex(t["coef_std_log"].abs().sort_values(ascending=False).index).round(4).reset_index(drop=True)


def group_importance(model: Ridge, X: pd.DataFrame) -> pd.Series:
    """Desvio-padrão da parte de cada grupo na previsão do treino (em log): o que mais mexe no preço."""
    contrib = model.contributions(X)
    return pd.Series({g: contrib[cols].sum(axis=1).std() for g, cols in FEATURES.items()}).sort_values(ascending=False)


def excluded_counts(df: pd.DataFrame) -> dict:
    cur = df[df["summer"] == SEASON]
    return {"total": len(cur), **cur["model_exclusion"].value_counts().to_dict()}


# --------------------------------------------------------------------------- gráficos


def charts(history: pd.DataFrame, ranking: pd.DataFrame, comparison: pd.DataFrame, coefs: pd.DataFrame, s: dict, out: Path) -> None:
    lo, hi = np.exp(s["band"])
    # 1. Preço pago vs. preço justo em 2026, com o intervalo de 80%.
    r = ranking.copy()
    fig = go.Figure()
    for verdict in ["Overpaid", "Fair", "Bargain"]:
        g = r[r["verdict"] == verdict]
        fig.add_trace(go.Scatter(
            x=g["fair_price"] / 1e6, y=g["fee"] / 1e6, mode="markers", name=f"{verdict} ({len(g)})",
            marker=dict(color=VERDICT_COLORS[verdict], size=9, opacity=0.8, line=dict(width=1.5, color="white")),
            customdata=np.c_[g["club_name"], g["fair_low"] / 1e6, g["fair_high"] / 1e6, g["fee_vs_fair"]], text=g["player_name"],
            hovertemplate="<b>%{text}</b> to %{customdata[0]}<br>Paid €%{y:.1f}m · fair €%{x:.1f}m (80%: €%{customdata[1]:.1f}–%{customdata[2]:.1f}m)"
                          "<br>%{customdata[3]:.2f}× the fair price<extra></extra>"))
    span = np.array([r[["fair_price", "fee"]].min().min(), r[["fair_price", "fee"]].max().max()]) / 1e6 * [0.8, 1.25]
    fig.add_trace(go.Scatter(x=span, y=span, mode="lines", line=dict(color=TEXT_SECONDARY, width=1.5), name="Paid = fair price", hoverinfo="skip"))
    for k, f in [("upper", hi), ("lower", lo)]:
        fig.add_trace(go.Scatter(x=span, y=span * f, mode="lines", line=dict(color=MUTED, dash="dot", width=1.5), hoverinfo="skip",
                                 name="80% interval", legendgroup="band", showlegend=k == "upper"))
    fig.update_xaxes(**log_axis("Fair price (€m)"))
    fig.update_yaxes(**log_axis("Fixed fee paid (€m)"))
    style(fig, f"Summer {SEASON}: fee paid vs. fair price",
          f"n={len(r)} purchases (≥900 league minutes the season before, known fee). Above the band: overpaid; below: bargain.")
    export(fig, out, "01_fee_vs_fair_2026")

    # 2. Os mais sobrepagos e as maiores pechinchas (preço / preço justo, escala log).
    top = pd.concat([r.nlargest(15, "fee_vs_fair"), r.nsmallest(15, "fee_vs_fair")]).drop_duplicates(KEY)
    top = top.sort_values("fee_vs_fair")
    # Barras a partir de 1× (preço justo): log2 da razão num eixo linear, com marcas em ×.
    fig = go.Figure(go.Bar(
        x=np.log2(top["fee_vs_fair"]), y=top["player_name"] + " · " + top["club_name"], orientation="h",
        marker=dict(color=[VERDICT_COLORS[v] for v in top["verdict"]], cornerradius=4),
        customdata=np.c_[top["fee"] / 1e6, top["fair_price"] / 1e6, top["verdict"], top["fee_vs_fair"]],
        hovertemplate="%{y}<br>Paid €%{customdata[0]:.1f}m · fair €%{customdata[1]:.1f}m<br>%{customdata[3]:.2f}× (%{customdata[2]})<extra></extra>",
        text=[f"{v:.2f}×" for v in top["fee_vs_fair"]], textposition="outside", textfont=dict(color=TEXT_SECONDARY)))
    ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    fig.update_xaxes(title="Fee paid ÷ fair price (log scale; 1× = fair)", tickvals=np.log2(ticks), ticktext=[f"{t:g}×" for t in ticks],
                     range=[np.log2(top["fee_vs_fair"].min()) - 0.6, np.log2(top["fee_vs_fair"].max()) + 0.6])
    fig.add_vline(x=0, line=dict(color=TEXT_SECONDARY, width=1))
    fig.update_layout(height=820, margin=dict(l=260))
    style(fig, f"Most overpaid and biggest bargains, summer {SEASON}",
          f"15 highest and 15 lowest ratios of n={len(r)}. Red: above the 80% interval; blue: below; grey: inside it.")
    export(fig, out, "02_ranking_2026")

    # 3. Coeficientes: efeito de +1 desvio-padrão no preço.
    c = coefs.sort_values("coef_std_log")
    pct = (c["price_multiplier_per_sd"] - 1) * 100
    fig = go.Figure(go.Bar(
        x=pct, y=c["label"], orientation="h", marker=dict(color=[ACCENT if v >= 0 else MUTED for v in pct], cornerradius=4),
        customdata=c["group"].map(GROUP_LABELS), hovertemplate="%{y} (%{customdata})<br>+1 SD: %{x:+.1f}% on the fair price<extra></extra>"))
    fig.update_xaxes(title="Change in fair price for +1 standard deviation (%)", ticksuffix="%")
    fig.update_layout(height=900, margin=dict(l=320))
    style(fig, "What moves the fair price", f"Ridge coefficients (alpha={s['alpha']:g}), trained on {s['n_train']:,} purchases 2019–2025. "
          "Other features held fixed; correlated features share the effect.")
    export(fig, out, "03_coefficients")

    # 4. Variantes: erro fora da amostra.
    comp = comparison.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=comp["median_abs_pct_error"], y=comp["variant"], orientation="h",
        marker=dict(color=[ACCENT if v == FINAL else MUTED for v in comp["variant"]], cornerradius=4),
        customdata=np.c_[comp["rmse_log"], comp["r2_log"], comp["n"]],
        hovertemplate="%{y}<br>Median error %{x:.1f}% · RMSE (log) %{customdata[0]:.3f} · R² %{customdata[1]:.2f} · n=%{customdata[2]}<extra></extra>",
        text=[f"{v:.0f}% · R² {r2:.2f}" for v, r2 in zip(comp["median_abs_pct_error"], comp["r2_log"])], textposition="outside",
        textfont=dict(color=TEXT_SECONDARY)))
    fig.update_xaxes(title="Median absolute error on the fee (%), each summer predicted without seeing it", ticksuffix="%",
                     range=[0, comp["median_abs_pct_error"].max() * 1.25])
    fig.update_layout(height=420, margin=dict(l=240))
    style(fig, "How well each variant predicts unseen summers", f"Leave-one-summer-out on 2019–2025 (n={s['n_train']:,}). "
          "Market value carries most of the signal.")
    export(fig, out, "04_model_comparison")

    # 5. Verificação: previsão fora da amostra vs. preço real, 2019–2025.
    h = history
    fig = go.Figure()
    for pos in POSITIONS:
        g = h[h["position_group"] == pos]
        fig.add_trace(go.Scattergl(
            x=g["fair_price"] / g["fee_index"] / 1e6, y=g["fee_adjusted"] / 1e6, mode="markers", name=POSITION_LABELS[pos],
            marker=dict(color=POSITION_COLORS[pos], size=7, opacity=0.6, line=dict(width=1, color="white")),
            text=g["player_name"] + " (" + g["summer"].astype(str) + ", " + g["club_name"] + ")",
            hovertemplate="%{text}<br>Paid €%{y:.1f}m · predicted €%{x:.1f}m (2026 prices)<extra></extra>"))
    span = np.array([h["fee_adjusted"].min(), h["fee_adjusted"].max()]) / 1e6
    fig.add_trace(go.Scatter(x=span, y=span, mode="lines", line=dict(color=TEXT_SECONDARY, width=1.5), name="Paid = predicted", hoverinfo="skip"))
    fig.update_xaxes(**log_axis("Predicted fee without seeing that summer (€m, 2026 prices)"))
    fig.update_yaxes(**log_axis("Fixed fee paid (€m, 2026 prices)"))
    inside = float(((h["verdict"] == "Fair").mean()) * 100)
    style(fig, "Back-test: each past summer predicted from the others", f"n={len(h):,} purchases 2019–2025; {inside:.0f}% fall inside the 80% interval.")
    export(fig, out, "05_backtest")


# --------------------------------------------------------------------------- resumo


def summary_markdown(s: dict) -> str:
    comp, t = s["comparison"].set_index("variant"), s["ranking"]
    final = comp.loc[FINAL]
    lo, hi = np.exp(s["band"])
    counts = t["verdict"].value_counts()
    ex = s["excluded_2026"]
    lines = [
        "# Phase 4: fair-price model (summary)",
        "",
        "Generated by `uv run main.py model` from `data/processed/purchases.parquet` and `percentiles.parquet` (offline). "
        "Charts are in this folder as `.html` (iframe fragment) and `.json` (Plotly); predictions for every model purchase are in "
        "`data/processed/fair_price.parquet`.",
        "",
        "## Sample (small: read the ranking with its interval)",
        f"- Training: {s['n_train']:,} purchases 2019–2025 (top-5 to top-5, known fee, ≥900 league minutes the season before).",
        f"- Ranking: {s['n_rank']} of {ex['total']} summer-{SEASON} purchases. Out: {ex.get('under_min_minutes_before', 0)} under 900 minutes, "
        f"{ex.get('no_league_minutes_before', 0)} with no league minutes, {ex.get('unknown_fee', 0)} with an unknown fee; they stay in the data for phase 5.",
        "- About 1,100 purchases for ~30 features is a small sample: per-player results carry a wide interval, and a single season's stats are noisy.",
        "",
        "## Model",
        f"Ridge regression on log(fixed fee in 2026 prices), alpha={s['alpha']:g} chosen by leaving one summer out at a time. "
        "Features come from an explicit list (`FEATURES` in `src/pechincha/model/fair_price.py`): Transfermarkt value before the transfer, age (and its square), "
        "league minutes, origin club's league position and origin league, position group, buying league, and 16 per-position percentiles "
        "(one of each redundant pair from the EDA).",
        "",
        "| Variant | Features | Median error | Within 25% | RMSE (log) | R² (log) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in comp.iterrows():
        lines.append(f"| {name} | {r['features']} | {r['median_abs_pct_error']:.0f}% | {r['within_25pct']:.0f}% | {r['rmse_log']:.3f} | {r['r2_log']:.2f} |")
    mv_only = comp.loc["Market value only"]
    ctx = comp.loc["Market value + context"]
    lines += [
        "",
        f"- Every variant is scored on summers it did not see (n={int(final['n']):,}). The fair-price model explains {final['r2_log'] * 100:.0f}% of the variance in log fee "
        f"against {mv_only['r2_log'] * 100:.0f}% for market value alone; the percentiles add little on top of market value and context "
        f"(RMSE {ctx['rmse_log']:.3f} → {final['rmse_log']:.3f}), because Transfermarkt values already price in performance.",
        f"- Taking the fee to be the market value has a lower median error ({comp.iloc[0]['median_abs_pct_error']:.0f}%) but bigger misses "
        f"(RMSE {comp.iloc[0]['rmse_log']:.3f}): the model corrects the deals where the market value is far off (young players, Premier League buyers).",
        f"- Typical miss: half the fees are within {final['median_abs_pct_error']:.0f}% of the prediction. Fees depend on things no stat sees (clauses, contract length, urgency).",
        f"- 80% interval: from {lo:.2f}× to {hi:.2f}× the fair price (10th and 90th percentile of the out-of-sample errors). "
        "Outside it a purchase is called overpaid or a bargain.",
        "- Buying league is a feature, so the fair price is fair *for a club in that league*: the Premier League premium is treated as the market, not as overpaying.",
        "",
        "What moves the price most (spread of each group's part in the prediction, log units): "
        + ", ".join(f"{GROUP_LABELS[g]} {v:.2f}" for g, v in s["groups"].items()) + ".",
        "",
        f"## Summer {SEASON}",
        f"- {counts.get('Overpaid', 0)} overpaid, {counts.get('Fair', 0)} fair, {counts.get('Bargain', 0)} bargains (n={len(t)}).",
        "",
        "Most overpaid (fee ÷ fair price):",
        "",
        "| # | Player | Buyer | Fee | Fair (80% interval) | Ratio |",
        "|---|---|---|---|---|---|",
    ]
    for r in t.head(10).itertuples():
        lines.append(f"| {r.rank} | {r.player_name} | {r.buying_club} | {_money(r.fee)} | {_money(r.fair_price)} ({_money(r.fair_low)}–{_money(r.fair_high)}) | {r.fee_vs_fair:.2f}× |")
    lines += ["", "Biggest bargains:", "", "| # | Player | Buyer | Fee | Fair (80% interval) | Ratio |", "|---|---|---|---|---|---|"]
    for r in t.tail(10).iloc[::-1].itertuples():
        lines.append(f"| {r.rank} | {r.player_name} | {r.buying_club} | {_money(r.fee)} | {_money(r.fair_price)} ({_money(r.fair_low)}–{_money(r.fair_high)}) | {r.fee_vs_fair:.2f}× |")
    missing = t[t["market_value_missing"]]
    listed = t[t["market_value_from_listing"]]
    lines += [
        "",
        "## Files",
        "- `ranking_2026.csv` / `.json`: every ranked purchase with fee, fair price, interval, verdict and `contrib_*` "
        "(each feature group's part of the predicted log price, relative to the average training purchase).",
        "- `model_comparison.csv`, `alpha_search.csv`, `cv_by_summer.csv`, `coefficients.csv`.",
        "",
        "## Limits",
        "- Market value is both the strongest feature and partly an opinion of the market itself: the model judges a fee against what similar deals cost, not against a player's true worth.",
        "- Fixed fee only: add-ons are not in the fee, so deals with large bonuses look cheaper than they are.",
        "- Years left on the selling club's contract are unknown (Transfermarkt keeps only the current contract).",
        "- Percentiles compare bought players of the same position across all summers; one season of stats is noisy.",
    ]
    if len(listed):
        lines.append(f"- {len(listed)} ranked purchase(s) have no market value before the transfer in the player history and use the value on "
                     "Transfermarkt's transfer list (identical wherever both exist): " + ", ".join(listed["player_name"]) + ".")
    if len(missing):
        lines.append(f"- {len(missing)} ranked purchase(s) have no market value before the transfer and use the training median: "
                     + ", ".join(missing["player_name"]) + ".")
    return "\n".join(lines) + "\n"

