"""Fase 5: desempenho das compras do verão de 2026 no clube novo.

Só lê o que já está no disco: ``data/processed`` (``purchases.parquet`` do
``build`` e, se existir, ``fair_price.parquet`` do ``model``) e as tabelas da
época 2026/27 em ``data/interim`` (jogos do Understat e totais de liga do
Sofascore). Não faz pedidos à rede, por isso volta a correr depois de cada
atualização semanal. ``uv run main.py performance`` grava:

- ``data/processed/performance.parquet``: uma linha por compra de 2026, com
  minutos no clube novo, as métricas por 90 antes e depois, percentis, o
  índice custo/desempenho e os resultados da equipa com e sem o jogador.
- ``outputs/performance/*.html`` e ``*.json``: gráficos Plotly no formato da Fase 6.
- ``outputs/performance/*.csv`` e ``*.json``: as tabelas por trás dos gráficos.
- ``outputs/performance/summary.md``: os números principais, sempre com a amostra.

Só contam os jogos de liga pelo clube comprador. O Understat dá os jogos um a
um (minutos, xG, equipa), por isso as métricas do Understat são só do clube
novo. O Sofascore só tem o total da época por liga e equipa: quem jogou na
mesma liga por outro clube antes de mudar fica sem as métricas do Sofascore
(o total misturava os dois clubes).

Amostra mínima: ``phase5.min_minutes_after`` minutos no clube novo, ou
``phase5.min_share_of_team_minutes`` dos minutos de liga da equipa desde a
chegada, o que for maior. No início da época manda o primeiro; o segundo
sobe com a época, por isso o critério acompanha a atualização semanal.

Percentis: contra a mesma referência da Fase 3 (compras do modelo de preço da
mesma posição, todos os verões, com pelo menos 900 minutos na época anterior),
para o antes e o depois estarem na mesma escala.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import build
from ..config import Config
from ..eda.analysis import KEY, LEAGUE_NAMES, LOWER_IS_BETTER, METRIC_LABELS, POSITIONS, PROFILE_METRICS, REFERENCE_MIN_MINUTES, _money
from ..ingest.sofascore import normalize_name
from ..pipeline import load, save
from ..report.theme import ACCENT, DIVERGING, MUTED, POSITION_COLORS, POSITION_LABELS, TEXT_SECONDARY, export, style

log = logging.getLogger(__name__)

SEASON = 2026
DEFAULT_MIN_MINUTES = 270
DEFAULT_MIN_SHARE = 0.3

# Métricas do Understat que os jogos dão um a um. Os jogos não separam os
# penáltis, por isso compara-se xG e golos (com penáltis), não as versões NP.
US_STATS = ["goals", "xg", "assists", "xa", "shots", "key_passes", "xg_chain", "xg_buildup"]
SS_RATIOS = ["ss_pass_accuracy", "ss_dribble_success", "ss_duels_won_pct", "ss_rating"]
COMPARE_METRICS = ([f"us_{c}_p90" for c in US_STATS] + [f"ss_{c}_p90" for c in build.SOFASCORE_PER90 if c != "ballRecovery"] + SS_RATIOS)
# Pontuação de desempenho: média dos percentis do perfil de cada posição (os do radar da Fase 3).
SCORE_METRICS = {pos: [m.replace("us_np_xg_p90", "us_xg_p90") for m in ms] for pos, ms in PROFILE_METRICS.items()}
METRIC_LABELS = {**METRIC_LABELS, "us_xg_p90": "xG"}

STATUS_LABELS = {
    "qualified": "Enough minutes",
    "under_min_minutes": "Under the minimum",
    "no_minutes": "No league minutes yet",
    "playing_elsewhere": "Playing for another club",
}
STATUS_COLORS = {"qualified": ACCENT, "under_min_minutes": MUTED, "no_minutes": "#c9c8c3", "playing_elsewhere": DIVERGING[-1][1]}


def run(cfg: Config) -> dict:
    """Junta os dados da época 2026/27, calcula as métricas e grava tabelas e gráficos. Devolve o resumo."""
    processed = cfg.root / "data" / "processed"
    inter = cfg.interim_dir
    purchases = load(processed / "purchases.parquet")
    understat = build.understat_seasons(load(inter / "understat_player_seasons.parquet"))
    clubs = build.club_map(load(inter / "tm_top5_clubs.parquet"), understat)
    matches = load(inter / f"understat_{SEASON}_player_matches.parquet")
    schedule = load(inter / f"understat_{SEASON}_schedule.parquet")
    sofa = load(inter / "sofascore_league_player_seasons.parquet")
    fair_path = processed / "fair_price.parquet"
    fair = load(fair_path) if fair_path.exists() else None
    rules = {"min_minutes_after": DEFAULT_MIN_MINUTES, "min_share_of_team_minutes": DEFAULT_MIN_SHARE, **cfg.phase5}

    games = team_games(schedule)
    df = performance_table(purchases, clubs, matches, games, sofa, fair, rules)
    team = team_results(df, matches, games)
    df = df.merge(team.drop(columns=["player_name", "club_name"]), on=KEY, how="left")
    save(df, processed / "performance.parquet")

    out = cfg.root / "outputs" / "performance"
    out.mkdir(parents=True, exist_ok=True)
    table = output_table(df)
    table.to_csv(out / "performance_2026.csv", index=False)
    (out / "performance_2026.json").write_text(table.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
    team_out = team[team["matches_since_arrival"] > 0].sort_values("ppg_difference", ascending=False)
    team_out.round(3).to_csv(out / "team_results_2026.csv", index=False)
    (out / "team_results_2026.json").write_text(team_out.round(3).to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
    changes = metric_changes(df)
    changes.to_csv(out / "metric_changes.csv", index=False)

    summary = {"rules": rules, "season": season_progress(games), "df": df, "table": table, "team": team, "changes": changes}
    charts(df, team, changes, summary, out)
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    counts = df["status"].value_counts().to_dict()
    log.info("Fase 5: %d compras de %d (%s); resultados em %s", len(df), SEASON, counts, out)
    return summary


# --------------------------------------------------------------------------- jogos da época


def team_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Um jogo com resultado por equipa (duas linhas por jogo): golos, xG e pontos do lado de cada equipa."""
    s = schedule[schedule["is_result"].astype(bool)].copy()
    s["lg"] = s["league"].map(build.UNDERSTAT_LEAGUES)
    s["date"] = pd.to_datetime(s["date"])
    sides = []
    for side, other in [("home", "away"), ("away", "home")]:
        sides.append(pd.DataFrame({
            "lg": s["lg"], "game_id": s["game_id"].astype(int), "date": s["date"], "team": s[f"{side}_team"], "opponent": s[f"{other}_team"],
            "home": side == "home", "goals_for": s[f"{side}_goals"].astype(float), "goals_against": s[f"{other}_goals"].astype(float),
            "xg_for": s[f"{side}_xg"].astype(float), "xg_against": s[f"{other}_xg"].astype(float),
        }))
    g = pd.concat(sides, ignore_index=True)
    g["points"] = np.select([g["goals_for"] > g["goals_against"], g["goals_for"] == g["goals_against"]], [3, 1], 0)
    return g.sort_values(["lg", "team", "date"]).reset_index(drop=True)


def season_progress(games: pd.DataFrame) -> pd.DataFrame:
    """Jornadas jogadas por liga (jogos por equipa, mínimo e máximo) e data do último resultado."""
    per_team = games.groupby(["lg", "team"]).size()
    out = per_team.groupby("lg").agg(["min", "max"]).rename(columns={"min": "min_team_matches", "max": "max_team_matches"})
    out["last_result"] = games.groupby("lg")["date"].max().dt.date
    return out.reset_index()


# --------------------------------------------------------------------------- tabela por compra


def _same_name(a: str, b: str) -> float:
    """Semelhança entre dois nomes (1 se um contém todas as palavras do outro)."""
    wa, wb = set(normalize_name(a).replace("'", "").split()), set(normalize_name(b).replace("'", "").split())
    if not wa or not wb:
        return 0.0
    if wa <= wb or wb <= wa:
        return 1.0
    return SequenceMatcher(None, " ".join(sorted(wa)), " ".join(sorted(wb))).ratio()


def _best_name(name: str, candidates: pd.Series, threshold: float = 0.8):
    """Índice do candidato com o nome mais parecido (ou None abaixo de ``threshold``)."""
    best, score = None, threshold
    for idx, cand in candidates.items():
        s = _same_name(name, cand)
        if s >= score:
            best, score = idx, s
    return best


def performance_table(purchases: pd.DataFrame, clubs: pd.DataFrame, matches: pd.DataFrame, games: pd.DataFrame,
                      sofa: pd.DataFrame, fair: pd.DataFrame | None, rules: dict) -> pd.DataFrame:
    """Uma linha por compra de 2026: minutos e métricas por 90 no clube novo, antes e depois, e o índice."""
    p = purchases[purchases["in_phase5"].astype(bool)].copy().reset_index(drop=True)
    teams = clubs[clubs["season"] == SEASON][["club_id", "league", "understat_team"]]
    p = p.merge(teams, on=["club_id", "league"], how="left")
    p["arrival"] = pd.to_datetime(p["transfer_date"], errors="coerce")

    m = matches.copy()
    m["lg"] = m["league"].map(build.UNDERSTAT_LEAGUES)
    m["date"] = pd.to_datetime(m["game"].str[:10])
    m["player_id"] = m["player_id"].astype(int)
    m = m[m["minutes"] > 0]

    s26 = sofa[sofa["season"] == SEASON].copy()
    pool = games.groupby("lg")["team"].unique()
    team_of = {(lg, t): build._closest_team(t, pd.Series(pool.get(lg, []))) for lg, t in s26[["league", "team_name"]].drop_duplicates().itertuples(index=False)}
    s26["understat_team"] = [team_of[(lg, t)] for lg, t in zip(s26["league"], s26["team_name"])]

    rows = []
    for r in p.itertuples():
        row = {"summer": r.summer, "player_id": r.player_id, "club_id": r.club_id}
        team_m = m[(m["lg"] == r.league) & (m["team"] == r.understat_team)]
        # Jogador no Understat: o id da Fase 2; sem ele (não jogou na época anterior), pelo nome no plantel novo.
        uid, link = (int(r.understat_id), "id") if pd.notna(r.understat_id) else (None, None)
        if uid is None and len(team_m):
            names = team_m.drop_duplicates("player_id").set_index("player_id")["player"]
            uid = _best_name(r.player_name, names)
            link = "name" if uid is not None else None
        row["understat_id_2026"], row["understat_link_2026"] = uid, link

        mine = m[m["player_id"] == uid] if uid is not None else m.iloc[:0]
        at_club = mine[(mine["lg"] == r.league) & (mine["team"] == r.understat_team)]
        elsewhere = mine.drop(at_club.index)
        row["minutes_after"] = float(at_club["minutes"].sum())
        row["appearances_after"] = len(at_club)
        row["starts_after"] = int((at_club["position"] != "Sub").sum())
        row["minutes_elsewhere"] = float(elsewhere["minutes"].sum())
        after_arrival = elsewhere if pd.isna(r.arrival) else elsewhere[elsewhere["date"] >= r.arrival]
        row["club_elsewhere"] = ", ".join(sorted(after_arrival["team"].unique())) or None
        for c in US_STATS:
            row[f"us_{c}_after"] = float(at_club[c].sum()) if len(at_club) else np.nan

        # Equipa: jogos de liga desde a chegada (antes dela o jogador não estava lá).
        tg = games[(games["lg"] == r.league) & (games["team"] == r.understat_team)]
        if pd.notna(r.arrival):
            tg = tg[tg["date"] >= r.arrival.normalize()]
        row["team_matches_since_arrival"] = len(tg)

        # Sofascore: total da liga pela equipa nova; vazio se jogou na mesma liga por outro clube.
        cand = s26[(s26["league"] == r.league) & (s26["understat_team"] == r.understat_team)]
        hit = cand[cand["sofascore_id"] == r.sofascore_id] if pd.notna(r.sofascore_id) else cand.iloc[:0]
        if hit.empty and pd.isna(r.sofascore_id) and len(cand):
            # Pelo nome do Transfermarkt ou do Understat ("Eric Ebimbe" só bate com "Eric Junior Dina Ebimbe").
            for name in [r.player_name, *at_club["player"].unique()[:1]]:
                idx = _best_name(name, cand["sofascore_name"])
                if idx is not None:
                    hit = cand.loc[[idx]]
                    break
        mixed = bool((elsewhere["lg"] == r.league).any())
        row["ss_mixed_clubs"] = mixed and not hit.empty
        if not hit.empty and not mixed:
            h = hit.iloc[0]
            for c in build.SOFASCORE_STATS:
                row[f"ss_{c}_after"] = float(h[c]) if pd.notna(h[c]) else np.nan
            row["ss_rating_after"] = float(h["rating"]) if pd.notna(h["rating"]) else np.nan
        rows.append(row)

    a = pd.DataFrame(rows)
    df = p.merge(a, on=KEY, how="left")
    for c in [f"ss_{c}_after" for c in build.SOFASCORE_STATS] + ["ss_rating_after"]:
        if c not in df:
            df[c] = np.nan
    df = df.join(per90_after(df))

    # Amostra mínima: o maior entre o mínimo fixo e uma parte dos minutos da equipa desde a chegada.
    df["min_minutes_required"] = np.maximum(rules["min_minutes_after"], rules["min_share_of_team_minutes"] * 90 * df["team_matches_since_arrival"])
    df["status"] = np.select(
        [df["minutes_after"] >= df["min_minutes_required"], df["minutes_after"] > 0, df["club_elsewhere"].notna()],
        ["qualified", "under_min_minutes", "playing_elsewhere"], "no_minutes")
    df["qualified"] = df["status"] == "qualified"

    df = df.join(scores(purchases, df))
    df = df.join(cost_performance(df))
    if fair is not None:
        f = fair[fair["summer"] == SEASON][KEY + ["fair_price", "fair_low", "fair_high", "fee_vs_fair", "verdict"]]
        df = df.merge(f, on=KEY, how="left")
    else:
        df[["fair_price", "fair_low", "fair_high", "fee_vs_fair", "verdict"]] = np.nan
    return df


def per90_after(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas por 90 e rácios da época 2026/27 no clube novo, com os nomes das de antes (``_after`` no fim)."""
    us90 = 90 / df["minutes_after"].where(df["minutes_after"] > 0)
    ss_min = df["ss_minutesPlayed_after"]
    ss90 = 90 / ss_min.where(ss_min > 0)
    out = {f"us_{c}_p90_after": df[f"us_{c}_after"] * us90 for c in US_STATS}
    out |= {f"ss_{c}_p90_after": df[f"ss_{c}_after"] * ss90 for c in build.SOFASCORE_PER90}
    out["ss_pass_accuracy_after"] = df["ss_accuratePasses_after"] / df["ss_totalPasses_after"].where(df["ss_totalPasses_after"] > 0)
    out["ss_dribble_success_after"] = df["ss_successfulDribbles_after"] / df["ss_totalContest_after"].where(df["ss_totalContest_after"] > 0)
    duels = df["ss_totalDuelsWon_after"] + df["ss_duelLost_after"]
    out["ss_duels_won_pct_after"] = df["ss_totalDuelsWon_after"] / duels.where(duels > 0)
    return pd.DataFrame(out, index=df.index).astype(float)


def reference(purchases: pd.DataFrame) -> pd.DataFrame:
    """Referência dos percentis: compras do modelo de preço com ≥900 minutos na época anterior (como na Fase 3)."""
    ok = purchases["in_price_model"].astype(bool) & (purchases["minutes_before"].fillna(0) >= REFERENCE_MIN_MINUTES)
    return purchases[ok]


def percentile_of(values: pd.Series, groups: pd.Series, ref: pd.DataFrame, metric: str) -> pd.Series:
    """Percentil (0–100, 100 = melhor) de cada valor contra a referência da sua posição; empates contam metade."""
    out = pd.Series(np.nan, index=values.index)
    for pos in groups.dropna().unique():
        r = np.sort(ref.loc[ref["position_group"] == pos, metric].dropna().astype(float).to_numpy())
        idx = values.index[(groups == pos) & values.notna()]
        if not len(r) or not len(idx):
            continue
        v = values.loc[idx].astype(float).to_numpy()
        p = (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / 2 / len(r) * 100
        out.loc[idx] = 100 - p if metric in LOWER_IS_BETTER else p
    return out.round(1)


def scores(purchases: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Percentis antes e depois de cada métrica, a pontuação do perfil da posição e a variação."""
    ref = reference(purchases)
    before_ok = df["minutes_before"].fillna(0) > 0
    after_ok = df["minutes_after"] > 0
    out = {}
    for metric in COMPARE_METRICS:
        out[f"pct_{metric}_before"] = percentile_of(df[metric].where(before_ok), df["position_group"], ref, metric)
        out[f"pct_{metric}_after"] = percentile_of(df[f"{metric}_after"].where(after_ok), df["position_group"], ref, metric)
    out = pd.DataFrame(out, index=df.index)
    for when in ["before", "after"]:
        score = pd.Series(np.nan, index=df.index)
        for pos, metrics in SCORE_METRICS.items():
            rows = df["position_group"] == pos
            pct = out.loc[rows, [f"pct_{m}_{when}" for m in metrics]]
            # Pelo menos metade das métricas do perfil (faltam as do Sofascore a quem mudou dentro da liga).
            score.loc[rows] = pct.mean(axis=1).where(pct.notna().sum(axis=1) >= len(metrics) / 2)
        out[f"score_{when}"] = score.round(1)
    out["score_change"] = out["score_after"] - out["score_before"]
    return out


def cost_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Índice custo/desempenho: percentil de desempenho em 2026/27 menos o percentil do preço na posição.

    O percentil do preço é contra as compras de 2026 da mesma posição com preço
    conhecido. Positivo: rende acima do que custou, comparado com os outros
    comprados para a mesma posição; negativo: abaixo. Só com a amostra mínima.
    """
    known = df["fee_known"].astype(bool) & (df["fee"] > 0)
    fee_pct = pd.Series(np.nan, index=df.index)
    for pos, g in df[known].groupby("position_group"):
        fee_pct.loc[g.index] = (g["fee"].rank(method="average") - 0.5) / len(g) * 100
    index = (df["score_after"] - fee_pct).where(df["qualified"] & known)
    per_point = (df["fee"] / df["score_after"].where(df["score_after"] > 0)).where(df["qualified"] & known)
    return pd.DataFrame({"fee_pct_position": fee_pct.round(1), "value_index": index.round(1), "fee_per_score_point": per_point.round(-3)})


# --------------------------------------------------------------------------- resultados da equipa


def team_results(df: pd.DataFrame, matches: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Pontos, golos e xG por jogo da equipa nova desde a chegada, nos jogos em que o jogador jogou e nos outros."""
    m = matches[matches["minutes"] > 0]
    played = set(zip(m["player_id"].astype(int), m["team"], m["game_id"].astype(int)))
    rows = []
    for r in df.itertuples():
        tg = games[(games["lg"] == r.league) & (games["team"] == r.understat_team)]
        if pd.notna(r.arrival):
            tg = tg[tg["date"] >= r.arrival.normalize()]
        uid = r.understat_id_2026
        with_ = tg["game_id"].map(lambda g: uid is not None and pd.notna(uid) and (int(uid), r.understat_team, g) in played).astype(bool)
        row = {"summer": r.summer, "player_id": r.player_id, "club_id": r.club_id, "player_name": r.player_name, "club_name": r.club_name,
               "matches_since_arrival": len(tg), "matches_with": int(with_.sum()), "matches_without": int((~with_).sum())}
        for name, g in [("with", tg[with_]), ("without", tg[~with_])]:
            n = len(g)
            row[f"ppg_{name}"] = g["points"].mean() if n else np.nan
            row[f"gd_per_match_{name}"] = (g["goals_for"] - g["goals_against"]).mean() if n else np.nan
            row[f"xgd_per_match_{name}"] = (g["xg_for"] - g["xg_against"]).mean() if n else np.nan
        row["ppg_difference"] = row["ppg_with"] - row["ppg_without"]
        row["xgd_difference"] = row["xgd_per_match_with"] - row["xgd_per_match_without"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- tabelas de saída


def metric_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Mediana da variação de percentil (depois - antes) por métrica e posição, só com a amostra mínima e minutos antes."""
    q = df[df["qualified"] & (df["minutes_before"].fillna(0) > 0)]
    rows = []
    for pos in POSITIONS:
        g = q[q["position_group"] == pos]
        for metric in COMPARE_METRICS:
            d = (g[f"pct_{metric}_after"] - g[f"pct_{metric}_before"]).dropna()
            rows.append({"position_group": pos, "metric": metric, "label": METRIC_LABELS[metric], "n": len(d),
                         "median_pct_change": d.median() if len(d) else np.nan,
                         "median_before": g[metric].median(), "median_after": g[f"{metric}_after"].median(),
                         "in_profile": metric in SCORE_METRICS[pos]})
    return pd.DataFrame(rows).round(3)


def output_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela da Fase 5 para o dashboard: uma linha por compra, as melhores do índice primeiro."""
    t = df.sort_values(["qualified", "value_index", "minutes_after"], ascending=[False, False, False]).copy()
    t["buying_league"] = t["league"].map(LEAGUE_NAMES)
    base = ["player_name", "club_name", "buying_league", "other_club_name", "position_group", "position", "age_at_transfer", "fee", "fee_known",
            "fair_price", "fee_vs_fair", "verdict", "status", "minutes_after", "min_minutes_required", "appearances_after", "starts_after",
            "team_matches_since_arrival", "minutes_before", "club_elsewhere", "ss_mixed_clubs", "score_before", "score_after", "score_change",
            "fee_pct_position", "value_index", "fee_per_score_point", "matches_with", "matches_without", "ppg_with", "ppg_without", "ppg_difference",
            "xgd_per_match_with", "xgd_per_match_without", "model_exclusion", "summer", "player_id", "club_id"]
    pairs = []
    for metric in COMPARE_METRICS:
        pairs += [metric, f"{metric}_after", f"pct_{metric}_before", f"pct_{metric}_after"]
    t = t[base + pairs].rename(columns={"club_name": "buying_club", "other_club_name": "origin_club", "age_at_transfer": "age",
                                         **{m: f"{m}_before" for m in COMPARE_METRICS}})
    num = t.select_dtypes("number").columns.difference(["fee", "fair_price", "fee_per_score_point", "summer", "player_id", "club_id"])
    t[num] = t[num].astype(float).round(3)
    t["fair_price"] = t["fair_price"].round(-4)
    return t.reset_index(drop=True)


# --------------------------------------------------------------------------- gráficos


def _sample_note(s: dict) -> str:
    prog = s["season"]
    lo, hi = int(prog["min_team_matches"].min()), int(prog["max_team_matches"].max())
    return f"2026/27 league matches only: {lo}–{hi} played per team (last result {prog['last_result'].max()}). Small sample."


def charts(df: pd.DataFrame, team: pd.DataFrame, changes: pd.DataFrame, s: dict, out: Path) -> None:
    note = _sample_note(s)
    rules = s["rules"]

    # 1. Minutos no clube novo, com o estado de cada compra.
    d = df.sort_values("minutes_after")
    fig = go.Figure()
    for status, label in STATUS_LABELS.items():
        g = d[d["status"] == status]
        fig.add_trace(go.Histogram(x=g["minutes_after"], xbins=dict(start=0, size=45), name=f"{label} ({len(g)})", marker=dict(color=STATUS_COLORS[status]),
                                   hovertemplate="%{x} min: %{y} purchases<extra>" + label + "</extra>"))
    fig.add_vline(x=rules["min_minutes_after"], line=dict(color=TEXT_SECONDARY, dash="dash", width=1.5),
                  annotation_text=f"minimum {rules['min_minutes_after']} min", annotation_position="top right")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="League minutes for the buying club, 2026/27")
    fig.update_yaxes(title="Purchases")
    style(fig, f"How much the summer-{SEASON} signings have played", f"n={len(df)} purchases. {note} "
          f"Minimum: {rules['min_minutes_after']} min or {rules['min_share_of_team_minutes']:.0%} of the team's minutes since arrival.")
    export(fig, out, "01_minutes_after")

    # 2. Pontuação do perfil antes vs. depois.
    q = df[df["qualified"] & df["score_before"].notna() & df["score_after"].notna()]
    fig = go.Figure()
    for pos in POSITIONS:
        g = q[q["position_group"] == pos]
        fig.add_trace(go.Scatter(
            x=g["score_before"], y=g["score_after"], mode="markers", name=f"{POSITION_LABELS[pos]} ({len(g)})",
            marker=dict(color=POSITION_COLORS[pos], size=9, opacity=0.8, line=dict(width=1.5, color="white")), text=g["player_name"],
            customdata=np.c_[g["club_name"], g["minutes_after"], g["minutes_before"]],
            hovertemplate="<b>%{text}</b> (%{customdata[0]})<br>Before %{x:.0f} · now %{y:.0f}<br>%{customdata[1]:.0f} min now, %{customdata[2]:.0f} the season before<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=MUTED, dash="dot", width=1.5), name="Same as before", hoverinfo="skip"))
    fig.update_xaxes(title="Profile score the season before (mean percentile)", range=[0, 100])
    fig.update_yaxes(title="Profile score in 2026/27 at the new club", range=[0, 100])
    style(fig, "Are the signings playing at their old level?", f"n={len(q)} with enough minutes now and some the season before. "
          f"Score = mean percentile of the position's profile metrics (100 = best). {note}")
    export(fig, out, "02_score_before_after")

    # 3. Índice custo/desempenho: percentil do preço vs. percentil de desempenho.
    v = df[df["value_index"].notna()]
    fig = go.Figure()
    for pos in POSITIONS:
        g = v[v["position_group"] == pos]
        fig.add_trace(go.Scatter(
            x=g["fee_pct_position"], y=g["score_after"], mode="markers", name=f"{POSITION_LABELS[pos]} ({len(g)})",
            marker=dict(color=POSITION_COLORS[pos], size=9, opacity=0.8, line=dict(width=1.5, color="white")), text=g["player_name"],
            customdata=np.c_[g["club_name"], g["fee"] / 1e6, g["value_index"], g["verdict"].fillna("not in the price model")],
            hovertemplate="<b>%{text}</b> (%{customdata[0]})<br>€%{customdata[1]:.1f}m: fee percentile %{x:.0f} · performance %{y:.0f}"
                          "<br>Index %{customdata[2]:+.0f} · phase 4: %{customdata[3]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=MUTED, dash="dot", width=1.5), name="Performing at price", hoverinfo="skip"))
    fig.update_xaxes(title="Fee percentile among summer-2026 purchases of the same position", range=[0, 100])
    fig.update_yaxes(title="Performance score in 2026/27 (mean percentile)", range=[0, 100])
    style(fig, "Cost against performance so far", f"n={len(v)} with a known fee and enough minutes. Above the line: performing above the price tier. {note}")
    export(fig, out, "03_cost_vs_performance")

    # 4. Ranking do índice.
    top = pd.concat([v.nlargest(12, "value_index"), v.nsmallest(12, "value_index")]).drop_duplicates(KEY).sort_values("value_index")
    fig = go.Figure(go.Bar(
        x=top["value_index"], y=top["player_name"] + " · " + top["club_name"], orientation="h",
        marker=dict(color=[DIVERGING[0][1] if x >= 0 else DIVERGING[-1][1] for x in top["value_index"]], cornerradius=4),
        customdata=np.c_[top["fee"] / 1e6, top["fee_pct_position"], top["score_after"], top["minutes_after"]],
        hovertemplate="%{y}<br>€%{customdata[0]:.1f}m (fee percentile %{customdata[1]:.0f}) · performance %{customdata[2]:.0f}"
                      "<br>%{customdata[3]:.0f} minutes<extra></extra>",
        text=[f"{x:+.0f}" for x in top["value_index"]], textposition="outside", textfont=dict(color=TEXT_SECONDARY)))
    fig.add_vline(x=0, line=dict(color=TEXT_SECONDARY, width=1))
    fig.update_xaxes(title="Cost/performance index (performance percentile − fee percentile)")
    fig.update_traces(cliponaxis=False)
    fig.update_layout(height=720, margin=dict(l=280, r=60))
    style(fig, "Best and worst value so far", f"12 highest and 12 lowest of n={len(v)}. Blue: performing above the price tier; red: below. {note}")
    export(fig, out, "04_value_ranking")

    # 5. Variação mediana dos percentis do perfil, por posição.
    c = changes[changes["in_profile"] & (changes["n"] > 0)]
    fig = go.Figure()
    for pos in POSITIONS:
        g = c[c["position_group"] == pos]
        fig.add_trace(go.Bar(x=g["median_pct_change"], y=g["label"], orientation="h", name=f"{POSITION_LABELS[pos]} (n≤{int(g['n'].max()) if len(g) else 0})",
                             marker=dict(color=POSITION_COLORS[pos], cornerradius=3), customdata=g["n"],
                             hovertemplate="%{y}: median change %{x:+.0f} percentile points (n=%{customdata})<extra>" + POSITION_LABELS[pos] + "</extra>"))
    fig.add_vline(x=0, line=dict(color=TEXT_SECONDARY, width=1))
    fig.update_xaxes(title="Median change in percentile, 2026/27 at the new club vs. the season before")
    fig.update_layout(barmode="group", height=760, margin=dict(l=220))
    style(fig, "Which parts of their game changed", f"Profile metrics per position; signings with enough minutes and some the season before. {note}")
    export(fig, out, "05_metric_changes")

    # 6. Resultados da equipa com e sem o jogador.
    t = team.merge(df[KEY + ["qualified", "fee"]], on=KEY)
    t = t[(t["matches_with"] > 0) & (t["matches_without"] > 0)].sort_values("ppg_difference")
    fig = go.Figure()
    for _, r in t.iterrows():
        fig.add_trace(go.Scatter(x=[r["ppg_without"], r["ppg_with"]], y=[f"{r['player_name']} · {r['club_name']}"] * 2, mode="lines",
                                 line=dict(color="#c9c8c3", width=2), showlegend=False, hoverinfo="skip"))
    labels = t["player_name"] + " · " + t["club_name"]
    fig.add_trace(go.Scatter(x=t["ppg_without"], y=labels, mode="markers", name="Without him", marker=dict(color=MUTED, size=10),
                             customdata=t["matches_without"], hovertemplate="%{y}<br>Without: %{x:.2f} points per match (%{customdata} matches)<extra></extra>"))
    fig.add_trace(go.Scatter(x=t["ppg_with"], y=labels, mode="markers", name="With him", marker=dict(color=ACCENT, size=11, line=dict(width=2, color="white")),
                             customdata=t["matches_with"], hovertemplate="%{y}<br>With: %{x:.2f} points per match (%{customdata} matches)<extra></extra>"))
    fig.update_xaxes(title="Team points per league match since his arrival", range=[-0.1, 3.1])
    fig.update_layout(height=max(420, 22 * len(t) + 160), margin=dict(l=280))
    style(fig, "Team results with and without each signing", f"n={len(t)} signings with at least one match on each side. With = played any minutes. "
          f"Very few matches per side: read as a description, not an effect. {note}")
    export(fig, out, "06_team_with_without")


# --------------------------------------------------------------------------- resumo


def summary_markdown(s: dict) -> str:
    df, table, team, rules, prog = s["df"], s["table"], s["team"], s["rules"], s["season"]
    counts = df["status"].value_counts()
    q = df[df["qualified"]]
    v = df[df["value_index"].notna()].sort_values("value_index", ascending=False)
    both = q[q["score_before"].notna() & q["score_after"].notna()]
    out_model = df[df["model_exclusion"].notna()]
    lines = [
        f"# Phase 5: performance at the new club (summary)",
        "",
        "Generated by `uv run main.py performance` from `data/processed` and the cached 2026/27 tables in `data/interim` (offline). "
        "It runs again after each weekly refresh. Charts are in this folder as `.html` (iframe fragment) and `.json` (Plotly).",
        "",
        "## Sample (very small: the season has just started)",
        "Matches played per team so far (league only):",
        "",
        "| League | Matches per team | Last result |",
        "|---|---|---|",
    ]
    for r in prog.itertuples():
        lines.append(f"| {LEAGUE_NAMES[r.lg]} | {r.min_team_matches}–{r.max_team_matches} | {r.last_result} |")
    lines += [
        "",
        f"- {len(df)} summer-{SEASON} purchases are followed (every top-5 to top-5 purchase, including the {len(out_model)} left out of the price model).",
        f"- Minimum sample: {rules['min_minutes_after']} league minutes for the buying club, or {rules['min_share_of_team_minutes']:.0%} of the team's league "
        "minutes since the player arrived, whichever is higher. The second rule takes over as the season goes on "
        f"(at 38 matches it asks for {rules['min_share_of_team_minutes'] * 90 * 38:.0f} minutes).",
        f"- {counts.get('qualified', 0)} have enough minutes, {counts.get('under_min_minutes', 0)} have played less, "
        f"{counts.get('no_minutes', 0)} have not played a league minute for the buying club, and {counts.get('playing_elsewhere', 0)} "
        "are playing for another club since the transfer (loaned back or out after the purchase).",
        "- Per 90 numbers on a few hundred minutes swing a lot with one goal or one good match. Treat every number here as provisional.",
        "",
        "## Before and after",
        "Each metric gets a percentile against the same reference as phase 3 (bought players of the same position with ≥900 league minutes the season before), "
        "so the season before and 2026/27 are on one scale. The profile score is the mean percentile of the position's profile metrics (the phase 3 radar).",
        "",
    ]
    if len(both):
        up = (both["score_change"] > 0).sum()
        lines.append(f"- {len(both)} signings have enough minutes now and some the season before: {up} score higher than before, {len(both) - up} lower; "
                     f"median change {both['score_change'].median():+.1f} points.")
        for label, g in [("Biggest rises", both.nlargest(5, "score_change")), ("Biggest drops", both.nsmallest(5, "score_change"))]:
            lines.append(f"- {label}: " + ", ".join(f"{r.player_name} ({r.club_name}, {r.score_before:.0f} → {r.score_after:.0f})" for r in g.itertuples()) + ".")
    lines += [
        "",
        "## Cost/performance index",
        "Index = performance score in 2026/27 minus the fee's percentile among summer-2026 purchases of the same position. "
        "Positive: performing above the price tier; negative: below. Needs a known fee and the minimum sample; the 17 unknown fees stay out of it. "
        "`fee_per_score_point` (fee ÷ score) and phase 4's verdict are in the table next to it.",
        "",
        "| # | Player | Buyer | Fee | Minutes | Score | Fee pct. | Index | Phase 4 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    def _row(i, r):
        verdict = r.verdict if isinstance(r.verdict, str) else "—"
        return f"| {i} | {r.player_name} | {r.club_name} | {_money(r.fee)} | {r.minutes_after:.0f} | {r.score_after:.0f} | {r.fee_pct_position:.0f} | {r.value_index:+.0f} | {verdict} |"
    for i, r in enumerate(v.head(10).itertuples(), 1):
        lines.append(_row(i, r))
    if len(v) > 10:
        lines += ["", "Lowest:", "", "| # | Player | Buyer | Fee | Minutes | Score | Fee pct. | Index | Phase 4 |", "|---|---|---|---|---|---|---|---|---|"]
        for i, r in list(enumerate(v.itertuples(), 1))[-10:][::-1]:
            lines.append(_row(i, r))
    if v["verdict"].notna().any():
        by = v.groupby("verdict")["value_index"].agg(["size", "median"])
        lines += ["", "Median index by phase 4 verdict: " + ", ".join(f"{k} {r['median']:+.0f} (n={int(r['size'])})" for k, r in by.iterrows()) + "."]
    t = team[(team["matches_with"] > 0) & (team["matches_without"] > 0)]
    lines += [
        "",
        "## Team results with and without the player",
        "Points, goal difference and xG difference per league match of the buying club since the transfer date, split by whether the player played. "
        f"{len(t)} signings have at least one match on each side so far; most have one or two, so this is a description, not an effect "
        "(the opponent, injuries and rotation all differ between the two groups).",
        "",
        "## Files",
        "- `performance_2026.csv` / `.json`: every purchase with status, minutes, before/after per 90 (`<metric>_before`, `<metric>_after`) and their percentiles, "
        "profile scores, the index and phase 4's fair price.",
        "- `team_results_2026.csv` / `.json`: points, goal and xG difference per match with and without the player.",
        "- `metric_changes.csv`: median change in percentile per metric and position.",
        "",
        "## Limits",
        "- Understat match data does not split out penalties, so xG and goals include them (the season before used the same totals here).",
        "- Sofascore only has season totals per league and team: players who played for another club in the same league earlier this season "
        f"({int(df['ss_mixed_clubs'].sum())}) have no Sofascore metrics for 2026/27, and their profile score uses the Understat metrics when at least half are there.",
        "- The reference is bought players' previous seasons, not the whole league; a player moving to a stronger or weaker team changes his numbers for reasons that are not his own.",
    ]
    return "\n".join(lines) + "\n"
