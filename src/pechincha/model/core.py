"""Regressão ridge sobre log(preço), só com o numpy, e validação por verão.

Separado dos gráficos para ser testado sozinho. As features vêm de uma lista
explícita (``fair_price.FEATURES``), não das 149 colunas da tabela por compra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Ridge:
    """Ridge com features padronizadas; a constante não é penalizada.

    Valores em falta são preenchidos com a mediana do treino (e a coluna
    ``<feature>_missing`` diz quando, se existir na lista de features).
    """

    alpha: float = 10.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Ridge":
        self.columns = list(X.columns)
        self.median = X.median()
        Xf = X.fillna(self.median)
        self.mean = Xf.mean()
        self.std = Xf.std(ddof=0).replace(0, 1.0)
        Z = ((Xf - self.mean) / self.std).to_numpy(float)
        yv = y.to_numpy(float)
        self.intercept = yv.mean()
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef_std = np.linalg.solve(A, Z.T @ (yv - self.intercept))
        return self

    def standardized(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X[self.columns].fillna(self.median) - self.mean) / self.std

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Parte de cada feature na previsão (em log), relativa ao comprador médio do treino."""
        return self.standardized(X) * self.coef_std

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.contributions(X).sum(axis=1).to_numpy()

    @property
    def coefficients(self) -> pd.Series:
        """Efeito em log(preço) de mais um desvio-padrão da feature."""
        return pd.Series(self.coef_std, index=self.columns)


def summer_folds(groups: pd.Series):
    """Deixa um verão de fora de cada vez (o modelo nunca vê o verão que prevê)."""
    for g in sorted(groups.unique()):
        yield g, (groups != g).to_numpy(), (groups == g).to_numpy()


def out_of_fold(X: pd.DataFrame, y: pd.Series, groups: pd.Series, alpha: float) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for _, train, test in summer_folds(groups):
        pred[test] = Ridge(alpha).fit(X[train], y[train]).predict(X[test])
    return pred


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    """Erros em log e em euros: RMSE e R² em log; erro mediano em % do preço."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    ratio = np.exp(np.abs(err)) - 1
    return {"n": len(y), "rmse_log": float(np.sqrt(np.mean(err ** 2))), "r2_log": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
            "median_abs_pct_error": float(np.median(ratio) * 100), "within_25pct": float(np.mean(ratio <= 0.25) * 100)}


def choose_alpha(X: pd.DataFrame, y: pd.Series, groups: pd.Series, grid: list[float]) -> tuple[float, pd.DataFrame]:
    """Alpha com o menor RMSE fora da amostra (validação por verão)."""
    rows = [{"alpha": a, **scores(y, out_of_fold(X, y, groups, a))} for a in grid]
    table = pd.DataFrame(rows)
    return float(table.loc[table["rmse_log"].idxmin(), "alpha"]), table
