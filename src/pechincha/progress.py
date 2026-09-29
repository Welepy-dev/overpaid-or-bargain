"""Barras de progresso com estimativa do tempo restante.

A estimativa vem do ritmo real (páginas em cache passam logo; as novas esperam
o intervalo entre pedidos), por isso fica mais fiável ao fim de alguns itens.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress(items: Iterable[T], desc: str, total: int | None = None) -> Iterable[T]:
    return tqdm(items, desc=desc, total=total, unit="it", dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}]")
