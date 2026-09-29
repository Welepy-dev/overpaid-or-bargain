"""Transfermarkt: lista de transferências, histórico de valor de mercado e perfil.

Três tipos de página:

1. Transferências de uma liga numa janela (HTML):
   ``/{slug}/transfers/wettbewerb/{code}/plus/?saison_id=...&s_w=s&leihe=1&intern=0``
   Uma caixa por clube com a tabela de entradas ("In") e de saídas ("Out").
2. Histórico de transferências e de valor de mercado de um jogador (JSON ``ceapi``).
3. Perfil do jogador (HTML): data de nascimento, posição, pé, altura, contrato.

As funções ``parse_*`` só recebem texto e são testadas com fixtures em ``tests/``;
as funções ``fetch_*`` tratam da rede e da cache.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone

import pandas as pd
from bs4 import BeautifulSoup

from ..config import Config, League
from ..http import CachedFetcher, FetchError

log = logging.getLogger(__name__)

BASE = "https://www.transfermarkt.com"

TRANSFER_TYPES = ("fee", "loan", "loan_return", "free", "unknown")


# --------------------------------------------------------------------------- #
# Valores e tipos de transferência
# --------------------------------------------------------------------------- #

_MONEY = re.compile(r"(?P<cur>[€£$])\s*(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>bn|m|k|th\.?)?", re.I)
_MULT = {"bn": 1e9, "m": 1e6, "k": 1e3, "th": 1e3, "th.": 1e3, None: 1.0}
_CURRENCY = {"€": "EUR", "£": "GBP", "$": "USD"}


def parse_money(text: str | None) -> tuple[float | None, str | None]:
    """'€45.00m' -> (45_000_000.0, 'EUR'); '-' ou '?' -> (None, None)."""
    if not text:
        return None, None
    m = _MONEY.search(text.replace("\xa0", " "))
    if not m:
        return None, None
    unit = m.group("unit").lower() if m.group("unit") else None
    value = float(m.group("num").replace(",", ".")) * _MULT[unit]
    return value, _CURRENCY[m.group("cur")]


def classify_fee(text: str) -> dict:
    """Classifica o texto da coluna 'Fee' do Transfermarkt.

    Devolve ``transfer_type`` (um de TRANSFER_TYPES), ``fee`` e ``fee_currency``.
    Empréstimos com obrigação de compra aparecem no Transfermarkt como
    empréstimo nesta janela (e como compra só quando a obrigação é ativada),
    por isso ficam excluídos pela regra dos empréstimos.
    """
    t = " ".join((text or "").split()).lower()
    fee, cur = parse_money(text)
    if "end of loan" in t or "loan return" in t or "fim de empréstimo" in t:
        kind = "loan_return"
        fee, cur = None, None
    elif "loan" in t:
        kind = "loan"  # 'loan transfer' ou 'Loan fee: €3.00m' (fee = taxa de empréstimo)
    elif "free" in t:
        kind, fee, cur = "free", 0.0, None
    elif fee is not None:
        kind = "fee"
    else:
        kind = "unknown"  # '?', '-', 'draft', vazio
    return {"transfer_type": kind, "fee": fee, "fee_currency": cur, "fee_raw": " ".join((text or "").split())}


# --------------------------------------------------------------------------- #
# 1. Transferências de uma liga
# --------------------------------------------------------------------------- #

_PLAYER_ID = re.compile(r"/spieler/(\d+)")
_CLUB_ID = re.compile(r"/verein/(\d+)")
_TRANSFER_ID = re.compile(r"/transfer_id/(\d+)")


def league_transfers_url(league: League, season: int, window: str = "s") -> str:
    return (
        f"{BASE}/{league.tm_slug}/transfers/wettbewerb/{league.tm_code}/plus/"
        f"?saison_id={season}&s_w={window}&leihe=1&intern=0"
    )


def parse_league_transfers(html: str, league_key: str, season: int, window: str = "s") -> pd.DataFrame:
    """Uma linha por movimento (entrada e saída) de cada clube da liga."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for box in soup.select("div.box"):
        club_link = _club_header_link(box)
        if club_link is None:
            continue
        club_name = club_link.get("title") or club_link.get_text(strip=True)
        club_id = _int(_CLUB_ID, club_link.get("href"))
        for table in box.find_all("table"):
            direction = _table_direction(table)
            if direction is None:
                continue
            body = table.find("tbody") or table
            for tr in body.find_all("tr", recursive=False):
                rec = _parse_transfer_row(tr)
                if rec is None:
                    continue
                rec.update(
                    league=league_key,
                    season=season,
                    window="summer" if window == "s" else "winter",
                    direction=direction,
                    club_id=club_id,
                    club_name=club_name,
                )
                rows.append(rec)
    df = pd.DataFrame(rows, columns=LEAGUE_TRANSFER_COLUMNS)
    return df


LEAGUE_TRANSFER_COLUMNS = [
    "league", "season", "window", "direction", "club_id", "club_name",
    "player_id", "player_name", "age", "nationality", "position", "position_short",
    "market_value_listed", "other_club_id", "other_club_name", "other_club_country",
    "transfer_type", "fee", "fee_currency", "fee_raw", "tm_transfer_id",
]


def _club_header_link(box):
    header = box.find(["h2", "h3"])
    if header is None:
        return None
    links = [a for a in header.find_all("a", href=True) if _CLUB_ID.search(a["href"])]
    # O primeiro link costuma ser o emblema (sem texto); preferir o que tem texto.
    for a in links:
        if a.get_text(strip=True):
            return a
    return links[0] if links else None


def _table_direction(table) -> str | None:
    th = table.find("th")
    if th is None:
        return None
    label = th.get_text(" ", strip=True).lower()
    if label in ("in", "arrivals", "zugänge"):
        return "in"
    if label in ("out", "departures", "abgänge"):
        return "out"
    return None


def _parse_transfer_row(tr) -> dict | None:
    tds = tr.find_all("td", recursive=False)
    player_a = tr.find("a", href=re.compile(r"/profil/spieler/\d+"))
    if player_a is None or len(tds) < 5:
        return None  # linha "No arrivals"/"No departures" ou separador

    def cell(cls: str):
        return tr.find("td", class_=cls)

    other_td = cell("verein-flagge-transfer-cell")
    other_a = None
    if other_td is not None:
        other_a = next((a for a in other_td.find_all("a", href=True) if a.get_text(strip=True)), None)
        other_a = other_a or other_td.find("a", href=True)
    country = None
    for td in [other_td, other_td.find_next_sibling("td") if other_td is not None else None]:
        img = td.find("img", class_="flaggenrahmen") if td is not None else None
        if img is not None and img.get("title"):
            country = img["title"]
            break

    nat_td = cell("nat-transfer-cell")
    nat_img = nat_td.find("img") if nat_td is not None else None

    fee_td = tds[-1]
    fee_a = fee_td.find("a", href=True)
    rec = {
        "player_id": _int(_PLAYER_ID, player_a["href"]),
        "player_name": player_a.get("title") or player_a.get_text(strip=True),
        "age": _to_int(_text(cell("alter-transfer-cell"))),
        "nationality": nat_img.get("title") if nat_img is not None else None,
        "position": _text(cell("pos-transfer-cell")),
        "position_short": _text(cell("kurzpos-transfer-cell")),
        "market_value_listed": parse_money(_text(cell("mw-transfer-cell")))[0],
        "other_club_id": _int(_CLUB_ID, other_a["href"]) if other_a is not None else None,
        "other_club_name": (other_a.get("title") or other_a.get_text(strip=True)) if other_a is not None else None,
        "other_club_country": country,
        "tm_transfer_id": _int(_TRANSFER_ID, fee_a["href"]) if fee_a is not None else None,
    }
    rec.update(classify_fee(fee_td.get_text(" ", strip=True)))
    return rec


# --------------------------------------------------------------------------- #
# 2. Histórico do jogador (ceapi, JSON)
# --------------------------------------------------------------------------- #

def transfer_history_url(player_id: int) -> str:
    return f"{BASE}/ceapi/transferHistory/list/{player_id}"


def market_value_url(player_id: int) -> str:
    return f"{BASE}/ceapi/marketValueDevelopment/graph/{player_id}"


def parse_transfer_history(data: dict, player_id: int) -> pd.DataFrame:
    rows = []
    for t in data.get("transfers", []) or []:
        fee_info = classify_fee(t.get("fee") or "")
        rows.append(
            {
                "player_id": player_id,
                "date": _parse_date(t.get("dateUnformatted") or t.get("date")),
                "season_label": t.get("season"),
                "from_club_id": _int(_CLUB_ID, (t.get("from") or {}).get("href")),
                "from_club_name": (t.get("from") or {}).get("clubName"),
                "to_club_id": _int(_CLUB_ID, (t.get("to") or {}).get("href")),
                "to_club_name": (t.get("to") or {}).get("clubName"),
                "market_value_at_transfer": parse_money(t.get("marketValue"))[0],
                "upcoming": bool(t.get("upcoming", False)),
                **fee_info,
            }
        )
    return pd.DataFrame(rows)


def parse_market_values(data: dict, player_id: int) -> pd.DataFrame:
    rows = []
    for p in data.get("list", []) or []:
        when = None
        if p.get("x") is not None:
            when = datetime.fromtimestamp(int(p["x"]) / 1000, tz=timezone.utc).date()
        elif p.get("datum_mw"):
            when = _parse_date(p["datum_mw"])
        rows.append(
            {
                "player_id": player_id,
                "date": when,
                "market_value": float(p["y"]) if p.get("y") is not None else parse_money(p.get("mw"))[0],
                "club_name": p.get("verein"),
            }
        )
    return pd.DataFrame(rows, columns=["player_id", "date", "market_value", "club_name"])


def market_value_before(values: pd.DataFrame, when: date | None) -> tuple[float | None, date | None]:
    """Último valor de mercado estritamente anterior à data da transferência."""
    if values.empty or when is None:
        return None, None
    prior = values[values["date"] < when].sort_values("date")
    if prior.empty:
        return None, None
    last = prior.iloc[-1]
    return last["market_value"], last["date"]


# --------------------------------------------------------------------------- #
# 3. Perfil do jogador
# --------------------------------------------------------------------------- #

def profile_url(player_id: int) -> str:
    return f"{BASE}/-/profil/spieler/{player_id}"


_PROFILE_LABELS = {
    "date of birth/age:": "date_of_birth",
    "date of birth:": "date_of_birth",
    "place of birth:": "place_of_birth",
    "height:": "height",
    "position:": "position_detail",
    "foot:": "foot",
    "joined:": "joined",
    "contract expires:": "contract_expires",
    "citizenship:": "citizenship",
}


def parse_profile(html: str, player_id: int) -> dict:
    soup = BeautifulSoup(html, "lxml")
    out: dict = {"player_id": player_id}
    for label in soup.select("span.info-table__content--regular"):
        key = _PROFILE_LABELS.get(label.get_text(" ", strip=True).lower())
        value = label.find_next_sibling("span")
        if key and value is not None:
            out[key] = " ".join(value.get_text(" ", strip=True).split())
    birth = soup.find(attrs={"itemprop": "birthDate"})
    if birth is not None:
        out["date_of_birth"] = birth.get_text(" ", strip=True)
    if "date_of_birth" in out:
        out["date_of_birth"] = _parse_date(re.sub(r"\(\d+\)", "", out["date_of_birth"]).strip())
    for k in ("contract_expires", "joined"):
        if k in out:
            out[k] = _parse_date(out[k])
    if "height" in out:
        m = re.search(r"(\d)[,.](\d{2})", out["height"])
        out["height_cm"] = int(m.group(1)) * 100 + int(m.group(2)) if m else None
        del out["height"]
    return out


# --------------------------------------------------------------------------- #
# Recolha
# --------------------------------------------------------------------------- #

def fetcher(cfg: Config) -> CachedFetcher:
    h = cfg.http
    return CachedFetcher(
        cfg.cache_dir, "transfermarkt",
        min_interval=h["min_interval_seconds"], timeout=h["timeout_seconds"], max_retries=h["max_retries"],
    )


def fetch_league_transfers(cfg: Config, seasons: list[int], f: CachedFetcher | None = None) -> pd.DataFrame:
    f = f or fetcher(cfg)
    frames = []
    for season in seasons:
        live = cfg.http["live_max_age_hours"] if season >= cfg.season else None
        for league in cfg.leagues:
            url = league_transfers_url(league, season)
            log.info("Transfermarkt %s %s", league.key, season)
            df = parse_league_transfers(f.get_text(url, max_age_hours=live), league.key, season)
            if df.empty:
                log.warning("Nenhuma transferência lida em %s (página mudou?)", url)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LEAGUE_TRANSFER_COLUMNS)


def fetch_player_details(cfg: Config, player_ids: list[int], f: CachedFetcher | None = None):
    """Histórico de transferências, valores de mercado e perfil de cada jogador."""
    f = f or fetcher(cfg)
    live = cfg.http["live_max_age_hours"]
    hist, values, profiles = [], [], []
    for i, pid in enumerate(sorted(set(int(p) for p in player_ids)), 1):
        if i % 50 == 0:
            log.info("Transfermarkt jogadores: %d/%d", i, len(player_ids))
        try:
            hist.append(parse_transfer_history(f.get_json(transfer_history_url(pid), max_age_hours=live), pid))
            values.append(parse_market_values(f.get_json(market_value_url(pid), max_age_hours=live), pid))
            profiles.append(parse_profile(f.get_text(profile_url(pid), max_age_hours=live), pid))
        except (FetchError, ValueError) as exc:
            log.warning("Jogador %s sem detalhes: %s", pid, exc)
    return (
        pd.concat(hist, ignore_index=True) if hist else pd.DataFrame(),
        pd.concat(values, ignore_index=True) if values else pd.DataFrame(),
        pd.DataFrame(profiles),
    )


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #

def _text(node) -> str | None:
    if node is None:
        return None
    t = " ".join(node.get_text(" ", strip=True).split())
    return t or None


def _int(pattern: re.Pattern, s: str | None) -> int | None:
    if not s:
        return None
    m = pattern.search(s)
    return int(m.group(1)) if m else None


def _to_int(s: str | None) -> int | None:
    return int(s) if s and s.strip().isdigit() else None


_DATE_FORMATS = ("%Y-%m-%d", "%b %d, %Y", "%d/%m/%Y", "%d.%m.%Y", "%B %d, %Y")


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None
