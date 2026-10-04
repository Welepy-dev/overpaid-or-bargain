"""Cliente HTTP com cache em disco e intervalo entre pedidos.

Cada resposta é guardada em ``data/raw/<fonte>/`` com o nome derivado do URL,
para não repetir pedidos e para o pipeline continuar a correr se uma página
mudar ou ficar indisponível (usa-se a última cópia guardada).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)


class FetchError(RuntimeError):
    pass


class BlockedError(FetchError):
    """O site respondeu com uma página anti-bot: insistir só prolonga o bloqueio."""


def _blocked(status: int, body: str) -> bool:
    return status == 429 or (status in (403, 405) and "Human Verification" in body)


class CachedFetcher:
    def __init__(
        self,
        cache_dir: Path,
        source: str,
        min_interval: float = 3.0,
        timeout: float = 30,
        max_retries: int = 3,
        headers: dict | None = None,
        session: requests.Session | None = None,
    ):
        self.dir = Path(cache_dir) / source
        self.dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-GB,en;q=0.9"})
        if headers:
            self.session.headers.update(headers)
        self._last_request: dict[str, float] = {}

    def cache_path(self, url: str, suffix: str) -> Path:
        parsed = urlparse(url)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", parsed.path + "_" + parsed.query).strip("_")[-80:]
        digest = hashlib.sha1(url.encode()).hexdigest()[:10]
        return self.dir / f"{slug}_{digest}{suffix}"

    def get_text(self, url: str, max_age_hours: float | None = None, suffix: str = ".html") -> str:
        """Devolve o corpo da resposta, da cache se existir e não tiver expirado.

        ``max_age_hours=None`` significa que a cópia em cache nunca expira.
        Se o pedido falhar e houver cópia antiga, usa-se a cópia antiga.
        """
        path = self.cache_path(url, suffix)
        if path.exists() and not _expired(path, max_age_hours):
            return path.read_text(encoding="utf-8")
        try:
            body = self._download(url)
        except FetchError:
            if path.exists():
                log.warning("Falha ao descarregar %s; a usar cópia em cache de %s", url, _mtime(path))
                return path.read_text(encoding="utf-8")
            raise
        path.write_text(body, encoding="utf-8")
        return body

    def get_json(self, url: str, max_age_hours: float | None = None):
        return json.loads(self.get_text(url, max_age_hours=max_age_hours, suffix=".json"))

    def close(self) -> None:
        self.session.close()

    def _download(self, url: str) -> str:
        host = urlparse(url).netloc
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_request.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last_request[host] = time.monotonic()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if resp.status_code == 200:
                    if "charset" not in resp.headers.get("content-type", "").lower():
                        # Sem charset no cabeçalho o requests assume latin-1 (kassiesa: "TÃ¼rkiye").
                        resp.encoding = "utf-8"
                    return resp.text
                if resp.status_code == 404:
                    raise FetchError(f"404 em {url}")
                if _blocked(resp.status_code, resp.text):
                    raise BlockedError(f"HTTP {resp.status_code} (bloqueio anti-bot) em {url}")
                last_error = FetchError(f"HTTP {resp.status_code} em {url}")
            log.info("Tentativa %d/%d falhou para %s: %s", attempt, self.max_retries, url, last_error)
            time.sleep(self.min_interval * 2**attempt)
        raise FetchError(str(last_error))


def find_browser() -> str | None:
    """Chromium/Chrome para o seleniumbase: ``PECHINCHA_BROWSER``, o PATH ou o do Playwright."""
    if os.environ.get("PECHINCHA_BROWSER"):
        return os.environ["PECHINCHA_BROWSER"]
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        if found := shutil.which(name):
            return found
    playwright = sorted(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-*/chrome"))
    return str(playwright[-1]) if playwright else None


def headless() -> bool:
    return os.environ.get("PECHINCHA_HEADLESS", "1") != "0"


class BrowserFetcher(CachedFetcher):
    """Igual ao CachedFetcher, mas descarrega com ``fetch()`` dentro de um Chromium.

    Para APIs que recusam clientes que não são browsers (ex.: Sofascore devolve
    403 a requests/curl). Abre ``home`` uma vez para ter os cookies do site. O
    browser vem de ``find_browser``; ``PECHINCHA_HEADLESS=0`` abre uma janela visível.
    """

    _SCRIPT = (
        "const done = arguments[arguments.length - 1];"
        "fetch(arguments[0])"
        ".then(r => r.text().then(t => done([r.status, t])))"
        ".catch(e => done([0, String(e)]));"
    )

    def __init__(self, cache_dir: Path, source: str, home: str, **kwargs):
        super().__init__(cache_dir, source, **kwargs)
        self.home = home
        self._driver = None

    def _browser(self):
        if self._driver is None:
            import seleniumbase as sb

            self._driver = sb.Driver(
                uc=True,
                headless=headless(),
                binary_location=find_browser(),
            )
            self._driver.set_script_timeout(self.timeout)
            self._driver.get(self.home)
        return self._driver

    def _download(self, url: str) -> str:
        host = urlparse(url).netloc
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_request.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last_request[host] = time.monotonic()
            try:
                status, body = self._browser().execute_async_script(self._SCRIPT, url)
            except Exception as exc:  # o browser pode ter morrido: abre-se outro
                last_error = exc
                self.close()
            else:
                if status == 200:
                    return body
                if status == 404:
                    raise FetchError(f"404 em {url}")
                if _blocked(status, body):
                    raise BlockedError(f"HTTP {status} (bloqueio anti-bot) em {url}")
                last_error = FetchError(f"HTTP {status} em {url}")
            log.info("Tentativa %d/%d falhou para %s: %s", attempt, self.max_retries, url, last_error)
            time.sleep(self.min_interval * 2**attempt)
        raise FetchError(str(last_error))

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None


def _expired(path: Path, max_age_hours: float | None) -> bool:
    if max_age_hours is None:
        return False
    return (time.time() - path.stat().st_mtime) > max_age_hours * 3600


def _mtime(path: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
