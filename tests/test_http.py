import pytest

from pechincha import http


class FakeResponse:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


class FakeSession:
    def __init__(self, response):
        self.response, self.calls, self.headers = response, 0, {}

    def get(self, url, timeout):
        self.calls += 1
        return self.response


def test_waf_page_stops_without_retrying(tmp_path, monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    session = FakeSession(FakeResponse(405, "<title>Human Verification</title>"))
    f = http.CachedFetcher(tmp_path, "tm", min_interval=0, session=session)
    with pytest.raises(http.BlockedError):
        f.get_text("https://www.transfermarkt.com/x")
    assert session.calls == 1


def test_other_errors_are_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    session = FakeSession(FakeResponse(503, "busy"))
    f = http.CachedFetcher(tmp_path, "tm", min_interval=0, max_retries=3, session=session)
    with pytest.raises(http.FetchError) as exc:
        f.get_text("https://www.transfermarkt.com/x")
    assert not isinstance(exc.value, http.BlockedError) and session.calls == 3
