import json
from pathlib import Path

import pytest

from pechincha.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_json(name: str):
    return json.loads(fixture_text(name))


@pytest.fixture
def cfg(tmp_path):
    """Config real, mas com a raiz num diretório temporário (cache e saídas isoladas)."""
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.yaml").write_text((root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    return load_config(tmp_path / "config.yaml")
