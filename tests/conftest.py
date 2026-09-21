import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The suite must run with no torch, no model download and no API key, so the
# deterministic backend is forced before healix.config is imported.
os.environ.setdefault("EMBEDDING_BACKEND", "hashing")

FIXTURE_DIR = ROOT / "data" / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir():
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def chunks(fixture_dir):
    from healix import corpus
    return corpus.build_chunks(fixture_dir)


@pytest.fixture(scope="session")
def built_index(chunks, tmp_path_factory):
    from healix import index
    directory = tmp_path_factory.mktemp("index")
    index.build(chunks, directory)
    return index.load(directory)


@pytest.fixture
def stub_llm():
    """A generate() stand-in that echoes a fixed grounded answer."""
    def _generate(messages):
        return ("Iron deficiency anemia develops when iron stores fall below the level "
                "required for erythropoiesis. Fatigue and pallor are typical. Treatment "
                "is oral ferrous sulfate.")
    return _generate
