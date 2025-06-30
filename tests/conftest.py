import sys, pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]   
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))    

from app.deps import get_rag, get_rec    

import pytest

@pytest.fixture(scope="session")
def rag_service():
    return get_rag()

@pytest.fixture(scope="session")
def rec_service():
    return get_rec()