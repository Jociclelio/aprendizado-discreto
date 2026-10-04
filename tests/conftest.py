"""Fixtures compartilhadas para testes pytest."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_net():
    """Rede mínima para testes determinísticos."""
    from src.ternary_net import TernaryNet
    return TernaryNet(input_dim=4, hidden1=4, hidden2=4, output_dim=2, seed=42)


@pytest.fixture
def small_net():
    """Rede pequena para testes de integração."""
    from src.ternary_net import TernaryNet
    return TernaryNet(input_dim=8, hidden1=6, hidden2=4, output_dim=3, seed=7)


@pytest.fixture
def sample_input():
    """Entrada sintética em [0, 1]."""
    return np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float32)
