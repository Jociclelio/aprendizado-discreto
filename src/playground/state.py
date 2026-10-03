"""
[017] Estado da sessao Streamlit — net+cm + cache do MNIST.

Toda view exibe `seed`, hiperparametros e indice da amostra; o botao
Reset restaura net+cm a partir da seed (reprodutibilidade).
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.counter_rules import CounterManager
from src.playground.tracer import LAYER_SHAPES, TRACE_FNS
from src.ternary_net import TernaryNet

DEFAULTS = {
    "variant": "B1",
    "scale": 1.0,
    "adaptive_radius": True,
    "scale_factor": 1.0,
    "radius": 1.0,
    "levels": 5,
    "seed": 42,
    "topk_frac": 0.01,
    "batch_size": 32,
    "sample_idx": 0,
}


@st.cache_data(show_spinner="Carregando MNIST...")
def get_mnist():
    """Carrega MNIST normalizado [0,1] com cache (1x por sessao)."""
    from src.utils import load_mnist

    train_data, train_labels, test_data, test_labels = load_mnist(normalize=True)
    return train_data, train_labels, test_data, test_labels


def make_net_cm(seed: int, decay_rate=None, decay_shift=None):
    """Cria (net, cm) frescos a partir da seed. Arquitetura 784->128->64->10."""
    net = TernaryNet(seed=seed)
    cm = CounterManager(
        list(LAYER_SHAPES), decay_rate=decay_rate, decay_shift=decay_shift
    )
    return net, cm


def ensure_state(params: dict) -> None:
    """Garante net+cm no session_state; recria se a seed mudou."""
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    if "net" not in st.session_state or st.session_state.get("seed_current") != params["seed"]:
        net, cm = make_net_cm(params["seed"])
        st.session_state["net"] = net
        st.session_state["cm"] = cm
        st.session_state["seed_current"] = params["seed"]
        st.session_state["steps_done"] = 0
        st.session_state.pop("trace", None)


def reset_net(seed: int) -> None:
    """Botao Reset: restaura net+cm da seed (descarta treino interativo)."""
    net, cm = make_net_cm(seed)
    st.session_state["net"] = net
    st.session_state["cm"] = cm
    st.session_state["seed_current"] = seed
    st.session_state["steps_done"] = 0
    st.session_state.pop("trace", None)


def trace_fn_for(variant: str):
    """Retorna trace_b1/trace_b2/trace_b3 para a variante."""
    return TRACE_FNS[variant]


def config_line(params: dict, idx: int) -> str:
    """Linha de proveniencia exibida em toda view: seed + hiperparams + amostra."""
    r = f"c={params['scale_factor']}" if params["adaptive_radius"] else f"r={params['radius']}"
    extra = f" topk={params['topk_frac']}" if params["variant"] in ("B2", "B3") else ""
    return (
        f"variante={params['variant']} scale={params['scale']} {r} "
        f"levels={params['levels']} seed={params['seed']}{extra} amostra={idx}"
    )


def clamp_idx(idx: int, n: int) -> int:
    """Limita o indice ao intervalo [0, n-1]."""
    return max(0, min(int(idx), n - 1))
