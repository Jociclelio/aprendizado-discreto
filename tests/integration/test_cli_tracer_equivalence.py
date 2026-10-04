"""Testes de equivalência CLI/tracer — bit-identidade entre train_sample_online e trace_b1."""
import numpy as np
import pytest

from src.counter_rules import CounterManager
from src.playground.tracer import trace_b1, trace_b2, trace_b3
from src.ternary_b1 import train_sample_online
from src.ternary_net import TernaryNet
from src.utils import set_seed

LAYER_SHAPES = [(128, 784), (64, 128), (10, 64)]


class TestBitIdentity:
    """Verifica que trace_b1 é bit-idêntico a train_sample_online."""

    def test_bit_identity_1_sample(self):
        set_seed(42)
        net_ref = TernaryNet(seed=42)
        cm_ref = CounterManager(list(LAYER_SHAPES))
        set_seed(42)
        net_tr = TernaryNet(seed=42)
        cm_tr = CounterManager(list(LAYER_SHAPES))

        x = np.random.randn(784).astype(np.float32)
        y = 3

        m_ref = train_sample_online(net_ref, cm_ref, x, y)
        tr = trace_b1(net_tr, cm_tr, x, y)
        m_tr = tr["metrics"]

        for l in range(3):
            assert np.array_equal(net_ref.layers[l].weights, net_tr.layers[l].weights)
            assert np.array_equal(net_ref.layers[l].bias, net_tr.layers[l].bias)
            assert np.array_equal(cm_ref.weight_counters[l], cm_tr.weight_counters[l])
            assert np.array_equal(cm_ref.bias_counters[l], cm_tr.bias_counters[l])

        for k in ("loss", "selected_weights", "blocked_weights", "transitions",
                  "bias_selected", "bias_blocked", "bias_transitions"):
            assert m_ref[k] == m_tr[k]

    def test_bit_identity_5_samples(self):
        rng = np.random.RandomState(42)
        set_seed(42)
        net_ref = TernaryNet(seed=42)
        cm_ref = CounterManager(list(LAYER_SHAPES))
        set_seed(42)
        net_tr = TernaryNet(seed=42)
        cm_tr = CounterManager(list(LAYER_SHAPES))

        x_seq = [rng.randn(784).astype(np.float32) for _ in range(5)]
        y_seq = [int(rng.randint(0, 10)) for _ in range(5)]

        for i in range(4):
            train_sample_online(net_ref, cm_ref, x_seq[i], y_seq[i])
            trace_b1(net_tr, cm_tr, x_seq[i], y_seq[i])

        m_ref = train_sample_online(net_ref, cm_ref, x_seq[4], y_seq[4])
        tr = trace_b1(net_tr, cm_tr, x_seq[4], y_seq[4])
        m_tr = tr["metrics"]

        for l in range(3):
            assert np.array_equal(net_ref.layers[l].weights, net_tr.layers[l].weights)
            assert np.array_equal(net_ref.layers[l].bias, net_tr.layers[l].bias)
            assert np.array_equal(cm_ref.weight_counters[l], cm_tr.weight_counters[l])
            assert np.array_equal(cm_ref.bias_counters[l], cm_tr.bias_counters[l])

        for k in ("loss", "selected_weights", "blocked_weights", "transitions",
                  "bias_selected", "bias_blocked", "bias_transitions"):
            assert m_ref[k] == m_tr[k]

    def test_bit_identity_com_adaptive_radius(self):
        set_seed(42)
        net_ref = TernaryNet(seed=42)
        cm_ref = CounterManager(list(LAYER_SHAPES))
        set_seed(42)
        net_tr = TernaryNet(seed=42)
        cm_tr = CounterManager(list(LAYER_SHAPES))

        x = np.random.randn(784).astype(np.float32)
        y = 5

        m_ref = train_sample_online(net_ref, cm_ref, x, y, adaptive_radius=True, scale_factor=1.0)
        tr = trace_b1(net_tr, cm_tr, x, y, adaptive_radius=True, scale_factor=1.0)
        m_tr = tr["metrics"]

        for l in range(3):
            assert np.array_equal(net_ref.layers[l].weights, net_tr.layers[l].weights)
            assert np.array_equal(net_ref.layers[l].bias, net_tr.layers[l].bias)
            assert np.array_equal(cm_ref.weight_counters[l], cm_tr.weight_counters[l])
            assert np.array_equal(cm_ref.bias_counters[l], cm_tr.bias_counters[l])

        for k in ("loss", "selected_weights", "blocked_weights", "transitions",
                  "bias_selected", "bias_blocked", "bias_transitions"):
            assert m_ref[k] == m_tr[k]


class TestTracerB2B3:
    """Verifica que B2/B3 produzem seleções válidas."""

    def test_b2_selecao(self):
        set_seed(7)
        net = TernaryNet(seed=7)
        cm = CounterManager(list(LAYER_SHAPES))
        x = np.random.randn(784).astype(np.float32)
        tr = trace_b2(net, cm, x, 3)
        sel = tr["metrics"]["selected_weights"]
        assert sel > 0

    def test_b3_selecao(self):
        set_seed(7)
        net = TernaryNet(seed=7)
        cm = CounterManager(list(LAYER_SHAPES))
        x = np.random.randn(784).astype(np.float32)
        tr = trace_b3(net, cm, x, 3)
        sel = tr["metrics"]["selected_weights"]
        assert sel > 0

    def test_b1_exposicao_completa(self):
        set_seed(7)
        net = TernaryNet(seed=7)
        cm = CounterManager(list(LAYER_SHAPES))
        tr = trace_b1(net, cm, np.random.randn(784).astype(np.float32), 3)
        for key in ("z_list", "fake_cont", "q_fake", "deltas", "dw_float",
                    "dw_int", "Cw_before", "Cw_acc", "Cw_after", "masks_w",
                    "W_after", "metrics"):
            assert key in tr
