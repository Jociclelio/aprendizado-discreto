"""Testes unitários para src/counter_rules.py."""
import numpy as np
import pytest

from src.counter_rules import (
    CounterManager,
    apply_update_neuron,
    apply_update_layer_topk,
    apply_update_network_topk,
    compute_counter_delta,
    compute_counter_delta_single,
    update_weight,
)
from src.ternary_net import TernaryLayer, TernaryNet


class TestUpdateWeight:
    def test_incremento_positivo(self):
        assert update_weight(np.int8(0), 1) == 1

    def test_incremento_negativo(self):
        assert update_weight(np.int8(0), -1) == -1

    def test_clipping_superior(self):
        assert update_weight(np.int8(1), 1) == 1

    def test_clipping_inferior(self):
        assert update_weight(np.int8(-1), -1) == -1

    def test_transicao_para_zero(self):
        assert update_weight(np.int8(1), -1) == 0
        assert update_weight(np.int8(-1), 1) == 0


class TestCounterManager:
    def test_inicializacao(self):
        shapes = [(128, 784), (64, 128), (10, 64)]
        cm = CounterManager(shapes)
        assert cm.num_layers == 3
        assert cm.weight_counters[0].shape == (128, 784)
        assert cm.weight_counters[0].dtype == np.int64
        assert cm.bias_counters[0].shape == (128,)

    def test_accumulate(self):
        cm = CounterManager([(128, 784)])
        delta = np.ones((128, 784), dtype=np.int64)
        cm.accumulate(0, delta)
        assert cm.weight_counters[0][0, 0] == 1

    def test_get_neuron_counters(self):
        cm = CounterManager([(128, 784)])
        cm.accumulate(0, np.ones((128, 784), dtype=np.int64))
        neuron_C = cm.get_neuron_counters(0, 0)
        assert neuron_C.shape == (784,)
        assert neuron_C[0] == 1

    def test_get_top_k_weights(self):
        cm = CounterManager([(128, 784)])
        cm.accumulate(0, np.ones((128, 784), dtype=np.int64))
        row_idx, col_idx = cm.get_top_k_weights(0, 5)
        assert len(row_idx) == 5

    def test_get_top_k_weights_k_zero(self):
        cm = CounterManager([(4, 4)])
        cm.accumulate(0, np.ones((4, 4), dtype=np.int64))
        r0, c0 = cm.get_top_k_weights(0, 0)
        assert len(r0) == 0 and len(c0) == 0

    def test_get_top_k_weights_k_maior_que_size(self):
        cm = CounterManager([(128, 784)])
        cm.accumulate(0, np.ones((128, 784), dtype=np.int64))
        row_idx, col_idx = cm.get_top_k_weights(0, 100000)
        assert len(row_idx) > 0

    def test_reset_selected(self):
        cm = CounterManager([(128, 784)])
        cm.accumulate(0, np.ones((128, 784), dtype=np.int64))
        cm.reset_selected(0, np.array([0]), np.array([0]))
        assert cm.weight_counters[0][0, 0] == 0

    def test_reset_all(self):
        cm = CounterManager([(128, 784), (64, 128)])
        cm.accumulate(0, np.ones((128, 784), dtype=np.int64))
        cm.accumulate(1, np.ones((64, 128), dtype=np.int64))
        cm.reset_all()
        assert cm.weight_counters[0][0, 0] == 0
        assert cm.weight_counters[1][0, 0] == 0

    def test_get_layer_stats(self):
        cm = CounterManager([(128, 784)])
        stats = cm.get_layer_stats(0)
        assert stats["layer"] == 0
        assert stats["shape"] == [128, 784]

    def test_decay_rate_invalido(self):
        with pytest.raises(ValueError):
            CounterManager([(4, 4)], decay_rate=1.5)

    def test_decay_shift_invalido(self):
        with pytest.raises(ValueError):
            CounterManager([(4, 4)], decay_shift=10)

    def test_decay_rate_e_shift_mutuamente_exclusivos(self):
        with pytest.raises(ValueError):
            CounterManager([(4, 4)], decay_rate=0.9, decay_shift=4)


class TestDecay:
    def test_decay_rate_simetrico(self):
        cm = CounterManager([(5, 1)], decay_rate=0.9)
        v = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
        result = cm._apply_decay_array(v.copy())
        expected = np.sign(v) * np.floor(0.9 * np.abs(v)).astype(np.int64)
        assert np.array_equal(result, expected)

    def test_decay_shift_simetrico(self):
        cm = CounterManager([(5, 1)], decay_shift=4)
        v = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
        result = cm._apply_decay_array(v.copy())
        assert np.array_equal(result, -cm._apply_decay_array(-v.copy()))

    def test_zero_ponto_fixo(self):
        cm = CounterManager([(1, 1)], decay_rate=0.9)
        result = cm._apply_decay_array(np.array([0], dtype=np.int64))
        assert result[0] == 0

    def test_decay_shift_passo_minimo(self):
        cm = CounterManager([(5, 1)], decay_shift=4)
        v = np.array([1], dtype=np.int64)
        result = cm._apply_decay_array(v.copy())
        assert result[0] == 0


class TestApplyUpdateNeuron:
    def test_direcao_descida(self):
        layer = TernaryLayer(4, 2, seed=0)
        layer.weights[:] = 0
        cm = CounterManager([(2, 4)])
        cm.accumulate(0, np.array([[5, 5, -5, -5], [0, 0, 0, 0]], dtype=np.int64))
        apply_update_neuron(layer, cm, 0, 0, threshold=1.0)
        assert layer.weights[0].tolist() == [-1, -1, 1, 1]

    def test_selecao_e_reset(self):
        layer = TernaryLayer(4, 3, seed=0)
        cm = CounterManager([(3, 4)])
        cm.accumulate(0, np.array([[9, 9, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64))
        Cpre = cm.weight_counters[0].copy()
        thr = max(1.0, float(np.mean(np.abs(Cpre))))
        mask = np.abs(Cpre) > thr
        apply_update_neuron(layer, cm, 0, 0, threshold=float(thr))
        assert np.all(cm.weight_counters[0][mask] == 0)
        assert np.all(cm.weight_counters[0][~mask] == Cpre[~mask])

    def test_bloqueio_saturacao(self):
        layer = TernaryLayer(4, 2, seed=0)
        layer.weights[0, 0] = 1
        cm = CounterManager([(2, 4)])
        cm.accumulate(0, np.array([[-5, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64))
        result = apply_update_neuron(layer, cm, 0, 0, threshold=1.0)
        assert result["blocked"] == 1
        assert result["transitions"] == 0


class TestApplyUpdateLayerTopk:
    def test_selecao_k(self):
        layer = TernaryLayer(784, 128, seed=42)
        cm = CounterManager([(128, 784)])
        cm.accumulate(0, np.ones((128, 784), dtype=np.int64))
        result = apply_update_layer_topk(layer, cm, 0, 10)
        assert result["selected"] == 10

    def test_reset_apos_topk(self):
        layer = TernaryLayer(784, 128, seed=42)
        cm = CounterManager([(128, 784)])
        cm.accumulate(0, np.ones((128, 784), dtype=np.int64))
        apply_update_layer_topk(layer, cm, 0, 10)
        non_zero_after = np.sum(cm.weight_counters[0] != 0)
        expected_non_zero = cm.weight_counters[0].size - 10
        assert non_zero_after == expected_non_zero


class TestApplyUpdateNetworkTopk:
    def test_multiplas_camadas(self):
        net = TernaryNet(seed=42)
        cm = CounterManager([(128, 784), (64, 128), (10, 64)])
        for i in range(3):
            shape = cm.layer_shapes[i]
            cm.accumulate(i, np.ones(shape, dtype=np.int64))
        result = apply_update_network_topk(net, cm, 5)
        assert result["num_layers"] == 3
        assert result["total_selected"] == 15


class TestComputeCounterDelta:
    def test_batch(self):
        batch = 32
        input_act = np.random.randn(batch, 784).astype(np.float32)
        output_act = np.random.randn(batch, 128).astype(np.float32)
        delta = compute_counter_delta(None, output_act, input_act)
        assert delta.shape == (128, 784)

    def test_single(self):
        output_val = np.random.randn(128).astype(np.float32)
        input_val = np.random.randn(784).astype(np.float32)
        delta = compute_counter_delta_single(output_val, input_val)
        assert delta.shape == (128, 784)
