"""Testes unitários para src/ternary_net.py."""
import numpy as np
import pytest

from src.ternary_net import (
    TernaryLayer,
    TernaryNet,
    step_activation,
    ternary_accuracy,
    ternary_binary_loss,
)


class TestStepActivation:
    def test_degrau_centrado_zero(self):
        z = np.array([-2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0])
        out = step_activation(z)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        assert np.allclose(out, expected)

    def test_preserva_shape(self):
        z = np.random.randn(32, 128)
        out = step_activation(z)
        assert out.shape == z.shape

    def test_saida_binaria(self):
        z = np.random.randn(100)
        out = step_activation(z)
        assert np.all(np.isin(out, [0.0, 1.0]))


class TestTernaryLayer:
    def test_inizializacao_ternaria(self):
        layer = TernaryLayer(784, 128, seed=42)
        assert layer.weights.shape == (128, 784)
        assert layer.weights.dtype == np.int8
        assert np.all(np.isin(layer.weights, [-1, 0, 1]))

    def test_bias_int32(self):
        layer = TernaryLayer(784, 128, seed=42)
        assert layer.bias.shape == (128,)
        assert layer.bias.dtype == np.int32

    def test_forward_single(self):
        layer = TernaryLayer(784, 128, seed=42)
        x = np.random.randn(784).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (128,)
        assert np.all(np.isin(out, [0.0, 1.0]))

    def test_forward_batch(self):
        layer = TernaryLayer(784, 128, seed=42)
        x = np.random.randn(32, 784).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (32, 128)

    def test_quantize_weights(self):
        layer = TernaryLayer(784, 128, seed=42)
        layer.weights = (layer.weights.astype(np.float64) + 0.3).astype(np.int8)
        layer.quantize_weights()
        assert np.all(np.isin(layer.weights, [-1, 0, 1]))

    def test_update_weights_clipping(self):
        layer = TernaryLayer(784, 128, seed=42)
        layer.update_weights(np.ones((128, 784), dtype=np.int64))
        assert np.all(layer.weights <= 1)

    def test_update_bias_faixa(self):
        layer = TernaryLayer(784, 128, seed=42)
        layer.update_bias(5)
        assert np.all(layer.bias >= -784)
        assert np.all(layer.bias <= 783)

    def test_reprodutibilidade(self):
        l1 = TernaryLayer(10, 5, seed=123)
        l2 = TernaryLayer(10, 5, seed=123)
        assert np.array_equal(l1.weights, l2.weights)


class TestTernaryNet:
    def test_arquitetura_padrao(self):
        net = TernaryNet(seed=42)
        assert len(net.layers) == 3
        assert net.layers[0].input_dim == 784
        assert net.layers[0].output_dim == 128
        assert net.layers[1].input_dim == 128
        assert net.layers[1].output_dim == 64
        assert net.layers[2].input_dim == 64
        assert net.layers[2].output_dim == 10

    def test_forward_completo(self):
        net = TernaryNet(seed=42)
        x = np.random.randn(1, 784).astype(np.float32)
        out = net.forward(x)
        assert out.shape == (1, 10)
        assert np.all(np.isin(out, [0.0, 1.0]))

    def test_forward_single(self):
        net = TernaryNet(seed=42)
        x = np.random.randn(784).astype(np.float32)
        out = net.forward(x)
        assert out.shape == (10,)

    def test_forward_batch(self):
        net = TernaryNet(seed=42)
        x = np.random.randn(64, 784).astype(np.float32)
        out = net.forward(x)
        assert out.shape == (64, 10)

    def test_forward_with_scores(self):
        net = TernaryNet(seed=42)
        x = np.random.randn(16, 784).astype(np.float32)
        scores, binary_out, pre_acts, acts = net.forward_with_scores(x)
        assert scores.shape == (16, 10)
        assert binary_out.shape == (16, 10)
        assert len(pre_acts) == 3
        assert len(acts) == 4

    def test_forward_with_scores_single(self):
        net = TernaryNet(seed=42)
        x = np.random.randn(784).astype(np.float32)
        scores, binary_out, pre_acts, acts = net.forward_with_scores(x)
        assert scores.shape == (10,)
        assert binary_out.shape == (10,)

    def test_reprodutibilidade(self):
        net1 = TernaryNet(seed=123)
        net2 = TernaryNet(seed=123)
        x = np.random.randn(1, 784).astype(np.float32)
        out1 = net1.forward(x)
        out2 = net2.forward(x)
        assert np.array_equal(out1, out2)

    def test_get_stats(self):
        net = TernaryNet(seed=42)
        stats = net.get_stats()
        assert "architecture" in stats
        assert "total_weights" in stats
        assert stats["ternary_fraction"] == 1.0


class TestLossFunctions:
    def test_ternary_binary_loss_perfeita(self):
        preds = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        targets = np.array([0, 1])
        loss, grad = ternary_binary_loss(preds, targets)
        assert loss == 0.0

    def test_ternary_binary_loss_imperfeita(self):
        preds = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        targets = np.array([0, 1])
        loss, grad = ternary_binary_loss(preds, targets)
        assert loss > 0.0

    def test_ternary_accuracy_perfeita(self):
        preds = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        targets = np.array([0, 1])
        acc = ternary_accuracy(preds, targets)
        assert acc == 1.0

    def test_ternary_accuracy_imperfeita(self):
        preds = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        targets = np.array([0, 1])
        acc = ternary_accuracy(preds, targets)
        assert acc == 0.0
