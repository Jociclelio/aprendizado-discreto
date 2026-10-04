"""Testes unitários para src/fake_gradient.py."""
import numpy as np
import pytest

from src.fake_gradient import (
    CounterAccumulator,
    adaptive_triangular_fake_grad,
    get_adaptive_radius,
    get_layer_radii,
    propagate_delta,
    propagate_delta_with_fake,
    propagate_delta_with_fake_quantized,
    quantize_fake_grad,
    triangular_fake_grad,
)


class TestTriangularFakeGrad:
    def test_pico_em_zero(self):
        z = np.array([0.0])
        result = triangular_fake_grad(z, scale=2.0, radius=1.0)
        assert np.isclose(result[0], 2.0)

    def test_zero_fora_do_raio(self):
        z = np.array([1.0, -1.0, 2.0, -2.0])
        result = triangular_fake_grad(z, scale=1.0, radius=1.0)
        assert np.allclose(result, [0.0, 0.0, 0.0, 0.0])

    def test_linear_decrescente(self):
        z = np.array([0.0, 0.5, 0.8, 1.0])
        result = triangular_fake_grad(z, scale=1.0, radius=1.0)
        expected = np.array([1.0, 0.5, 0.2, 0.0])
        assert np.allclose(result, expected)

    def test_preserva_shape(self):
        z = np.random.randn(32, 128)
        result = triangular_fake_grad(z, scale=0.5, radius=2.0)
        assert result.shape == (32, 128)


class TestQuantizeFakeGrad:
    def test_levels_3(self):
        z = np.array([-0.5, 0.0, 0.3, 1.5, -2.0])
        result = quantize_fake_grad(z, levels=3)
        expected = np.array([-1, 0, 1, 1, -1], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_levels_5(self):
        z = np.array([-0.9, -0.3, 0.0, 0.3, 0.9])
        result = quantize_fake_grad(z, levels=5)
        expected = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_levels_7(self):
        z = np.array([-0.9, -0.3, 0.0, 0.3, 0.9])
        result = quantize_fake_grad(z, levels=7)
        expected = np.array([-3, -1, 0, 1, 3], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_deterministico(self):
        z = np.random.randn(1000)
        r1 = quantize_fake_grad(z, levels=5)
        r2 = quantize_fake_grad(z, levels=5)
        assert np.array_equal(r1, r2)

    def test_round_vs_trunc(self):
        prod = np.array([[0.9, -0.9, 0.4, 1.6]])
        trunc = prod.astype(np.int64)
        rnd = np.round(prod).astype(np.int64)
        assert rnd.tolist() == [[1, -1, 0, 2]]
        assert trunc.tolist() == [[0, 0, 0, 1]]


class TestPropagateDelta:
    def test_shape(self):
        batch, next_units, current_units = 4, 10, 20
        delta_next = np.random.randn(batch, next_units)
        weights = np.random.randn(current_units, next_units)
        delta = propagate_delta(delta_next, weights, fake_grad_val=1.0)
        assert delta.shape == (batch, current_units)

    def test_escala(self):
        batch, next_units, current_units = 4, 10, 20
        delta_next = np.random.randn(batch, next_units)
        weights = np.random.randn(current_units, next_units)
        delta = propagate_delta(delta_next, weights, fake_grad_val=1.0)
        delta2 = propagate_delta(delta_next, weights, fake_grad_val=0.5)
        assert np.allclose(delta2, delta * 0.5)


class TestPropagateDeltaWithFake:
    def test_shape(self):
        batch, next_units, current_units = 4, 10, 20
        delta_next = np.random.randn(batch, next_units)
        weights = np.random.randn(current_units, next_units)
        z_local = np.random.randn(batch, current_units)
        delta = propagate_delta_with_fake(delta_next, weights, z_local, scale=1.0, radius=1.0)
        assert delta.shape == (batch, current_units)


class TestPropagateDeltaWithFakeQuantized:
    def test_shape(self):
        batch, next_units, current_units = 4, 10, 20
        delta_next = np.random.randn(batch, next_units)
        weights = np.random.randn(current_units, next_units)
        z_local = np.random.randn(batch, current_units)
        delta = propagate_delta_with_fake_quantized(
            delta_next, weights, z_local, scale=1.0, radius=1.0, levels=5
        )
        assert delta.shape == (batch, current_units)


class TestAdaptiveRadius:
    def test_get_adaptive_radius(self):
        assert np.isclose(get_adaptive_radius(784, 1.0), 28.0)
        assert np.isclose(get_adaptive_radius(128, 1.0), np.sqrt(128))
        assert np.isclose(get_adaptive_radius(64, 1.0), 8.0)
        assert np.isclose(get_adaptive_radius(784, 0.5), 14.0)

    def test_get_layer_radii(self):
        radii = get_layer_radii([784, 128, 64], scale_factor=1.0)
        assert len(radii) == 3
        assert np.isclose(radii[0], 28.0)

    def test_adaptive_wrapper(self):
        z = np.array([0.0, 5.0, 30.0])
        a = adaptive_triangular_fake_grad(z, input_dim=784, scale=1.0, scale_factor=1.0)
        b = triangular_fake_grad(z, scale=1.0, radius=28.0)
        assert np.allclose(a, b)


class TestCounterAccumulator:
    def test_inicializacao(self):
        acc = CounterAccumulator(shape=(3, 4))
        assert np.all(acc.get_counters() == 0)

    def test_update(self):
        acc = CounterAccumulator(shape=(3, 4))
        acc.update(np.ones((3, 4)))
        assert np.all(acc.get_counters() == 1)

    def test_acumulacao(self):
        acc = CounterAccumulator(shape=(3, 4))
        acc.update(np.ones((3, 4)))
        acc.update(np.ones((3, 4)) * 2)
        assert np.all(acc.get_counters() == 3)

    def test_reset_selected_bool(self):
        acc = CounterAccumulator(shape=(3, 4))
        acc.update(np.ones((3, 4)) * 3)
        mask = np.array([[True, False, False, True],
                         [False, True, False, False],
                         [False, False, True, False]])
        acc.reset_selected(mask)
        expected = np.array([[0, 3, 3, 0],
                             [3, 0, 3, 3],
                             [3, 3, 0, 3]])
        assert np.array_equal(acc.get_counters(), expected)

    def test_reset_all(self):
        acc = CounterAccumulator(shape=(3, 4))
        acc.update(np.ones((3, 4)))
        acc.reset_all()
        assert np.all(acc.get_counters() == 0)

    def test_normalize(self):
        acc = CounterAccumulator(shape=(3, 4))
        acc.update(np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12]]))
        acc.normalize()
        assert acc.get_counters().dtype == np.int64 or np.issubdtype(
            acc.get_counters().dtype, np.integer)
        assert int(np.max(np.abs(acc.get_counters()))) == 1

    def test_top_k_indices(self):
        acc = CounterAccumulator(shape=(2, 5))
        acc.update(np.array([[1, 5, 2, 8, 3],
                             [4, 1, 7, 2, 6]]))
        top3 = acc.top_k_indices(3)
        assert top3.sum() == 6
        expected_top = np.array([[False, True, False, True, True],
                                 [True, False, True, False, True]])
        assert np.array_equal(top3, expected_top)

    def test_top_k_indices_k_zero(self):
        acc = CounterAccumulator(shape=(2, 5))
        assert not acc.top_k_indices(0).any()
