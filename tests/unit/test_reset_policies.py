"""Testes unitários para src/reset_policies.py."""
import numpy as np
import pytest

from src.counter_rules import CounterManager
from src.reset_policies import (
    ResetPolicy,
    apply_reset_on_manager,
    apply_reset_on_manager_all_layers,
    reset_error_feedback,
    reset_full,
    reset_normalize,
)


class TestResetErrorFeedback:
    def test_zera_apenas_indices(self):
        C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        reset_error_feedback(C, np.array([0, 1]), np.array([0, 2]))
        assert C[0, 0] == 0
        assert C[1, 2] == 0
        assert C[0, 1] == 2
        assert C[2, 2] == 9

    def test_preserva_nao_selecionados(self):
        C = np.array([[10, 20], [30, 40]], dtype=np.int64)
        reset_error_feedback(C, np.array([0]), np.array([0]))
        assert C[0, 0] == 0
        assert C[0, 1] == 20
        assert C[1, 0] == 30
        assert C[1, 1] == 40


class TestResetFull:
    def test_zera_tudo(self):
        C = np.array([[1, 2], [3, 4]], dtype=np.int64)
        reset_full(C)
        assert np.all(C == 0)


class TestResetNormalize:
    def test_reduz_escala(self):
        C = np.array([[10, 20, 30, 40]], dtype=np.int64)
        reset_normalize(C, np.array([0]), np.array([0]), target_scale=10.0)
        assert C[0, 0] == 0
        remaining = C[0, 1:]
        mean_after = float(np.mean(np.abs(remaining)))
        assert abs(mean_after - 10.0) <= 2.0

    def test_fator_correto(self):
        C = np.array([[10, 20, 30, 40]], dtype=np.int64)
        reset_normalize(C, np.array([0]), np.array([0]), target_scale=10.0)
        # mean([20,30,40]) = 30, fator = 10/30 = 1/3
        # [20,30,40] * 1/3 = [6.67, 10, 13.33] -> round = [7, 10, 13]
        assert C[0, 1] == 7
        assert C[0, 2] == 10
        assert C[0, 3] == 13

    def test_indices_vazios(self):
        C = np.array([[10, 20], [30, 40]], dtype=np.int64)
        reset_normalize(C, np.array([], dtype=np.int64), np.array([], dtype=np.int64), target_scale=10.0)
        # Todos são não-atualizados, deve escalar todos
        assert C[0, 0] > 0  # não zerou


class TestResetPolicy:
    def test_politicas_disponiveis(self):
        assert ResetPolicy.AVAILABLE == {"error_feedback", "full_reset", "normalize"}

    def test_politica_invalida(self):
        with pytest.raises(ValueError):
            ResetPolicy("invalid_policy")

    def test_execute_error_feedback(self):
        policy = ResetPolicy("error_feedback")
        C = np.array([[1, 2], [3, 4]], dtype=np.int64)
        policy.execute(C, np.array([0]), np.array([1]))
        assert C[0, 1] == 0
        assert C[0, 0] == 1

    def test_apply_estatico_full_reset(self):
        C = np.array([[10, 20], [30, 40]], dtype=np.int64)
        ResetPolicy.apply("full_reset", C)
        assert np.all(C == 0)

    def test_apply_estatico_normalize(self):
        C = np.array([[10, 20], [30, 40]], dtype=np.int64)
        ResetPolicy.apply("normalize", C, np.array([0]), np.array([0]), target_scale=10.0)
        assert C[0, 0] == 0
        # mean([20,30,40]) = 30, fator = 1/3
        assert C[1, 0] == 10
        assert C[0, 1] == 7
        assert C[1, 1] == 13


class TestIntegracaoCounterManager:
    def test_apply_reset_on_manager(self):
        cm = CounterManager([(128, 784)])
        cm.accumulate(0, np.ones((128, 784), dtype=np.int64))
        apply_reset_on_manager(cm, 0, np.array([0]), np.array([0]), policy="error_feedback")
        assert cm.weight_counters[0][0, 0] == 0
        assert cm.weight_counters[0][1, 1] == 1

    def test_apply_reset_on_manager_all_layers_full(self):
        cm = CounterManager([(128, 784), (64, 128)])
        cm.accumulate(0, np.ones((128, 784), dtype=np.int64))
        cm.accumulate(1, np.ones((64, 128), dtype=np.int64))
        apply_reset_on_manager_all_layers(cm, policy="full_reset")
        assert cm.weight_counters[0].sum() == 0
        assert cm.weight_counters[1].sum() == 0

    def test_apply_reset_on_manager_all_layers_normalize(self):
        cm = CounterManager([(4, 4), (2, 4)])
        cm.accumulate(0, np.full((4, 4), 10, dtype=np.int64))
        cm.accumulate(1, np.full((2, 4), 10, dtype=np.int64))
        # Sem try/except: IndexError deve propagar e falhar o teste.
        apply_reset_on_manager_all_layers(cm, policy="normalize", target_scale=1.0)
        # Oráculo: escala p/ o alvo em todas as camadas (10 * 1/10 = 1)
        # e dtype inteiro preservado.
        assert np.all(cm.weight_counters[0] == 1)
        assert np.all(cm.weight_counters[1] == 1)
        assert np.issubdtype(cm.weight_counters[0].dtype, np.integer)
        assert np.issubdtype(cm.weight_counters[1].dtype, np.integer)
