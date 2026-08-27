"""
[006] Políticas de zeragem de contadores

Três estratégias de reset de contadores após atualização:
- Error-feedback: zera apenas conexões atualizadas (preserva resíduo)
- Full reset: zera todos os contadores da camada
- Normalize: zera atualizados, escala restantes

Uso:
    from src.reset_policies import (
        reset_error_feedback,
        reset_full,
        reset_normalize,
        ResetPolicy,
    )
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent dir to path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.counter_rules import CounterManager


# ---------------------------------------------------------------------------
# Políticas individuais
# ---------------------------------------------------------------------------
def reset_error_feedback(counters: np.ndarray,
                         row_indices: np.ndarray,
                         col_indices: np.ndarray) -> None:
    """
    Error-feedback: zera contadores apenas nas posições atualizadas.

    Contadores não-atualizados mantêm seu valor (resíduo preservado).

    Args:
        counters: array de contadores (modificado in-place).
        row_indices: índices das linhas a zerar.
        col_indices: índices das colunas a zerar.
    """
    counters[row_indices, col_indices] = 0


def reset_full(counters: np.ndarray) -> None:
    """
    Full reset: zera todos os contadores.

    Args:
        counters: array de contadores (modificado in-place).
    """
    counters.fill(0)


def reset_normalize(counters: np.ndarray,
                    row_indices: np.ndarray,
                    col_indices: np.ndarray,
                    target_scale: float = 1.0) -> None:
    """
    Normalize: zera contadores atualizados, escala restantes.

    Escala os contadores não-atualizados para manter a magnitude média
    em target_scale, preservando a informação relativa.

    Args:
        counters: array de contadores (modificado in-place).
        row_indices: índices das linhas a zerar.
        col_indices: índices das colunas a zerar.
        target_scale: magnitude alvo para os contadores restantes.
    """
    # Copiar para identificar não-atualizados
    mask = np.ones_like(counters, dtype=bool)
    mask[row_indices, col_indices] = False

    non_updated = counters[mask]
    nonzero_mask = non_updated != 0

    if nonzero_mask.sum() == 0:
        # Todos os contadores estão nos índices a zerar — não há o que escalar
        counters[row_indices, col_indices] = 0
        return

    mean_abs = np.mean(np.abs(non_updated[nonzero_mask]))

    if mean_abs == 0:
        # Não-atualizados são todos zero — não escalar
        counters[row_indices, col_indices] = 0
        return

    scale_factor = mean_abs / target_scale

    # Zerar atualizados
    counters[row_indices, col_indices] = 0

    # Escalar restantes
    if scale_factor != 1.0:
        counters[mask] = (counters[mask] * scale_factor).astype(np.int64)


# ---------------------------------------------------------------------------
# ResetPolicy — factory/enum
# ---------------------------------------------------------------------------
class ResetPolicy:
    """
    Factory para políticas de reset de contadores.

    Uso:
        policy = ResetPolicy("error_feedback")
        policy.apply(counters, row_idx, col_idx)

    Ou via método estático:
        ResetPolicy.apply("error_feedback", counters, row_idx, col_idx)
    """

    AVAILABLE = {"error_feedback", "full_reset", "normalize"}

    def __init__(self, policy_name: str = "error_feedback",
                 target_scale: float = 1.0):
        """
        Inicializa uma política de reset.

        Args:
            policy_name: nome da política (error_feedback, full_reset, normalize).
            target_scale: escala alvo para política normalize (padrão: 1.0).

        Raises:
            ValueError: se policy_name não estiver disponível.
        """
        if policy_name not in self.AVAILABLE:
            raise ValueError(
                f"Política '{policy_name}' não disponível. "
                f"Escolha entre: {sorted(self.AVAILABLE)}"
            )
        self.policy_name = policy_name
        self.target_scale = target_scale

    def execute(self, counters: np.ndarray,
                row_indices: np.ndarray,
                col_indices: np.ndarray) -> None:
        """
        Aplica a política de reset selecionada.

        Args:
            counters: array de contadores (modificado in-place).
            row_indices: índices das linhas a zerar.
            col_indices: índices das colunas a zerar.
        """
        if self.policy_name == "error_feedback":
            reset_error_feedback(counters, row_indices, col_indices)
        elif self.policy_name == "full_reset":
            reset_full(counters)
        elif self.policy_name == "normalize":
            reset_normalize(counters, row_indices, col_indices, self.target_scale)

    @staticmethod
    def apply(policy_name: str, counters: np.ndarray,
              row_indices: Optional[np.ndarray] = None,
              col_indices: Optional[np.ndarray] = None,
              target_scale: float = 1.0) -> None:
        """
        Aplica uma política de reset via nome (método estático).

        Args:
            policy_name: nome da política.
            counters: array de contadores (modificado in-place).
            row_indices: índices das linhas.
            col_indices: índices das colunas.
            target_scale: escala alvo para normalize.
        """
        if policy_name not in ResetPolicy.AVAILABLE:
            raise ValueError(
                f"Política '{policy_name}' não disponível. "
                f"Escolha entre: {sorted(ResetPolicy.AVAILABLE)}"
            )

        if policy_name == "error_feedback":
            reset_error_feedback(counters, row_indices, col_indices)
        elif policy_name == "full_reset":
            reset_full(counters)
        elif policy_name == "normalize":
            reset_normalize(counters, row_indices, col_indices, target_scale)


# ---------------------------------------------------------------------------
# Integração com CounterManager
# ---------------------------------------------------------------------------
def apply_reset_on_manager(cm: CounterManager,
                           layer_idx: int,
                           row_indices: np.ndarray,
                           col_indices: np.ndarray,
                           policy: str = "error_feedback",
                           target_scale: float = 1.0) -> None:
    """
    Aplica uma política de reset no CounterManager.

    Args:
        cm: CounterManager.
        layer_idx: índice da camada.
        row_indices: índices das linhas.
        col_indices: índices das colunas.
        policy: nome da política.
        target_scale: escala alvo para normalize.
    """
    if policy == "error_feedback":
        reset_error_feedback(cm.weight_counters[layer_idx],
                             row_indices, col_indices)
    elif policy == "full_reset":
        reset_full(cm.weight_counters[layer_idx])
    elif policy == "normalize":
        reset_normalize(cm.weight_counters[layer_idx],
                        row_indices, col_indices, target_scale)


def apply_reset_on_manager_all_layers(cm: CounterManager,
                                      policy: str = "error_feedback",
                                      target_scale: float = 1.0) -> None:
    """
    Aplica uma política de reset em todas as camadas do CounterManager.

    Args:
        cm: CounterManager.
        policy: nome da política.
        target_scale: escala alvo para normalize.
    """
    if policy == "full_reset":
        cm.reset_all()
    elif policy == "error_feedback":
        # Error-feedback sem índices específicos: não faz nada
        pass
    elif policy == "normalize":
        for i in range(cm.num_layers):
            reset_normalize(cm.weight_counters[i],
                            np.array([]), np.array([]), target_scale)


# ---------------------------------------------------------------------------
# Testes unitários
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testes — reset_policies.py")
    print("=" * 60)

    # --- Teste 1: reset_error_feedback ---
    print("\n[1] reset_error_feedback")

    C1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    C1_copy = C1.copy()
    reset_error_feedback(C1, np.array([0, 1]), np.array([0, 2]))
    assert C1[0, 0] == 0, "Erro: não zerou (0,0)"
    assert C1[1, 2] == 0, "Erro: não zerou (1,2)"
    assert C1[0, 1] == 2, "Erro: zerou indevido (0,1)"
    assert C1[2, 2] == 9, "Erro: zerou indevido (2,2)"
    print("   ✓ zera apenas índices fornecidos")
    print(f"     antes: {C1_copy.tolist()}")
    print(f"     depois: {C1.tolist()}")

    # --- Teste 2: reset_full ---
    print("\n[2] reset_full")

    C2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    reset_full(C2)
    assert np.all(C2 == 0), "Erro: não zerou tudo"
    print("   ✓ zera todos os contadores")

    # --- Teste 3: reset_normalize ---
    print("\n[3] reset_normalize")

    C3 = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    C3_copy = C3.copy()
    # Zerar (0,0) e (1,1), escalar restantes para target_scale=10
    reset_normalize(C3, np.array([0, 1]), np.array([0, 1]), target_scale=10.0)

    # (0,0) e (1,1) devem ser 0
    assert C3[0, 0] == 0, "Erro: não zerou (0,0)"
    assert C3[1, 1] == 0, "Erro: não zerou (1,1)"

    # Não-atualizados: [20, 30, 40, 60]
    # mean(|non_zero|) = mean([20, 30, 40, 60]) = 37.5
    # scale_factor = 37.5 / 10.0 = 3.75
    expected_scale = 37.5 / 10.0
    assert C3[0, 1] == int(20 * expected_scale), \
        f"Erro: esperado {int(20 * expected_scale)}, got {C3[0, 1]}"
    print(f"   ✓ escala restantes corretamente (factor={expected_scale:.2f})")
    print(f"     antes: {C3_copy.tolist()}")
    print(f"     depois: {C3.tolist()}")

    # --- Teste 4: ResetPolicy class ---
    print("\n[4] ResetPolicy class")

    # Validar políticas disponíveis
    assert ResetPolicy.AVAILABLE == {"error_feedback", "full_reset", "normalize"}
    print("   ✓ políticas disponíveis: {sorted(ResetPolicy.AVAILABLE)}")

    # Instanciar e aplicar
    policy = ResetPolicy("error_feedback")
    C4 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    policy.execute(C4, np.array([0]), np.array([1]))
    assert C4[0, 1] == 0, "Erro: error_feedback via objeto"
    assert C4[0, 0] == 1, "Erro: alterou indevido via objeto"
    print("   ✓ ResetPolicy objeto: execute() roteia corretamente")

    # Validar política inválida
    try:
        ResetPolicy("invalid_policy")
        assert False, "Erro: não levantou ValueError"
    except ValueError:
        print("   ✓ levanta ValueError para política inválida")

    # --- Teste 5: ResetPolicy.apply (estático) ---
    print("\n[5] ResetPolicy.apply (estático)")

    C5 = np.array([[10, 20], [30, 40]], dtype=np.int64)
    ResetPolicy.apply("full_reset", C5)
    assert np.all(C5 == 0), "Erro: full_reset estático"
    print("   ✓ full_reset via método estático")

    C6 = np.array([[10, 20], [30, 40]], dtype=np.int64)
    ResetPolicy.apply("error_feedback", C6,
                      np.array([0, 1]), np.array([0, 1]))
    assert C6[0, 0] == 0 and C6[1, 1] == 0
    assert C6[0, 1] == 20 and C6[1, 0] == 30
    print("   ✓ error_feedback via método estático")

    C7 = np.array([[10, 20], [30, 40]], dtype=np.int64)
    ResetPolicy.apply("normalize", C7,
                      np.array([0]), np.array([0]),
                      target_scale=10.0)
    assert C7[0, 0] == 0, "Erro: normalize estático não zerou"
    # mean(|não-atualizados|) = mean([20, 30, 40]) = 30
    # scale_factor = 30 / 10 = 3.0
    # C7[1,0] = 30 * 3 = 90
    assert C7[1, 0] == 90, f"Erro: normalize estático, esperado 90, got {C7[1, 0]}"
    print("   ✓ normalize via método estático")

    # --- Teste 6: Integração com CounterManager ---
    print("\n[6] Integração com CounterManager")

    from src.counter_rules import CounterManager

    cm = CounterManager([(128, 784)])
    cm.accumulate(0, np.ones((128, 784), dtype=np.int64))

    # Error-feedback
    reset_error_feedback(cm.weight_counters[0],
                         np.array([0, 1]), np.array([0, 1]))
    assert cm.weight_counters[0][0, 0] == 0
    assert cm.weight_counters[0][2, 2] == 1  # não foi zerado
    print("   ✓ error_feedback via função direta")

    # Full reset
    reset_full(cm.weight_counters[0])
    assert cm.weight_counters[0].sum() == 0
    print("   ✓ full_reset via função direta")

    # Via apply_reset_on_manager
    cm2 = CounterManager([(128, 784)])
    cm2.accumulate(0, np.ones((128, 784), dtype=np.int64))
    apply_reset_on_manager(cm2, 0, np.array([0]), np.array([0]),
                           policy="error_feedback")
    assert cm2.weight_counters[0][0, 0] == 0
    assert cm2.weight_counters[0][1, 1] == 1
    print("   ✓ apply_reset_on_manager roteia corretamente")

    # --- Teste 7: apply_reset_on_manager_all_layers ---
    print("\n[7] apply_reset_on_manager_all_layers")

    cm3 = CounterManager([(128, 784), (64, 128)])
    cm3.accumulate(0, np.ones((128, 784), dtype=np.int64))
    cm3.accumulate(1, np.ones((64, 128), dtype=np.int64))

    apply_reset_on_manager_all_layers(cm3, policy="full_reset")
    assert cm3.weight_counters[0].sum() == 0
    assert cm3.weight_counters[1].sum() == 0
    print("   ✓ full_reset em todas as camadas")

    # --- Teste 8: Integração completa com B2 + reset ---
    print("\n[8] Integração completa: B2 + reset policies")

    from src.ternary_net import TernaryNet
    from src.counter_rules import (
        apply_update_layer_topk,
        CounterManager,
    )

    net = TernaryNet(seed=42)
    cm_full = CounterManager([(128, 784), (64, 128), (10, 64)])

    # Acumular contadores
    for i in range(3):
        shape = cm_full.layer_shapes[i]
        cm_full.accumulate(i, np.ones(shape, dtype=np.int64))

    # B2 na camada 0
    row0, col0 = cm_full.get_top_k_weights(0, 10)
    apply_update_layer_topk(net.layers[0], cm_full, 0, 10, delta_sign=1)

    # Aplicar error-feedback manualmente
    reset_error_feedback(cm_full.weight_counters[0], row0, col0)

    # Verificar: top-k zerados, demais mantidos
    assert cm_full.weight_counters[0][row0[0], col0[0]] == 0
    assert cm_full.weight_counters[0][0, 0] == 1  # não foi top-k
    print("   ✓ B2 + error-feedback: top-k zerados, resíduo preservado")

    # Full reset na mesma camada
    cm_full2 = CounterManager([(128, 784)])
    cm_full2.accumulate(0, np.ones((128, 784), dtype=np.int64))
    row2, col2 = cm_full2.get_top_k_weights(0, 10)
    reset_full(cm_full2.weight_counters[0])
    assert cm_full2.weight_counters[0].sum() == 0
    # Recarregar para testar normalize
    cm_full2.accumulate(0, np.ones((128, 784), dtype=np.int64))
    row2, col2 = cm_full2.get_top_k_weights(0, 10)
    reset_normalize(cm_full2.weight_counters[0], row2, col2, target_scale=1.0)
    assert cm_full2.weight_counters[0][row2[0], col2[0]] == 0
    # Demais devem ter sido escalados
    nonzero_remaining = cm_full2.weight_counters[0][np.arange(128) != row2[0]][:, np.arange(784) != col2[0]]
    assert nonzero_remaining.sum() > 0, "Erro: normalize zerou demais"
    print("   ✓ B2 + normalize: top-k zerados, restantes escalados")

    # --- Teste 9: Reprodutibilidade ---
    print("\n[9] Reprodutibilidade")

    C_r1 = np.array([[5, 10, 15], [20, 25, 30]], dtype=np.int64)
    C_r2 = C_r1.copy()

    reset_error_feedback(C_r1, np.array([0]), np.array([1]))
    reset_error_feedback(C_r2, np.array([0]), np.array([1]))

    assert np.array_equal(C_r1, C_r2), "Error-feedback não reprodutível"
    print("   ✓ error_feedback reprodutível")

    C_r3 = np.array([[5, 10, 15], [20, 25, 30]], dtype=np.int64)
    C_r4 = C_r3.copy()
    reset_normalize(C_r3, np.array([0]), np.array([1]), target_scale=1.0)
    reset_normalize(C_r4, np.array([0]), np.array([1]), target_scale=1.0)
    assert np.array_equal(C_r3, C_r4), "Normalize não reprodutível"
    print("   ✓ normalize reprodutível")

    print("\n" + "=" * 60)
    print("Todos os testes passaram!")
    print("=" * 60)
