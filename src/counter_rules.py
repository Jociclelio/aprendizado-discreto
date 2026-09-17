"""
[005] Regras de atualização por contador (B1, B2, B3)

Três mecanismos de seleção de conexões para atualização discreta:
- B1: Por neurônio (disparo individual)
- B2: Por camada (Top-K na camada)
- B3: Por rede (Top-K por camada simultaneamente)

Uso:
    from src.counter_rules import (
        CounterManager,
        apply_update_neuron,
        apply_update_layer_topk,
        apply_update_network_topk,
    )
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent dir to path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ternary_net import TernaryLayer, TernaryNet


# ---------------------------------------------------------------------------
# Utilitário: atualização unitária de peso
# ---------------------------------------------------------------------------
def update_weight(W: np.ndarray, sign: int) -> np.ndarray:
    """
    Aplica uma unidade de mudança ao peso: W += sign.

    Clipa para manter W ∈ {−1, 0, +1}.

    Args:
        W: peso atual (int8).
        sign: direção do incremento (+1 ou −1).

    Returns:
        Novo peso clipado em {−1, 0, +1}.
    """
    new_w = W.astype(np.int16) + sign
    return np.clip(new_w, -1, 1).astype(np.int8)


# ---------------------------------------------------------------------------
# CounterManager
# ---------------------------------------------------------------------------
class CounterManager:
    """
    Gerencia contadores inteiros por camada.

    Cada conexão (peso) tem um contador C_ij que acumula evidências
    de que deve ser atualizada.

    Contrato [012]: C acumula GRADIENTE (sinal de prediction − target
    propagado pelo surrogate). A atualização é sempre descida:
        W <- clip(W − sign(C), −1, +1).
    Sem multiplicação extra por sign(−error) da amostra corrente.

    Política de leak [012] (simétrica em sinal, zero é ponto fixo):
    - decay_rate (gamma):  C <- sign(C) * floor(gamma * |C|).
      Truncamento em direção a zero sobre a magnitude; |C|=1 decai
      para 0 quando gamma < 1. Simétrico: decay(−C) = −decay(C).
    - decay_shift (s):     C <- sign(C) * (|C| − step),
      onde step = |C| >> s se nonzero, senão 1 (para |C| > 0).
      O passo mínimo de 1 garante que valores pequenos (|C| < 2^s)
      ainda vazam em direção a zero; único ponto fixo é 0.
      (O bitshift puro `C − (C >> s)` é aritmético/floor e NÃO é
      simétrico; por isso não é mais usado.)

    Suporta decaimento temporal (leaky) — task 011, corrigido em 012:
        C^{(t+1)} = decay(C^{(t)}) + Delta
    """

    def __init__(self, layer_shapes: List[Tuple[int, int]],
                 decay_rate: Optional[float] = None,
                 decay_shift: Optional[int] = None):
        """
        Inicializa contadores para todas as camadas.

        Args:
            layer_shapes: lista de (output_dim, input_dim) por camada.
            decay_rate: fator gamma em (0,1) para decaimento multiplicativo.
                        Ex.: 0.99, 0.95, 0.90. Se None, sem decaimento.
            decay_shift: alternativa bitshift (int 3..5). Se setado, usa
                         C - (C >> shift) em vez de gamma. Mais eficiente em HW.
        """
        if decay_rate is not None and decay_shift is not None:
            raise ValueError("Use apenas um: decay_rate ou decay_shift, não ambos")
        if decay_rate is not None and not (0.0 < decay_rate < 1.0):
            raise ValueError(f"decay_rate deve estar em (0,1), got {decay_rate}")
        if decay_shift is not None and decay_shift not in (1, 2, 3, 4, 5, 6, 7, 8):
            raise ValueError(f"decay_shift deve estar em 1..8, got {decay_shift}")

        self.num_layers = len(layer_shapes)
        self.layer_shapes = layer_shapes
        self.decay_rate = decay_rate
        self.decay_shift = decay_shift

        # Contadores de pesos: (output_dim, input_dim) por camada
        self.weight_counters: List[np.ndarray] = []
        for shape in layer_shapes:
            self.weight_counters.append(np.zeros(shape, dtype=np.int64))

        # Contadores de bias: (output_dim,) por camada
        self.bias_counters: List[np.ndarray] = []
        for shape in layer_shapes:
            self.bias_counters.append(np.zeros(shape[0], dtype=np.int64))

    def _apply_decay_array(self, arr: np.ndarray) -> np.ndarray:
        """[012] Aplica decaimento simétrico (ver docstring da classe)."""
        if self.decay_rate is None and self.decay_shift is None:
            return arr
        if self.decay_shift is not None:
            # Simétrico: opera na magnitude, passo mínimo 1 p/ |C|>0.
            # gamma efetivo ≈ 1 − 2^-shift para |C| grande.
            abs_c = np.abs(arr)
            step = abs_c >> self.decay_shift
            # Onde |C|>0 mas step==0, vazar 1 unidade em direção a zero.
            step = np.where((abs_c > 0) & (step == 0), 1, step)
            decayed_abs = abs_c - step
            return np.sign(arr).astype(np.int64) * decayed_abs.astype(np.int64)
        # decay_rate: sign(C) * floor(gamma * |C|) — simétrico, zero fixo.
        abs_c = np.abs(arr).astype(np.float64)
        decayed_abs = np.floor(self.decay_rate * abs_c).astype(np.int64)
        return np.sign(arr).astype(np.int64) * decayed_abs

    def decay(self, layer_idx: Optional[int] = None) -> None:
        """
        Aplica decaimento temporal aos contadores.

        C <- floor(gamma*C)  ou  C <- C - (C >> shift)

        Args:
            layer_idx: se None, aplica a todas as camadas; senão só à camada indicada.
        """
        if self.decay_rate is None and self.decay_shift is None:
            return
        indices = [layer_idx] if layer_idx is not None else range(self.num_layers)
        for i in indices:
            self.weight_counters[i] = self._apply_decay_array(self.weight_counters[i])
            self.bias_counters[i] = self._apply_decay_array(self.bias_counters[i])

    def accumulate(self, layer_idx: int, delta_weights: np.ndarray,
                   delta_biases: Optional[np.ndarray] = None,
                   apply_decay: bool = False) -> None:
        """
        Acumula deltas nos contadores de uma camada.

        Args:
            layer_idx: índice da camada (0-based).
            delta_weights: incremento por conexão, shape (output_dim, input_dim).
            delta_biases: incremento por neurônio, shape (output_dim,).
            apply_decay: se True, aplica decaimento antes de somar Delta
                         (C = decay(C) + Delta). Útil para leak por amostra.
                         Se False, decaimento deve ser chamado explicitamente via decay().
        """
        if apply_decay and (self.decay_rate is not None or self.decay_shift is not None):
            self.weight_counters[layer_idx] = self._apply_decay_array(self.weight_counters[layer_idx])
            self.bias_counters[layer_idx] = self._apply_decay_array(self.bias_counters[layer_idx])

        self.weight_counters[layer_idx] += delta_weights.astype(np.int64, casting="unsafe")
        if delta_biases is not None:
            self.bias_counters[layer_idx] += delta_biases.astype(np.int64, casting="unsafe")

    def get_top_k_weights(self, layer_idx: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna índices das K conexões com maior |C_ij| em uma camada.

        [012] K <= 0 retorna seleção vazia (antes, [-k:] selecionava tudo).

        Args:
            layer_idx: índice da camada.
            k: número de conexões a selecionar.
        Returns:
            (row_indices, col_indices) dos top-k pesos.
        """
        C = self.weight_counters[layer_idx]
        abs_C = np.abs(C)

        if k is None or k <= 0:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

        if k >= C.size:
            # Todos os índices não-nulos
            nonzero_flat = np.flatnonzero(abs_C > 0)
            if len(nonzero_flat) == 0:
                # Todos zerados, retorna todos
                return np.indices(C.shape).reshape(2, -1)
            return np.unravel_index(nonzero_flat, C.shape)

        # argpartition eficiente O(n)
        flat_abs = abs_C.flatten()
        top_k_flat = np.argpartition(flat_abs, -k)[-k:]
        # Converter flat indices de volta para 2D
        return np.unravel_index(top_k_flat, C.shape)

    def get_neuron_counters(self, layer_idx: int, neuron_idx: int) -> np.ndarray:
        """
        Retorna contadores de todas as conexões de entrada de um neurônio.

        Args:
            layer_idx: índice da camada.
            neuron_idx: índice do neurônio de saída.

        Returns:
            Array de contadores, shape (input_dim,).
        """
        return self.weight_counters[layer_idx][neuron_idx].copy()

    def get_layer_stats(self, layer_idx: int) -> dict:
        """Retorna estatísticas dos contadores de uma camada."""
        C = self.weight_counters[layer_idx]
        return {
            "layer": layer_idx,
            "shape": list(C.shape),
            "sum": int(C.sum()),
            "max_abs": int(np.max(np.abs(C))),
            "nonzero": int(np.sum(C != 0)),
        }

    def reset_selected(self, layer_idx: int, row_indices: np.ndarray,
                       col_indices: np.ndarray) -> None:
        """
        Zera contadores selecionados (error-feedback).

        Args:
            layer_idx: índice da camada.
            row_indices: índices das linhas a zerar.
            col_indices: índices das colunas a zerar.
        """
        row_indices = np.asarray(row_indices, dtype=np.int64)
        col_indices = np.asarray(col_indices, dtype=np.int64)
        self.weight_counters[layer_idx][row_indices, col_indices] = 0

    def reset_all(self) -> None:
        """Zera todos os contadores."""
        for i in range(self.num_layers):
            self.weight_counters[i].fill(0)
            self.bias_counters[i].fill(0)

    def weight_flip_rate(self, prev_weights: List[np.ndarray],
                         curr_weights: List[np.ndarray]) -> float:
        """
        Calcula fração de pesos que trocaram de valor entre dois snapshots.
        Útil para medir oscilação/estabilidade (task 011 critério).
        """
        total = sum(p.size for p in prev_weights)
        flips = sum(int(np.sum(p != c)) for p, c in zip(prev_weights, curr_weights))
        return flips / total if total > 0 else 0.0

    def get_all_stats(self) -> dict:
        """Retorna estatísticas de todas as camadas."""
        return {
            "num_layers": self.num_layers,
            "layer_stats": [self.get_layer_stats(i) for i in range(self.num_layers)],
        }

    def __repr__(self) -> str:
        return f"CounterManager({self.num_layers} layers, {self.get_all_stats()})"


# ---------------------------------------------------------------------------
# B1: Regra por neurônio (disparo individual)
# ---------------------------------------------------------------------------
def apply_update_neuron(
    layer: TernaryLayer,
    counters: CounterManager,
    layer_idx: int,
    neuron_idx: int,
    delta_sign: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Dict:
    """
    Regra B1: atualiza conexões de um neurônio específico.

    Contrato [012]: C acumula gradiente ⇒ W <- clip(W − sign(C)).
    O parâmetro legado `delta_sign` é ignorado (mantido por
    compatibilidade; a direção é sempre −sign(C)).

    Semântica de saturação [012]: a tentativa em peso já saturado
    (±1 pedindo o mesmo sentido) é BLOQUEADA (peso não muda), mas o
    contador é consumido (resetado) e o evento é contado como
    `blocked` separadamente de `transitions` efetivas.

    Args:
        layer: camada TernaryLayer a atualizar.
        counters: CounterManager com os contadores acumulados.
        layer_idx: índice da camada no CounterManager.
        neuron_idx: índice do neurônio que disparou.
        delta_sign: DEPRECIADO, ignorado.
        threshold: limiar mínimo. Se None, usa max(1, mean(|C|)).

    Returns:
        Dict com métricas: selected, attempts, blocked, transitions.
    """
    input_dim = layer.input_dim
    C = counters.weight_counters[layer_idx][neuron_idx].copy()  # (input_dim,)

    if threshold is None:
        threshold = max(1, np.mean(np.abs(C)))

    mask = np.abs(C) > threshold
    selected = int(np.sum(mask))

    # Direção de descida: −sign(C)
    transitions = 0
    blocked = 0
    signs = np.sign(C[mask])
    for i, col_idx in enumerate(np.where(mask)[0]):
        step = -int(signs[i])  # [012] descida sobre gradiente acumulado
        if step != 0:
            w_old = int(layer.weights[neuron_idx, col_idx])
            w_new = int(np.clip(w_old + step, -1, 1))
            if w_new == w_old:
                blocked += 1
            else:
                layer.weights[neuron_idx, col_idx] = np.int8(w_new)
                transitions += 1

    # Reset dos contadores atualizados (error-feedback): consome mesmo
    # tentativas bloqueadas (saturação), preservando não-selecionados.
    counters.reset_selected(layer_idx,
                            np.array([neuron_idx] * selected),
                            np.where(mask)[0])

    return {
        "updated": transitions,  # legado: transições efetivas
        "selected": selected,
        "attempts": selected,
        "blocked": blocked,
        "transitions": transitions,
        "skipped": input_dim - selected,
        "total": input_dim,
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# B2: Regra por camada (Top-K)
# ---------------------------------------------------------------------------
def apply_update_layer_topk(
    layer: TernaryLayer,
    counters: CounterManager,
    layer_idx: int,
    k: int,
    delta_sign: int = 1,
) -> Dict:
    """
    Regra B2: atualiza K conexões com maior |C_ij| em uma camada.

    [012] Direção: W <- clip(W − sign(C)); delta_sign ignorado.
    [012] K <= 0 → nenhuma seleção. Distingue bloqueios de transições.

    Args:
        layer: camada TernaryLayer a atualizar.
        counters: CounterManager com os contadores acumulados.
        layer_idx: índice da camada.
        k: número de conexões a atualizar.
        delta_sign: DEPRECIADO, ignorado.

    Returns:
        Dict com métricas da atualização.
    """
    weight_counters = counters.weight_counters[layer_idx]

    # Selecionar top-k
    row_idx, col_idx = counters.get_top_k_weights(layer_idx, k)

    # Direção [012]: −sign(C) (descida sobre gradiente acumulado)
    C_vals = weight_counters[row_idx, col_idx] if len(row_idx) else np.array([], dtype=np.int64)
    signs = np.sign(C_vals).astype(np.int8)
    update_signs = -signs

    # Atualizar pesos
    updated = 0
    blocked = 0
    for i in range(len(row_idx)):
        if update_signs[i] != 0:
            r, c = int(row_idx[i]), int(col_idx[i])
            w_old = int(layer.weights[r, c])
            w_new = int(np.clip(w_old + int(update_signs[i]), -1, 1))
            if w_new == w_old:
                blocked += 1
            else:
                layer.weights[r, c] = np.int8(w_new)
                updated += 1

    # Reset contadores atualizados
    counters.reset_selected(layer_idx, row_idx, col_idx)

    return {
        "updated": updated,
        "selected": len(row_idx),
        "attempts": len(row_idx),
        "blocked": blocked,
        "transitions": updated,
        "k": k,
    }


# ---------------------------------------------------------------------------
# B3: Regra por rede (Top-K por camada simultaneamente)
# ---------------------------------------------------------------------------
def apply_update_network_topk(
    net: TernaryNet,
    counters: CounterManager,
    k_per_layer: int,
    delta_sign: int = 1,
) -> Dict:
    """
    Regra B3: aplica Top-K em cada camada simultaneamente.

    [012] Direção: W <- clip(W − sign(C)); delta_sign ignorado.

    Total de atualizações: N_layers × K.

    Args:
        net: rede TernaryNet completa.
        counters: CounterManager com contadores de todas as camadas.
        k_per_layer: número de conexões a atualizar por camada.
        delta_sign: DEPRECIADO, ignorado.

    Returns:
        Dict com métricas por camada e totais.
    """
    total_updated = 0
    layer_results = []

    for layer_idx, layer in enumerate(net.layers):
        result = apply_update_layer_topk(
            layer, counters, layer_idx, k_per_layer, delta_sign
        )
        result["layer"] = layer_idx
        layer_results.append(result)
        total_updated += result["updated"]

    return {
        "total_updated": total_updated,
        "total_selected": sum(r["selected"] for r in layer_results),
        "num_layers": len(net.layers),
        "k_per_layer": k_per_layer,
        "layer_results": layer_results,
    }


# ---------------------------------------------------------------------------
# Utilitário: propagação de sinais para contadores
# ---------------------------------------------------------------------------
def compute_counter_delta(
    layer: TernaryLayer,
    output_activation: np.ndarray,
    input_activation: np.ndarray,
) -> np.ndarray:
    """
    Computa o delta para acumular nos contadores.

    delta_ij = output_activation_i × input_activation_j

    Args:
        layer: camada ativa.
        output_activation: saída da camada (batch, output_dim).
        input_activation: entrada da camada (batch, input_dim).

    Returns:
        Delta por conexão, shape (output_dim, input_dim).
    """
    # Outer product: (batch, output_dim)^T × (batch, input_dim)
    # = (output_dim, batch) × (batch, input_dim) = (output_dim, input_dim)
    delta = output_activation.T @ input_activation
    return delta


def compute_counter_delta_single(
    output_val: np.ndarray,
    input_val: np.ndarray,
) -> np.ndarray:
    """
    Versão single-sample de compute_counter_delta.

    Args:
        output_val: saída do neurônio, shape (output_dim,) ou (1, output_dim).
        input_val: entrada, shape (input_dim,) ou (1, input_dim).

    Returns:
        Delta, shape (output_dim, input_dim).
    """
    # Outer product: (output_dim, 1) @ (1, input_dim) = (output_dim, input_dim)
    return output_val.reshape(-1, 1) @ input_val.reshape(1, -1)


# ---------------------------------------------------------------------------
# Testes unitários
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testes — counter_rules.py")
    print("=" * 60)

    # --- Teste 1: update_weight ---
    print("\n[1] update_weight")

    assert update_weight(np.int8(0), 1) == 1, "0 + 1 deve ser 1"
    assert update_weight(np.int8(1), 1) == 1, "1 + 1 clipado deve ser 1"
    assert update_weight(np.int8(-1), -1) == -1, "-1 - 1 clipado deve ser -1"
    assert update_weight(np.int8(0), -1) == -1, "0 - 1 deve ser -1"
    assert update_weight(np.int8(1), -1) == 0, "1 - 1 deve ser 0"
    assert update_weight(np.int8(-1), 1) == 0, "-1 + 1 deve ser 0"
    print("   ✓ update_weight com clipping em {-1, 0, +1}")

    # --- Teste 2: CounterManager ---
    print("\n[2] CounterManager")

    shapes = [(128, 784), (64, 128), (10, 64)]
    cm = CounterManager(shapes)

    assert cm.num_layers == 3
    assert cm.weight_counters[0].shape == (128, 784)
    assert cm.weight_counters[0].dtype == np.int64
    assert cm.bias_counters[0].shape == (128,)
    print("   ✓ contadores inicializados corretamente")

    # accumulate
    delta = np.ones((128, 784), dtype=np.int64)
    cm.accumulate(0, delta)
    assert cm.weight_counters[0][0, 0] == 1, "accumulate falhou"
    print("   ✓ accumulate incrementa contadores")

    # get_neuron_counters
    neuron_C = cm.get_neuron_counters(0, 0)
    assert neuron_C.shape == (784,), f"Shape incorreto: {neuron_C.shape}"
    assert neuron_C[0] == 1, "Contador do neurônio 0 incorreto"
    print("   ✓ get_neuron_counters")

    # get_top_k_weights
    row_idx, col_idx = cm.get_top_k_weights(0, 5)
    assert len(row_idx) == 5, f"Esperado 5, got {len(row_idx)}"
    print("   ✓ get_top_k_weights retorna k índices")

    # get_top_k_weights > size
    row_idx_all, col_idx_all = cm.get_top_k_weights(0, 100000)
    assert len(row_idx_all) > 0, "Deve retornar todos quando k >= size"
    print("   ✓ get_top_k_weights com k >= size")

    # reset_selected
    cm.reset_selected(0, np.array([0]), np.array([0]))
    assert cm.weight_counters[0][0, 0] == 0, "reset_selected falhou"
    print("   ✓ reset_selected (error-feedback)")

    # reset_all
    cm.accumulate(0, np.ones((128, 784)))
    cm.reset_all()
    assert cm.weight_counters[0][0, 0] == 0
    assert cm.weight_counters[1][0, 0] == 0
    print("   ✓ reset_all")

    # get_layer_stats
    stats = cm.get_layer_stats(0)
    assert stats["layer"] == 0
    assert stats["shape"] == [128, 784]
    print(f"   ✓ get_layer_stats: {stats}")

    # --- Teste 3: apply_update_neuron (B1) ---
    print("\n[3] apply_update_neuron (B1)")

    layer = TernaryLayer(784, 128, seed=42)
    cm_b1 = CounterManager([(128, 784)])

    # Acumular contadores positivos
    cm_b1.accumulate(0, np.ones((128, 784), dtype=np.int64))

    # Aplicar B1 no neurônio 0
    # [012] C>0 acumula gradiente ⇒ descida W −= sign(C) (pesos diminuem).
    w_before = layer.weights[0].copy()
    result = apply_update_neuron(layer, cm_b1, 0, 0, delta_sign=1, threshold=0.5)
    assert result["selected"] == 784, f"Esperado 784 selecionados, got {result['selected']}"
    assert result["total"] == 784
    assert result["transitions"] + result["blocked"] == result["selected"]
    print(f"   ✓ B1: {result['selected']}/{result['total']} selecionados, "
          f"{result['transitions']} transições, {result['blocked']} bloqueios")
    # Direção de descida: nenhum peso deve ter aumentado
    assert np.all(layer.weights[0].astype(np.int16) <= w_before.astype(np.int16)), \
        "B1 deve mover pesos para baixo quando C>0 (W − sign(C))"

    # Verificar que pesos mudaram
    w_after = layer.weights[0]
    assert np.any(w_after != 0), "Pesos deveriam ter mudado"
    print("   ✓ pesos atualizados corretamente")

    # Verificar contadores resetados
    assert cm_b1.weight_counters[0][0].sum() == 0, "Contadores não foram resetados"
    print("   ✓ contadores resetados (error-feedback)")

    # --- Teste 4: apply_update_layer_topk (B2) ---
    print("\n[4] apply_update_layer_topk (B2)")

    layer_b2 = TernaryLayer(784, 128, seed=42)
    cm_b2 = CounterManager([(128, 784)])

    # Acumular todos os contadores
    cm_b2.accumulate(0, np.ones((128, 784), dtype=np.int64))

    k = 10
    result = apply_update_layer_topk(layer_b2, cm_b2, 0, k, delta_sign=1)
    assert result["selected"] == k, f"Esperado {k} selecionados, got {result['selected']}"
    print(f"   ✓ B2: {result['selected']} conexões selecionadas (k={k})")

    # Verificar reset: os contadores das conexões selecionadas foram zerados
    # (os demais permanecem com valor 1)
    non_zero_after = np.sum(cm_b2.weight_counters[0] != 0)
    expected_non_zero = cm_b2.weight_counters[0].size - k
    assert non_zero_after == expected_non_zero, \
        f"Esperado {expected_non_zero} não-nulos, got {non_zero_after}"
    print("   ✓ contadores resetados (k zerados, demais mantidos)")

    # --- Teste 5: apply_update_network_topk (B3) ---
    print("\n[5] apply_update_network_topk (B3)")

    net_b3 = TernaryNet(seed=42)
    cm_b3 = CounterManager([(128, 784), (64, 128), (10, 64)])

    # Acumular em todas as camadas
    for i in range(3):
        shape = cm_b3.layer_shapes[i]
        cm_b3.accumulate(i, np.ones(shape, dtype=np.int64))

    k_per_layer = 5
    result = apply_update_network_topk(net_b3, cm_b3, k_per_layer, delta_sign=1)
    assert result["num_layers"] == 3
    assert result["k_per_layer"] == k_per_layer
    expected_total = k_per_layer * 3
    # [012] selecionados = k*camadas; transições podem ser menores (bloqueio)
    assert result["total_selected"] == expected_total, \
        f"Esperado {expected_total} selecionados, got {result['total_selected']}"
    print(f"   ✓ B3: {result['total_selected']} selecionados, "
          f"{result['total_updated']} transições em 3 camadas")
    print(f"       {result['layer_results']}")

    # --- Teste 6: compute_counter_delta ---
    print("\n[6] compute_counter_delta")

    batch = 32
    input_act = np.random.randn(batch, 784).astype(np.float32)
    output_act = np.random.randn(batch, 128).astype(np.float32)

    delta = compute_counter_delta(None, output_act, input_act)
    assert delta.shape == (128, 784), f"Shape incorreto: {delta.shape}"
    print(f"   ✓ delta shape: {delta.shape}")

    # Single sample
    delta_single = compute_counter_delta_single(
        output_act[0], input_act[0]
    )
    assert delta_single.shape == (128, 784), f"Single shape incorreto: {delta_single.shape}"
    print(f"   ✓ single sample delta shape: {delta_single.shape}")

    # --- Teste 7: Integração B1 com rede real ---
    print("\n[7] Integração B1 com rede real")

    from src.utils import load_mnist
    train_data, train_labels, _, _ = load_mnist(normalize=False)

    net_int = TernaryNet(seed=42)
    cm_int = CounterManager([
        (128, 784), (64, 128), (10, 64)
    ])

    # Forward em uma amostra
    x = train_data[0].astype(np.float32).reshape(1, -1)
    out = net_int.forward(x)

    # Acumular deltas na primeira camada: output(1,128) × input(1,784)^T
    # Simplificação: se output > 0, acumular input como delta
    mask = (out > 0).astype(np.int64)  # (1, 128) — na verdade é (1, 10)
    # A saída da rede é (1, 10), mas queremos o delta da camada 0
    # Forward manual camada por camada para obter activations intermediárias
    h0 = net_int.layers[0].forward(x)  # (1, 128)
    delta_w0 = h0.T @ x  # (128, 1) × (1, 784) → (128, 784)

    # Acumular na primeira camada
    cm_int.accumulate(0, delta_w0.astype(np.int64))

    # Verificar contadores
    assert cm_int.weight_counters[0].sum() > 0, "Contadores não acumularam"
    print(f"   ✓ contadores acumulados: sum={cm_int.weight_counters[0].sum()}")

    # --- Teste 8: Reprodutibilidade ---
    print("\n[8] Reprodutibilidade")

    cm_r1 = CounterManager([(10, 5)])
    cm_r1.accumulate(0, np.ones((10, 5), dtype=np.int64) * 3)
    r1_row, r1_col = cm_r1.get_top_k_weights(0, 3)

    cm_r2 = CounterManager([(10, 5)])
    cm_r2.accumulate(0, np.ones((10, 5), dtype=np.int64) * 3)
    r2_row, r2_col = cm_r2.get_top_k_weights(0, 3)

    assert np.array_equal(r1_row, r2_row), "Top-K não é reprodutível"
    assert np.array_equal(r1_col, r2_col), "Top-K não é reprodutível"
    print("   ✓ top-k reprodutível")

    print("\n" + "=" * 60)
    print("Todos os testes passaram!")
    print("=" * 60)
