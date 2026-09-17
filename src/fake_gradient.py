"""
[003] Fake gradient triangular e quantização

Implementa o "fake gradient" (surrogate gradient) usado pela Rede B
como substituto da derivada (indefinida) da função degrau.

Fluxo:
    z (ativação) → triangular_fake_grad(z) → quantize_fake_grad() → Δ
    → propagate_delta() para camada anterior

Uso:
    from src.fake_gradient import triangular_fake_grad, quantize_fake_grad, propagate_delta
"""

import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Fake gradient triangular
# ---------------------------------------------------------------------------
def triangular_fake_grad(
    z: np.ndarray,
    scale: float = 1.0,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Função fake gradient triangular centrada em zero.

    Forma:  f(z) = scale * max(0, 1 - |z| / radius)

    - Pico em z=0: f(0) = scale
    - Base: f(z) = 0 para |z| >= radius
    - Linear entre 0 e radius (decrescente)

    Args:
        z: ativações do neurônio (input contínuo).
        scale: altura do pico (hiperparâmetro ≈ lr discreto).
        radius: largura da base — quanto longe de zero o gradiente é não-nulo.

    Returns:
        Array numpy com o mesmo shape de z, valores em [0, scale].
    """
    z = np.asarray(z, dtype=np.float64)
    result = scale * np.maximum(0.0, 1.0 - np.abs(z) / radius)
    return result


def triangular_fake_grad_derivative(
    z: np.ndarray,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Derivada analítica da função triangular (para debugging/monitoramento).

    f'(z) = -scale/radius * sign(z)  para 0 < |z| < radius
    f'(z) = 0                         para |z| >= radius

    Nota: esta função NÃO é usada no treino — só para análise.
    """
    z = np.asarray(z, dtype=np.float64)
    mask = (np.abs(z) < radius) & (np.abs(z) > 0)
    result = np.zeros_like(z)
    result[mask] = -np.sign(z[mask]) / radius
    return result


# ---------------------------------------------------------------------------
# Quantização
# ---------------------------------------------------------------------------
def quantize_fake_grad(
    z: np.ndarray,
    levels: int = 5,
) -> np.ndarray:
    """
    Quantiza valores contínuos para um conjunto discreto de inteiros.

    Conjuntos suportados:
        levels=3 → {-1, 0, +1}   (SignSGD-like)
        levels=5 → {-2, -1, 0, +1, +2}
        levels=7 → {-3, -2, -1, 0, +1, +2, +3}

    Mapeamento:
        - Valor 0 → 0
        - Valores positivos → 1, 2, 3, ... (escalonados por bin)
        - Valores negativos → -1, -2, -3, ...

    Args:
        z: array de valores contínuos (fake gradient ou gradiente).
        levels: número de níveis discretos (deve ser ímpar: 2k+1).

    Returns:
        Array numpy com inteiros quantizados.
    """
    z = np.asarray(z, dtype=np.float64)

    if levels == 3:
        # {-1, 0, +1} — equivalente a SignSGD
        return np.sign(z).astype(np.int64)

    if levels == 5:
        # {-2, -1, 0, +1, +2}
        # Normaliza para [-1, 1], multiplica por 2, round
        normalized = np.clip(z, -1.0, 1.0)
        quantized = np.round(normalized * 2).astype(np.int64)
        return quantized

    if levels == 7:
        # {-3, -2, -1, 0, +1, +2, +3}
        normalized = np.clip(z, -1.0, 1.0)
        quantized = np.round(normalized * 3).astype(np.int64)
        return quantized

    # Genérico: levels = 2k+1
    k = (levels - 1) // 2
    if k <= 0:
        raise ValueError(f"levels deve ser ímpar e >= 3, got {levels}")
    normalized = np.clip(z, -1.0, 1.0)
    quantized = np.round(normalized * k).astype(np.int64)
    return quantized


def quantize_to_sign(z: np.ndarray) -> np.ndarray:
    """
    Quantização pura ao sinal: {-1, 0, +1}.
    Equivalente a quantize_fake_grad(z, levels=3).
    """
    return np.sign(z).astype(np.int64)


# ---------------------------------------------------------------------------
# Propagação de δ (regra da cadeia discreta)
# ---------------------------------------------------------------------------
def propagate_delta(
    delta_next: np.ndarray,
    weights: np.ndarray,
    fake_grad_val: float = 1.0,
) -> np.ndarray:
    """
    Aplica a regra da cadeia discreta para propagar δ para a camada anterior.

    δ_j^(l) = fake_grad_val · Σ_k w_jk^(l+1) · δ_k^(l+1)

    Args:
        delta_next: δ da camada seguinte, shape (batch_size, next_units).
        weights: pesos da camada atual, shape (current_units, next_units).
                 w[j, k] = peso da conexão j→k entre camadas.
        fake_grad_val: valor do fake gradient local (escalar ou array broadcast).

    Returns:
        δ da camada atual, shape (batch_size, current_units).
    """
    # δ_current = δ_next @ W^T  →  (batch, next) @ (next, current) = (batch, current)
    delta = delta_next @ weights.T  # (batch_size, current_units)
    delta = delta * fake_grad_val
    return delta


def propagate_delta_with_fake(
    delta_next: np.ndarray,
    weights: np.ndarray,
    z_local: np.ndarray,
    scale: float = 1.0,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Propagação completa: calcula fake gradient local e aplica regra da cadeia.

    Combina propagate_delta com o cálculo do fake gradient triangular
    baseado nas ativações locais z.

    Args:
        delta_next: δ da camada seguinte, shape (batch_size, next_units).
        weights: pesos da camada atual, shape (current_units, next_units).
        z_local: ativações (antes da degrau) da camada atual, shape (batch_size, current_units).
        scale: altura do pico do fake gradient.
        radius: largura da base do fake gradient.

    Returns:
        δ da camada atual, shape (batch_size, current_units).
    """
    # Calcula fake gradient local por neurônio
    # z_local é (batch_size, current_units) → fake_grad é (batch_size, current_units)
    fake_grad = triangular_fake_grad(z_local, scale=scale, radius=radius)
    # Média por neurônio ao longo do batch (para obter um escalar por neurônio)
    fake_grad_mean = fake_grad.mean(axis=0)  # (current_units,)

    # Propaga com o fake gradient médio
    return propagate_delta(delta_next, weights, fake_grad_val=fake_grad_mean)


def propagate_delta_with_fake_quantized(
    delta_next: np.ndarray,
    weights: np.ndarray,
    z_local: np.ndarray,
    scale: float = 1.0,
    radius: float = 1.0,
    levels: int = 5,
) -> np.ndarray:
    """
    [012] Propagação com surrogate QUANTIZADO (coerente com a saída).

    Calcula o triangular contínuo, quantiza com `levels` (mesma regra da
    saída) e aplica a regra da cadeia. Evita que o gate contínuo das
    ocultas divirja da representação inteira usada no acúmulo.

    Args:
        delta_next: δ da camada seguinte, shape (batch, next_units).
        weights: pesos, shape (current_units, next_units).
        z_local: pré-ativações locais, shape (batch, current_units).
        scale, radius: parâmetros do triangular.
        levels: níveis de quantização (default 5 → {-2..+2}).

    Returns:
        δ da camada atual, shape (batch, current_units).
    """
    fake_cont = triangular_fake_grad(z_local, scale=scale, radius=radius)
    q_fake = quantize_fake_grad(fake_cont, levels=levels).astype(np.float64)
    # Média por neurônio ao longo do batch (equivale ao local p/ 1 amostra)
    q_fake_mean = q_fake.mean(axis=0)
    return propagate_delta(delta_next, weights, fake_grad_val=q_fake_mean)


# ---------------------------------------------------------------------------
# Raio adaptativo ao fan-in (task 010)
# ---------------------------------------------------------------------------
def get_adaptive_radius(input_dim: int, scale_factor: float = 1.0) -> float:
    """
    Calcula raio adaptativo proporcional a sqrt(fan-in).

    radius_l = scale_factor * sqrt(input_dim_l)

    A variância da soma de N variáveis independentes escala com sqrt(N).
    Casar a base do triângulo com sqrt(N) mantém os neurônios dentro
    do cone de aprendizado do fake gradient.

    Args:
        input_dim: fan-in da camada (número de entradas).
        scale_factor: c em [0.5, 1.0], ajusta largura do cone.

    Returns:
        Raio adaptativo (float > 0).
    """
    return float(scale_factor * np.sqrt(input_dim))


def get_layer_radii(layer_dims: List[int], scale_factor: float = 1.0) -> List[float]:
    """
    Retorna lista de raios adaptativos para cada camada.

    Args:
        layer_dims: lista de input_dim por camada, ex. [784, 128, 64].
        scale_factor: fator c global.

    Returns:
        Lista de raios, um por camada.
    """
    return [get_adaptive_radius(d, scale_factor) for d in layer_dims]


def adaptive_triangular_fake_grad(
    z: np.ndarray,
    input_dim: int,
    scale: float = 1.0,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Fake gradient triangular com raio adaptativo ao fan-in.

    Wrapper de triangular_fake_grad que calcula radius automaticamente.

    Args:
        z: ativações pré-degrau.
        input_dim: fan-in da camada.
        scale: altura do pico (lr discreto).
        scale_factor: fator c para radius = c * sqrt(input_dim).

    Returns:
        Fake gradient com base adaptativa.
    """
    radius = get_adaptive_radius(input_dim, scale_factor)
    return triangular_fake_grad(z, scale=scale, radius=radius)


# ---------------------------------------------------------------------------
# Acumulador de contadores (helper para Rede B)
# ---------------------------------------------------------------------------
class CounterAccumulator:
    """
    Acumulador de contadores inteiros por conexão.

    Cada peso w[j, k] tem seu próprio contador C[j, k] que acumula
    incrementos Δ_ij ao longo de várias amostras.

    Uso:
        acc = CounterAccumulator(shape=(128, 784))
        acc.update(delta_grads)  # shape (batch, 128, 784)
        counters = acc.get_counters()
        acc.reset_selected(indices)
    """

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype = np.int64):
        self.shape = shape
        self.counters = np.zeros(shape, dtype=dtype)

    def update(self, increments: np.ndarray) -> None:
        """
        Acumula incrementos nos contadores.

        Args:
            increments: array com o mesmo shape de self.counters.
        """
        self.counters += increments.astype(self.counters.dtype, casting="unsafe")

    def get_counters(self) -> np.ndarray:
        """Retorna cópia dos contadores atuais."""
        return self.counters.copy()

    def reset_selected(self, indices: np.ndarray) -> None:
        """
        Zera somente os contadores nas posições indicadas.
        (política error-feedback: resíduo não-selecionado persiste)

        Args:
            indices: array de índices booleanos ou inteiros.
        """
        if indices.dtype == bool:
            self.counters[indices] = 0
        else:
            self.counters[indices] = 0

    def reset_all(self) -> None:
        """Zera todos os contadores."""
        self.counters.fill(0)

    def normalize(self) -> None:
        """[012] Normaliza preservando dtype inteiro (round simétrico).

        Escala para que max|C| ≈ 1 via rint; mantém int64 (contrato inteiro).
        """
        max_abs = np.max(np.abs(self.counters))
        if max_abs > 0:
            scaled = np.rint(
                self.counters.astype(np.float64) / float(max_abs)
            ).astype(np.int64)
            self.counters = scaled.astype(self.counters.dtype, casting="unsafe")

    def top_k_indices(self, k: int) -> np.ndarray:
        """
        Retorna índices das K conexões com maior |C_ij| por linha (por neurônio).

        [012] k <= 0 → máscara vazia.

        Args:
            k: número de conexões a selecionar por linha.

        Returns:
            Array de índices booleanos do mesmo shape que counters.
        """
        abs_count = np.abs(self.counters)
        shape = self.counters.shape

        if k is None or k <= 0:
            return np.zeros_like(self.counters, dtype=bool)

        if k >= shape[-1]:
            return np.ones_like(self.counters, dtype=bool)

        # Para cada linha, pega top-k por argpartition
        mask = np.zeros_like(abs_count, dtype=bool)
        for i in range(shape[0]):
            flat = abs_count[i]
            if k >= flat.size:
                mask[i] = True
                continue
            top_k_flat = np.argpartition(flat, -k)[-k:]
            mask[i, top_k_flat] = True
        return mask

    def __repr__(self) -> str:
        return (
            f"CounterAccumulator(shape={self.shape}, "
            f"sum={self.counters.sum()}, "
            f"max_abs={np.max(np.abs(self.counters))})"
        )


# ---------------------------------------------------------------------------
# Testes unitários
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testes — fake_gradient.py")
    print("=" * 60)

    # --- Teste 0: raio adaptativo (task 010) ---
    print("\n[0] get_adaptive_radius / adaptive_triangular_fake_grad")
    assert np.isclose(get_adaptive_radius(784, 1.0), 28.0), f"784→{get_adaptive_radius(784)}"
    assert np.isclose(get_adaptive_radius(128, 1.0), np.sqrt(128)), "128"
    assert np.isclose(get_adaptive_radius(64, 1.0), 8.0), "64"
    assert np.isclose(get_adaptive_radius(784, 0.5), 14.0), "c=0.5"
    radii = get_layer_radii([784, 128, 64], scale_factor=1.0)
    assert len(radii) == 3 and np.isclose(radii[0], 28.0)
    print(f"   ✓ radii [784,128,64] c=1.0 → {radii}")
    # adaptive wrapper deve equivaler a triangular com radius calculado
    z_tmp = np.array([0.0, 5.0, 30.0])
    a = adaptive_triangular_fake_grad(z_tmp, input_dim=784, scale=1.0, scale_factor=1.0)
    b = triangular_fake_grad(z_tmp, scale=1.0, radius=28.0)
    assert np.allclose(a, b), "adaptive wrapper diverge"
    print("   ✓ adaptive_triangular_fake_grad equivale a triangular com radius=28")

    # --- Teste 1: triangular_fake_grad ---
    print("\n[1] triangular_fake_grad")

    # Pico em z=0
    z_zero = np.array([0.0])
    result = triangular_fake_grad(z_zero, scale=2.0, radius=1.0)
    assert np.isclose(result[0], 2.0), f"Pico esperado 2.0, got {result[0]}"
    print("   ✓ pico em z=0")

    # Zero para |z| >= radius
    z_edge = np.array([1.0, -1.0, 2.0, -2.0])
    result = triangular_fake_grad(z_edge, scale=1.0, radius=1.0)
    assert np.allclose(result, [0.0, 0.0, 0.0, 0.0]), f"Esperado zero, got {result}"
    print("   ✓ zero para |z| >= radius")

    # Linear decrescente
    z_linear = np.array([0.0, 0.5, 0.8, 1.0])
    result = triangular_fake_grad(z_linear, scale=1.0, radius=1.0)
    expected = np.array([1.0, 0.5, 0.2, 0.0])
    assert np.allclose(result, expected), f"Esperado {expected}, got {result}"
    print("   ✓ linear decrescente")

    # Shape preservation
    z_2d = np.random.randn(32, 128)
    result = triangular_fake_grad(z_2d, scale=0.5, radius=2.0)
    assert result.shape == (32, 128), f"Shape incorreto: {result.shape}"
    print("   ✓ preservação de shape")

    # --- Teste 2: quantize_fake_grad ---
    print("\n[2] quantize_fake_grad")

    # levels=3 → {-1, 0, +1}
    z = np.array([-0.5, 0.0, 0.3, 1.5, -2.0])
    result = quantize_fake_grad(z, levels=3)
    expected = np.array([-1, 0, 1, 1, -1], dtype=np.int64)
    assert np.array_equal(result, expected), f"Esperado {expected}, got {result}"
    print("   ✓ levels=3 → {-1, 0, +1}")

    # levels=5 → {-2, -1, 0, +1, +2}
    z = np.array([-0.9, -0.3, 0.0, 0.3, 0.9])
    result = quantize_fake_grad(z, levels=5)
    expected = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    assert np.array_equal(result, expected), f"Esperado {expected}, got {result}"
    print("   ✓ levels=5 → {-2, -1, 0, +1, +2}")

    # Determinístico
    z = np.random.randn(1000)
    r1 = quantize_fake_grad(z, levels=5)
    r2 = quantize_fake_grad(z, levels=5)
    assert np.array_equal(r1, r2), "Quantização não é determinística!"
    print("   ✓ determinístico")

    # --- Teste 3: propagate_delta ---
    print("\n[3] propagate_delta")

    # Dimensões: batch=4, next_units=10, current_units=20
    batch, next_units, current_units = 4, 10, 20
    delta_next = np.random.randn(batch, next_units)
    weights = np.random.randn(current_units, next_units)

    delta = propagate_delta(delta_next, weights, fake_grad_val=1.0)
    assert delta.shape == (batch, current_units), f"Shape incorreto: {delta.shape}"
    print("   ✓ shape correto: (batch, current_units)")

    # Fake gradient escalar
    delta2 = propagate_delta(delta_next, weights, fake_grad_val=0.5)
    assert np.allclose(delta2, delta * 0.5), "Escala do fake grad incorreta"
    print("   ✓ escala correta com fake_grad_val")

    # --- Teste 4: propagate_delta_with_fake ---
    print("\n[4] propagate_delta_with_fake")

    z_local = np.random.randn(batch, current_units)
    delta3 = propagate_delta_with_fake(
        delta_next, weights, z_local, scale=1.0, radius=1.0
    )
    assert delta3.shape == (batch, current_units), f"Shape incorreto: {delta3.shape}"
    print("   ✓ shape correto")

    # --- Teste 5: CounterAccumulator ---
    print("\n[5] CounterAccumulator")

    acc = CounterAccumulator(shape=(3, 4))
    assert np.all(acc.get_counters() == 0), "Contadores não iniciam em zero"
    print("   ✓ inicia em zero")

    acc.update(np.ones((3, 4)))
    assert np.all(acc.get_counters() == 1), "Update falhou"
    print("   ✓ update soma corretamente")

    acc.update(np.ones((3, 4)) * 2)
    assert np.all(acc.get_counters() == 3), "Acumulação falhou"
    print("   ✓ acumulação correta")

    # reset_selected
    mask = np.array([[True, False, False, True],
                     [False, True, False, False],
                     [False, False, True, False]])
    acc.reset_selected(mask)
    expected = np.array([[0, 3, 3, 0],
                         [3, 0, 3, 3],
                         [3, 3, 0, 3]])
    assert np.array_equal(acc.get_counters(), expected), "reset_selected falhou"
    print("   ✓ reset_selected (error-feedback)")

    # reset_all
    acc.reset_all()
    assert np.all(acc.get_counters() == 0), "reset_all falhou"
    print("   ✓ reset_all")

    # normalize
    acc.update(np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]]))
    acc.normalize()
    # [012] preserva int64: max|C|→1, resto rounded (1/12→0, 6/12→0/1...)
    assert acc.get_counters().dtype == np.int64 or np.issubdtype(
        acc.get_counters().dtype, np.integer), "normalize quebrou dtype int"
    assert int(np.max(np.abs(acc.get_counters()))) == 1, "normalize falhou"
    print("   ✓ normalize (int-safe)")

    # top_k_indices (k por linha)
    acc2 = CounterAccumulator(shape=(2, 5))
    acc2.update(np.array([[1, 5, 2, 8, 3],
                          [4, 1, 7, 2, 6]]))
    top3 = acc2.top_k_indices(3)
    # 3 por linha × 2 linhas = 6 no total
    assert top3.sum() == 6, f"top_k retorna {top3.sum()} índices, esperado 6"
    # Os top-3 por linha devem ser:
    #   linha 0 → valores [1,5,2,8,3] → top-3 abs: {8,5,3} → indices [3,1,4]
    #   linha 1 → valores [4,1,7,2,6] → top-3 abs: {7,6,4} → indices [2,4,0]
    expected_top = np.array([[False, True, False, True, True],
                             [True, False, True, False, True]])
    assert np.array_equal(top3, expected_top), f"top_k incorreto: {top3}"
    print("   ✓ top_k_indices (k por linha)")

    print("\n" + "=" * 60)
    print("Todos os testes passaram!")
    print("=" * 60)
