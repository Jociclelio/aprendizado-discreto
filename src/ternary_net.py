"""
[004] Núcleo da Rede B — pesos ternários, bias inteiro, ativação degrau

Arquitetura: 784 → 128 → 64 → 10 (mesma topologia da Rede A)
Pesos: int8 em {-1, 0, +1}
Bias: int32 em [-N, N-1]
Ativação: degrau centrado em zero (1 se z > 0, 0 se z ≤ 0)

Uso:
    from src.ternary_net import TernaryNet
    net = TernaryNet()
    output = net.forward(input_data)  # shape (batch, 10)
"""

from pathlib import Path

import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Ativação degrau
# ---------------------------------------------------------------------------
def step_activation(z: np.ndarray) -> np.ndarray:
    """
    Função degrau centrada em zero.

    step(z) = 1 se z > 0, 0 se z ≤ 0

    Args:
        z: ativações pré-ativação.

    Returns:
        Array binário do mesmo shape.
    """
    return (z > 0).astype(np.float32)


def step_activation_with_threshold(z: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Versão da degrau com limiar configurável.

    step(z) = 1 se z > threshold, 0 se z ≤ threshold
    """
    return (z > threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# TernaryLayer
# ---------------------------------------------------------------------------
class TernaryLayer:
    """
    Camada individual com pesos ternários e bias inteiro.

    Forward: z = W @ x + b → step(z)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: Optional[int] = None,
    ):
        """
        Inicializa uma camada ternária.

        Args:
            input_dim: número de entradas da camada anterior.
            output_dim: número de neurônios de saída.
            seed: semente para reprodutibilidade.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # Pesos ternários: int8 em {-1, 0, +1}
        # Distribuição balanceada: ~1/3 para cada valor
        # Estratégia: gerar randn, quantizar: >0.5→+1, <−0.5→−1, else→0
        raw = rng.randn(output_dim, input_dim)
        self.weights = np.zeros((output_dim, input_dim), dtype=np.int8)
        self.weights[raw > 0.5] = 1
        self.weights[raw < -0.5] = -1
        # O resto já está em 0

        # Bias inteiro: int32, inicializado em 0
        # Faixa: b ∈ [-N, N-1] onde N = input_dim
        self.bias = np.zeros(output_dim, dtype=np.int32)

        # Cache para backprop (será usado pelas tasks seguintes)
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass da camada.

        Args:
            x: entradas, shape (batch, input_dim) ou (input_dim,)

        Returns:
            Saída após degrau, shape (batch, output_dim) ou (output_dim,)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single = True
        else:
            single = False

        # z = W @ x + b
        # W: (output_dim, input_dim), x: (batch, input_dim)
        # z: (batch, output_dim)
        z = x @ self.weights.T + self.bias  # (batch, output_dim)

        # Cache para futuras atualizações
        self._cache = {
            "x": x,
            "z_pre": z,
        }

        # Ativação degrau
        out = step_activation(z)

        if single:
            return out[0]

        return out

    def quantize_weights(self) -> None:
        """
        Garante que W ∈ {-1, 0, +1}.

        Quantiza qualquer peso contínuo para o valor ternário mais próximo.
        """
        w = self.weights.astype(np.float64)
        self.weights = np.round(w).astype(np.int8)
        # Garante limites
        self.weights = np.clip(self.weights, -1, 1)

    def update_weights(self, delta: np.ndarray) -> None:
        """
        Atualiza pesos com um delta (geralmente inteiro pequeno).

        Args:
            delta: incremento por conexão, shape (output_dim, input_dim).
        """
        w = self.weights.astype(np.int64) + delta
        self.weights = np.clip(w, -1, 1).astype(np.int8)

    def update_bias(self, delta: int) -> None:
        """
        Atualiza bias com um delta inteiro.

        Args:
            delta: incremento por neurônio.
        """
        self.bias += delta
        # Garante faixa [-input_dim, input_dim-1]
        max_bias = self.input_dim - 1
        self.bias = np.clip(self.bias, -self.input_dim, max_bias)

    def get_weights(self) -> np.ndarray:
        """Retorna cópia dos pesos."""
        return self.weights.copy()

    def get_biases(self) -> np.ndarray:
        """Retorna cópia dos biases."""
        return self.bias.copy()

    def get_stats(self) -> dict:
        """Retorna estatísticas da camada."""
        w = self.weights
        total = w.size
        positive = int(np.sum(w == 1))
        zero = int(np.sum(w == 0))
        negative = int(np.sum(w == -1))
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "total": total,
            "positive": positive,
            "zero": zero,
            "negative": negative,
            "ternary_fraction": round((positive + zero + negative) / total, 4),
            "bias_mean": float(np.mean(self.bias)),
            "bias_std": float(np.std(self.bias)),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"TernaryLayer({stats['input_dim']}→{stats['output_dim']}, "
            f"ternary={stats['ternary_fraction']})"
        )


# ---------------------------------------------------------------------------
# TernaryNet
# ---------------------------------------------------------------------------
class TernaryNet:
    """
    Rede ternária completa: 784 → 128 → 64 → 10

    Mesma topologia da Rede A, mas com pesos ternários e bias inteiro.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden1: int = 128,
        hidden2: int = 64,
        output_dim: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Inicializa a rede ternária.

        Args:
            input_dim: dimensão de entrada.
            hidden1: neurônios na primeira camada oculta.
            hidden2: neurônios na segunda camada oculta.
            output_dim: neurônios de saída.
            seed: semente para reprodutibilidade.
        """
        self.layers: List[TernaryLayer] = []

        if seed is not None:
            rng = np.random.RandomState(seed)
            seeds = rng.randint(0, 2**31, size=3)
        else:
            seeds = [None, None, None]

        # Camada 1: 784 → 128
        self.layers.append(TernaryLayer(input_dim, hidden1, seed=seeds[0]))
        # Camada 2: 128 → 64
        self.layers.append(TernaryLayer(hidden1, hidden2, seed=seeds[1]))
        # Camada 3: 64 → 10 (saída)
        self.layers.append(TernaryLayer(hidden2, output_dim, seed=seeds[2]))

        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = output_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass completo pela rede.

        Args:
            x: entradas, shape (batch, 784) ou (784,)

        Returns:
            Saída binária, shape (batch, 10) ou (10,)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single = True
        else:
            single = False

        # Passa por todas as camadas
        out = x
        for layer in self.layers:
            out = layer.forward(out)

        if single:
            return out[0]

        return out

    def forward_with_scores(self, x: np.ndarray):
        """
        [012] Forward unificado: expõe scores e ativações sem duplicar lógica.

        Contrato [012]:
            z_l = W_l @ x_{l-1} + b_l (pré-ativação, inclui bias int32),
            a_l = step(z_l) com step(z) = 1 se z > 0 else 0,
            scores = z_out (pré-ativação da camada de saída, ANTES do degrau),
            a_out = step(z_out) (saída binária).

        Todas as camadas ocultas aplicam o degrau. `evaluate()` deve usar
        os scores desta função (argmax), nunca reimplementar as camadas.

        Args:
            x: entradas, shape (batch, input_dim) ou (input_dim,).

        Returns:
            (scores, binary_out, pre_activations, activations):
            - scores: (batch, output_dim) float — z da última camada.
            - binary_out: (batch, output_dim) float {0,1} — step(scores).
            - pre_activations: lista de z_l por camada, cada (batch, dim_l).
            - activations: lista [x, a_1, ..., a_L], cada (batch, dim).
              Para input 1-D, retorna formas 1-D em scores/binary_out.
        """
        single = (x.ndim == 1)
        if single:
            x = x.reshape(1, -1)

        pre_activations = []
        activations = [x.astype(np.float32)]
        out = x.astype(np.float32)
        for layer in self.layers:
            z = out @ layer.weights.T + layer.bias  # (batch, out_dim)
            pre_activations.append(z)
            layer._cache = {"x": out, "z_pre": z}
            out = step_activation(z)
            activations.append(out)

        scores = pre_activations[-1]
        binary_out = activations[-1]
        if single:
            return scores[0], binary_out[0], pre_activations, activations
        return scores, binary_out, pre_activations, activations

    def firing_rates(self, activations) -> list:
        """[012] Taxa de disparo por camada (média de a_l). Anti-colapso."""
        return [float(np.mean(a)) for a in activations[1:]]

    def get_weights(self) -> List[np.ndarray]:
        """Retorna lista de matrizes de peso."""
        return [layer.get_weights() for layer in self.layers]

    def get_biases(self) -> List[np.ndarray]:
        """Retorna lista de vetores de bias."""
        return [layer.get_biases() for layer in self.layers]

    def get_all_weights_flat(self) -> np.ndarray:
        """Retorna todos os pesos concatenados em um único array."""
        return np.concatenate([w.flatten() for w in self.get_weights()])

    def get_stats(self) -> dict:
        """Retorna estatísticas de toda a rede."""
        total_weights = 0
        total_positive = 0
        total_zero = 0
        total_negative = 0

        for layer in self.layers:
            w = layer.weights
            total_weights += w.size
            total_positive += int(np.sum(w == 1))
            total_zero += int(np.sum(w == 0))
            total_negative += int(np.sum(w == -1))

        ternary_frac = (total_positive + total_zero + total_negative) / total_weights

        return {
            "architecture": f"{self.input_dim}→{self.hidden1}→{self.hidden2}→{self.output_dim}",
            "total_weights": total_weights,
            "total_positive": total_positive,
            "total_zero": total_zero,
            "total_negative": total_negative,
            "ternary_fraction": round(ternary_frac, 4),
            "total_bias": sum(len(l.bias) for l in self.layers),
            "layer_stats": [l.get_stats() for l in self.layers],
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"TernaryNet({stats['architecture']}, "
            f"weights={stats['total_weights']}, "
            f"ternary={stats['ternary_fraction']})"
        )


# ---------------------------------------------------------------------------
# Loss functions para Rede B (saída degrau binária)
# ---------------------------------------------------------------------------
def ternary_binary_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [012] Perda L1 média por neurônio sobre bits (NÃO é BCE).

    loss = mean(|pred - target|) sobre (batch × 10).
    Nota anti-colapso: uma saída toda-zero tem L1 ≈ 0.1 em 10 classes
    (1/10); loss pequena isolada pode esconder colapso — monitorar
    também taxa de disparos por camada (ver TernaryNet.firing_rates).

    Para saída degrau binária (10 neurônios, cada um decide binariamente
    se a imagem pertence àquela classe).

    Args:
        predictions: saídas da rede, shape (batch, 10), valores em {0, 1}.
        targets: labels verdadeiros, shape (batch,) ou (batch, 10).
                 Se shape (batch,), será convertido para one-hot.

    Returns:
        (loss_scalar, gradient)
        - loss_scalar: perda média por amostra
        - gradient: gradiente da perda em relação às predições
    """
    if targets.ndim == 1:
        # Converte para one-hot
        batch_size = predictions.shape[0]
        one_hot = np.zeros_like(predictions)
        one_hot[np.arange(batch_size), targets.astype(int)] = 1
        targets = one_hot

    # Erro absoluto por neurônio
    error = predictions - targets  # (batch, 10)
    loss = np.mean(np.abs(error))

    # Gradiente: sign(error) — derivada do L1
    gradient = np.sign(error) / error.size

    return loss, gradient


def ternary_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Calcula acurácia para saída degrau binária.

    Conta quantas amostras têm o vetor de saída binária idêntico ao one-hot alvo.

    Args:
        predictions: saídas da rede, shape (batch, 10).
        targets: labels verdadeiros, shape (batch,) ou (batch, 10).

    Returns:
        Acurácia (0.0 a 1.0).
    """
    if targets.ndim == 1:
        batch_size = predictions.shape[0]
        one_hot = np.zeros_like(predictions)
        one_hot[np.arange(batch_size), targets.astype(int)] = 1
        targets = one_hot

    # Acerto = predição idêntica ao one-hot
    correct = np.sum(np.all(predictions == targets, axis=1))
    return correct / predictions.shape[0]


# ---------------------------------------------------------------------------
# Testes unitários
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testes — ternary_net.py")
    print("=" * 60)

    # --- Teste 1: step_activation ---
    print("\n[1] step_activation")

    z = np.array([-2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0])
    out = step_activation(z)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert np.allclose(out, expected), f"Esperado {expected}, got {out}"
    print("   ✓ degrau centrado em zero")

    # --- Teste 2: TernaryLayer ---
    print("\n[2] TernaryLayer")

    layer = TernaryLayer(784, 128, seed=42)
    assert layer.weights.shape == (128, 784), "Shape dos pesos incorreto"
    assert layer.weights.dtype == np.int8, "Pesos não são int8"
    assert np.all(np.isin(layer.weights, [-1, 0, 1])), "Pesos fora de {-1, 0, +1}"
    assert layer.bias.shape == (128,), "Shape do bias incorreto"
    assert layer.bias.dtype == np.int32, "Bias não é int32"
    print("   ✓ pesos int8 em {-1, 0, +1}")
    print("   ✓ bias int32")

    # Forward com input single
    x = np.random.randn(784).astype(np.float32)
    out = layer.forward(x)
    assert out.shape == (128,), f"Shape incorreto: {out.shape}"
    assert np.all(np.isin(out, [0.0, 1.0])), "Saída não é binária"
    print("   ✓ forward single input")

    # Forward com batch
    x_batch = np.random.randn(32, 784).astype(np.float32)
    out_batch = layer.forward(x_batch)
    assert out_batch.shape == (32, 128), f"Shape incorreto: {out_batch.shape}"
    print("   ✓ forward batch")

    # quantize_weights
    layer.weights = (layer.weights.astype(np.float64) + 0.3).astype(np.int8)
    layer.quantize_weights()
    assert np.all(np.isin(layer.weights, [-1, 0, 1])), "quantize_weights falhou"
    print("   ✓ quantize_weights")

    # update_weights
    layer.update_weights(np.ones((128, 784), dtype=np.int64))
    assert np.all(layer.weights <= 1), "update_weights estourou limite"
    print("   ✓ update_weights com clipping")

    # update_bias
    layer.update_bias(5)
    assert np.all(layer.bias >= -784) and np.all(layer.bias <= 783), "bias fora da faixa"
    print("   ✓ update_bias com faixa correta")

    # get_stats
    stats = layer.get_stats()
    assert "total" in stats and "ternary_fraction" in stats
    assert stats["ternary_fraction"] == 1.0, "Todos os pesos devem ser ternários"
    print(f"   ✓ stats: {stats['total']} pesos, ternary={stats['ternary_fraction']}")

    # --- Teste 3: TernaryNet ---
    print("\n[3] TernaryNet")

    net = TernaryNet(seed=42)
    assert len(net.layers) == 3, "Deve ter 3 camadas"
    assert net.layers[0].input_dim == 784
    assert net.layers[0].output_dim == 128
    assert net.layers[1].input_dim == 128
    assert net.layers[1].output_dim == 64
    assert net.layers[2].input_dim == 64
    assert net.layers[2].output_dim == 10
    print("   ✓ arquitetura 784→128→64→10")

    # Forward pass
    x = np.random.randn(1, 784).astype(np.float32)
    out = net.forward(x)
    assert out.shape == (1, 10), f"Shape incorreto: {out.shape}"
    assert np.all(np.isin(out, [0.0, 1.0])), "Saída não é binária"
    print("   ✓ forward (1, 784) → (1, 10)")

    # Forward single
    x_single = np.random.randn(784).astype(np.float32)
    out_single = net.forward(x_single)
    assert out_single.shape == (10,), f"Shape incorreto: {out_single.shape}"
    print("   ✓ forward single (784,) → (10,)")

    # Forward batch
    x_batch = np.random.randn(64, 784).astype(np.float32)
    out_batch = net.forward(x_batch)
    assert out_batch.shape == (64, 10), f"Shape incorreto: {out_batch.shape}"
    print("   ✓ forward batch (64, 784) → (64, 10)")

    # get_weights / get_biases
    weights = net.get_weights()
    biases = net.get_biases()
    assert len(weights) == 3
    assert weights[0].shape == (128, 784)
    assert weights[1].shape == (64, 128)
    assert weights[2].shape == (10, 64)
    print("   ✓ get_weights / get_biases")

    # get_stats
    stats = net.get_stats()
    print(f"   ✓ stats: {stats['architecture']}, {stats['total_weights']} pesos")
    print(f"       +{stats['total_positive']} 0:{stats['total_zero']} -{stats['total_negative']}")

    # --- Teste 4: Loss e acurácia ---
    print("\n[4] Loss e acurácia")

    # Predições e targets
    preds = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    targets = np.array([0, 1])

    loss, grad = ternary_binary_loss(preds, targets)
    assert loss >= 0.0, "Loss deve ser não-negativo"
    print(f"   ✓ loss = {loss:.4f}")

    acc = ternary_accuracy(preds, targets)
    assert acc == 1.0, "Predições perfeitas devem dar acurácia 1.0"
    print(f"   ✓ acurácia = {acc:.4f} (predições perfeitas)")

    # Predições erradas
    preds_wrong = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    acc_wrong = ternary_accuracy(preds_wrong, targets)
    assert acc_wrong == 0.0, "Predições invertidas devem dar acurácia 0.0"
    print(f"   ✓ acurácia = {acc_wrong:.4f} (predições erradas)")

    # --- Teste 5: MNIST real ---
    print("\n[5] Forward com MNIST real")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils import load_mnist
    _, _, test_data, test_labels = load_mnist(normalize=False)

    # Forward em mini-batch de 100 amostras
    x_mnist = test_data[:100].astype(np.float32)
    out_mnist = net.forward(x_mnist)
    assert out_mnist.shape == (100, 10), f"Shape incorreto: {out_mnist.shape}"
    acc_mnist = ternary_accuracy(out_mnist, test_labels[:100])
    print(f"   ✓ MNIST: 100 amostras, acurácia inicial = {acc_mnist:.4f}")
    print("   (esperado ~0.0 — rede não treinada ainda)")

    # --- Teste 6: Reprodutibilidade ---
    print("\n[6] Reprodutibilidade")

    net1 = TernaryNet(seed=123)
    net2 = TernaryNet(seed=123)
    x_test = np.random.randn(1, 784).astype(np.float32)
    out1 = net1.forward(x_test)
    out2 = net2.forward(x_test)
    assert np.array_equal(out1, out2), "Redes com mesma seed devem ser idênticas"
    print("   ✓ reprodutível com seed")

    print("\n" + "=" * 60)
    print("Todos os testes passaram!")
    print("=" * 60)
