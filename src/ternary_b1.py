"""
[007] Treino B1 — regra por neurônio + reset error-feedback

Integra `ternary_net.py`, `counter_rules.py`, `reset_policies.py`,
`fake_gradient.py` e `utils.py` em um loop de treino completo para MNIST.

Fluxo por amostra:
    1. Forward pass (step activation)
    2. Cálculo de erro (L1 por neurônio de saída)
    3. Fake gradient triangular quantizado
    4. Propagação de δ camada a camada
    5. Acumulação nos contadores
    6. Seleção B1 (por neurônio disparado)
    7. Atualização de pesos
    8. Reset error-feedback

Uso:
    from src.ternary_b1 import train_b1
    net, metrics = train_b1(epochs=20, lr_fake=1.0)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent dir to path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ternary_net import (
    TernaryNet,
    ternary_binary_loss,
    ternary_accuracy,
)
from src.counter_rules import (
    CounterManager,
    apply_update_neuron,
    compute_counter_delta_single,
    update_weight,
)
from src.reset_policies import (
    reset_error_feedback,
    apply_reset_on_manager,
)
from src.fake_gradient import (
    triangular_fake_grad,
    quantize_fake_grad,
    propagate_delta_with_fake,
    get_adaptive_radius,
    get_layer_radii,
)
from src.utils import (
    load_mnist,
    save_metrics,
    ensure_dir,
    set_seed,
    format_number,
)


# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMS = {
    "scale": 1.0,         # fake gradient amplitude
    "radius": 1.0,        # fake gradient base width (None = adaptativo)
    "scale_factor": 1.0,  # c para radius adaptativo: radius_l = c * sqrt(fan_in)
    "adaptive_radius": True,  # se True usa radius por camada = c*sqrt(fan_in)
    "levels": 5,          # quantization levels {-2,-1,0,+1,+2}
    "batch_size": 32,     # samples per batch
    "epochs": 20,         # training epochs
    "target_scale": 1.0,  # normalize target scale
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Treino por amostra (online)
# ---------------------------------------------------------------------------
def train_sample_online(
    net: TernaryNet,
    cm: CounterManager,
    x: np.ndarray,
    target: int,
    scale: float = 1.0,
    radius: float = 1.0,
    levels: int = 5,
    adaptive_radius: bool = False,
    scale_factor: float = 1.0,
) -> Dict:
    """
    Treina uma única amostra com regra B1 + error-feedback.

    Args:
        net: TernaryNet.
        cm: CounterManager com contadores.
        x: input, shape (784,).
        target: label verdadeiro (0-9).
        scale: amplitude do fake gradient.
        radius: largura da base do fake gradient.
        levels: níveis de quantização.

    Returns:
        Dict com métricas da amostra.
    """
    input_dim = net.input_dim
    metrics = {
        "fired_neurons": 0,
        "updated_weights": 0,
        "total_updates": 0,
    }

    # 1. Forward pass — garantir shape (1, dim) para cada camada
    pre_activations = []  # z pré-degrau por camada
    activations = [x.reshape(1, -1)]  # ativações por camada (input → hidden1 → hidden2 → output)
    out = x.reshape(1, -1)
    for layer in net.layers:
        z = out @ layer.weights.T + layer.bias
        pre_activations.append(z)
        out = (z > 0).astype(np.float32)
        activations.append(out)

    # 2. One-hot target
    batch_size = 1
    one_hot = np.zeros((batch_size, net.output_dim), dtype=np.float32)
    one_hot[0, target] = 1.0

    # Helpers de raio
    def _radius_for(layer_input_dim: int) -> float:
        if adaptive_radius:
            return get_adaptive_radius(layer_input_dim, scale_factor)
        return radius

    # 3. Erro e fake gradient na saída
    error = out - one_hot  # (1, 10)
    loss = np.mean(np.abs(error))

    # Fake gradient triangular na saída
    z_output = pre_activations[-1]  # (1, 10)
    r_out = _radius_for(net.layers[2].input_dim)
    fake_grad_output = triangular_fake_grad(z_output, scale=scale, radius=r_out)
    # Quantiza
    q_fake_grad = quantize_fake_grad(fake_grad_output, levels=levels)

    # 4. Propagação de δ
    # δ na saída: gradiente da perda × fake gradient
    delta = np.sign(error) * q_fake_grad  # (1, 10)

    # Acumular contadores na camada de saída (layer 2)
    delta_w = delta.T @ activations[-2]  # (10, 64)
    cm.accumulate(2, delta_w.astype(np.int64))

    # Propaga δ para camada 1 (hidden2) — usa pesos da camada 2 (saída)
    # net.layers[2].weights é (10, 64), propagate_delta espera (current, next)
    r1 = _radius_for(net.layers[1].input_dim)
    delta = propagate_delta_with_fake(
        delta, net.layers[2].weights.T, pre_activations[-2],
        scale=scale, radius=r1
    )
    delta_w1 = delta.T @ activations[-3]  # (64, 128)
    cm.accumulate(1, delta_w1.astype(np.int64))

    # Propaga δ para camada 0 (hidden1) — usa pesos da camada 1 (hidden2)
    r0 = _radius_for(net.layers[0].input_dim)
    delta = propagate_delta_with_fake(
        delta, net.layers[1].weights.T, pre_activations[-3],
        scale=scale, radius=r0
    )
    delta_w0 = delta.T @ activations[-4]  # (128, 784)
    cm.accumulate(0, delta_w0.astype(np.int64))

    # 5. Atualização B1: por neurônio disparado (vetorizado)
    # Para a camada de saída, cada neurônio disparado atualiza suas conexões
    fired_mask = (out[0] > 0).astype(int)  # (10,)
    metrics["fired_neurons"] = int(fired_mask.sum())

    def apply_b1_layer(layer, cm, layer_idx, delta_signs):
        """Aplica B1 para todos os neurônios de uma camada de uma vez."""
        updated_total = 0
        total_total = 0
        threshold = max(1, np.mean(np.abs(cm.weight_counters[layer_idx])))

        for neuron_idx in range(layer.output_dim):
            C = cm.weight_counters[layer_idx][neuron_idx]
            mask = np.abs(C) > threshold
            n_updated = int(np.sum(mask))
            updated_total += n_updated
            total_total += layer.input_dim

            if n_updated > 0:
                signs = np.sign(C[mask])
                for i, col_idx in enumerate(np.where(mask)[0]):
                    sign = delta_signs[neuron_idx] * int(signs[i])
                    if sign != 0:
                        layer.weights[neuron_idx, col_idx] = update_weight(
                            layer.weights[neuron_idx, col_idx], sign
                        )

        return {"updated": updated_total, "total": total_total}

    # Camada de saída
    delta_signs_out = np.sign(-error[0])
    delta_signs_out[delta_signs_out == 0] = 1
    result = apply_b1_layer(net.layers[2], cm, 2, delta_signs_out)
    metrics["updated_weights"] += result["updated"]
    metrics["total_updates"] += result["total"]

    # Camadas ocultas
    for layer_idx in [0, 1]:
        layer = net.layers[layer_idx]
        layer_activations = activations[layer_idx + 1]  # saída desta camada
        layer_fired = (layer_activations[0] > 0).astype(int)

        delta_signs_hidden = np.sign(delta[0])
        delta_signs_hidden[delta_signs_hidden == 0] = 1

        result = apply_b1_layer(layer, cm, layer_idx, delta_signs_hidden)
        metrics["updated_weights"] += result["updated"]
        metrics["total_updates"] += result["total"]

    # 6. Reset error-feedback
    for layer_idx in range(len(net.layers)):
        # Encontrar índices atualizados (todos os que foram processados)
        layer = net.layers[layer_idx]
        layer_fired = (activations[layer_idx + 1][0] > 0).astype(int)
        fired_indices = np.where(layer_fired > 0)[0]

        if len(fired_indices) > 0:
            # Para B1, os contadores resetados são os das conexões atualizadas
            # Simplificação: resetar todos os contadores dos neurônios disparados
            all_col_indices = []
            all_row_indices = []
            for neuron_idx in fired_indices:
                C = cm.weight_counters[layer_idx][neuron_idx]
                threshold = max(1, np.mean(np.abs(C)))
                mask = np.abs(C) > threshold
                if mask.any():
                    cols = np.where(mask)[0]
                    all_row_indices.extend([neuron_idx] * len(cols))
                    all_col_indices.extend(cols.tolist())

            if all_row_indices:
                reset_error_feedback(
                    cm.weight_counters[layer_idx],
                    np.array(all_row_indices, dtype=np.int64),
                    np.array(all_col_indices, dtype=np.int64),
                )

    metrics["loss"] = float(loss)
    return metrics


# ---------------------------------------------------------------------------
# Treino por mini-batch (batched)
# ---------------------------------------------------------------------------
def train_batch(
    net: TernaryNet,
    cm: CounterManager,
    batch_x: np.ndarray,
    batch_targets: np.ndarray,
    scale: float = 1.0,
    radius: float = 1.0,
    levels: int = 5,
    adaptive_radius: bool = False,
    scale_factor: float = 1.0,
) -> Dict:
    """
    Treina um mini-batch com regra B1 + error-feedback.

    Args:
        net: TernaryNet.
        cm: CounterManager.
        batch_x: inputs, shape (batch_size, 784).
        batch_targets: labels, shape (batch_size,).
        scale: amplitude do fake gradient.
        radius: largura da base do fake gradient.
        levels: níveis de quantização.

    Returns:
        Dict com métricas do batch.
    """
    batch_size = batch_x.shape[0]
    total_loss = 0.0
    total_fired = 0
    total_updated = 0
    total_updates = 0

    for i in range(batch_size):
        sample_metrics = train_sample_online(
            net, cm, batch_x[i], int(batch_targets[i]),
            scale=scale, radius=radius, levels=levels,
            adaptive_radius=adaptive_radius, scale_factor=scale_factor,
        )
        total_loss += sample_metrics["loss"]
        total_fired += sample_metrics["fired_neurons"]
        total_updated += sample_metrics["updated_weights"]
        total_updates += sample_metrics["total_updates"]

    return {
        "loss": total_loss / batch_size,
        "fired_neurons": total_fired,
        "updated_weights": total_updated,
        "total_updates": total_updates,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def evaluate(
    net: TernaryNet,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 128,
) -> float:
    """
    Avalia acurácia da rede no conjunto de teste.

    Args:
        net: TernaryNet.
        test_data: dados de teste, shape (N, 784).
        test_labels: labels de teste, shape (N,).
        batch_size: tamanho do batch para avaliação.

    Returns:
        Acurácia (0.0 a 1.0).
    """
    n_samples = test_data.shape[0]
    correct = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_x = test_data[start:end].astype(np.float32)
        batch_y = test_labels[start:end]

        # Forward pass
        out = net.forward(batch_x)

        # Classificação: argmax do pre-activation (z antes do degrau)
        # Para isso, recompute z
        z = batch_x @ net.layers[0].weights.T + net.layers[0].bias
        z = (z > 0).astype(np.float32) @ net.layers[1].weights.T + net.layers[1].bias
        z = z @ net.layers[2].weights.T + net.layers[2].bias

        preds = np.argmax(z, axis=1)
        correct += np.sum(preds == batch_y)

    return correct / n_samples


# ---------------------------------------------------------------------------
# Loop principal de treino
# ---------------------------------------------------------------------------
def train_b1(
    epochs: int = 20,
    batch_size: int = 32,
    scale: float = 1.0,
    radius: float = 1.0,
    levels: int = 5,
    seed: int = 42,
    exp_name: str = "exp-007-b1-error-feedback",
    adaptive_radius: bool = False,
    scale_factor: float = 1.0,
) -> Tuple[TernaryNet, Dict]:
    """
    Treina a Rede B com regra B1 + error-feedback.

    Args:
        epochs: número de epochs.
        batch_size: tamanho do batch.
        scale: amplitude do fake gradient.
        radius: largura da base do fake gradient.
        levels: níveis de quantização.
        seed: semente para reprodutibilidade.
        exp_name: nome do experimento para salvar métricas.

    Returns:
        (net, metrics) — rede treinada e dicionário de métricas.
    """
    set_seed(seed)

    # Carregar dados
    print("Carregando MNIST...")
    train_data, train_labels, test_data, test_labels = load_mnist(normalize=True)
    print(f"  Train: {train_data.shape}, Test: {test_data.shape}")

    # Inicializar rede e contadores
    net = TernaryNet(seed=seed)
    cm = CounterManager([
        (128, 784), (64, 128), (10, 64)
    ])

    if adaptive_radius:
        layer_radii = get_layer_radii([784, 128, 64], scale_factor=scale_factor)
        print(f"\n{'='*60}")
        print(f"Treino B1 + error-feedback (ADAPTIVE radius c={scale_factor} → {layer_radii})")
        print(f"  Architecture: {net.get_stats()['architecture']}")
        print(f"  Hyperparams: scale={scale}, adaptive_radius={adaptive_radius}, scale_factor={scale_factor}, levels={levels}")
        print(f"  Batch size: {batch_size}, Epochs: {epochs}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"Treino B1 + error-feedback")
        print(f"  Architecture: {net.get_stats()['architecture']}")
        print(f"  Hyperparams: scale={scale}, radius={radius}, levels={levels}")
        print(f"  Batch size: {batch_size}, Epochs: {epochs}")
        print(f"{'='*60}\n")

    # Métricas
    metrics = {
        "experiment": exp_name,
        "hyperparams": {
            "scale": scale,
            "radius": radius,
            "adaptive_radius": adaptive_radius,
            "scale_factor": scale_factor,
            "levels": levels,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
        },
        "architecture": net.get_stats()["architecture"],
        "total_weights": net.get_stats()["total_weights"],
        "epoch_metrics": [],
    }

    # Loop de treino
    for epoch in range(1, epochs + 1):
        # Embaralhar dados
        perm = np.random.permutation(train_data.shape[0])
        shuffled_x = train_data[perm]
        shuffled_y = train_labels[perm]

        epoch_loss = 0.0
        epoch_fired = 0
        epoch_updated = 0
        n_batches = 0

        # Mini-batches
        for start in range(0, train_data.shape[0], batch_size):
            end = min(start + batch_size, train_data.shape[0])
            batch_x = shuffled_x[start:end]
            batch_y = shuffled_y[start:end]

            batch_metrics = train_batch(
                net, cm, batch_x, batch_y,
                scale=scale, radius=radius, levels=levels,
                adaptive_radius=adaptive_radius, scale_factor=scale_factor,
            )
            epoch_loss += batch_metrics["loss"]
            epoch_fired += batch_metrics["fired_neurons"]
            epoch_updated += batch_metrics["updated_weights"]
            n_batches += 1

        # Avaliação
        test_acc = evaluate(net, test_data, test_labels, batch_size=128)

        # Métricas do epoch
        avg_loss = epoch_loss / n_batches
        avg_fired = epoch_fired / n_batches
        avg_updated = epoch_updated / n_batches

        epoch_record = {
            "epoch": epoch,
            "loss": avg_loss,
            "fired_neurons": avg_fired,
            "updated_weights": avg_updated,
            "test_accuracy": test_acc,
        }
        metrics["epoch_metrics"].append(epoch_record)

        # Log
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Fired: {avg_fired:.1f} | "
              f"Updated: {avg_updated:.1f} | "
              f"Test Acc: {test_acc:.4f}")

    # Salvar resultados
    results_dir = Path("results") / exp_name
    ensure_dir(results_dir)

    # Salvar métricas
    save_metrics(metrics, str(results_dir / "metrics.json"))

    # Salvar pesos
    np.savez(
        str(results_dir / "weights.npz"),
        **{f"layer_{i}": net.layers[i].weights
           for i in range(len(net.layers))},
    )

    # Estatísticas finais
    stats = net.get_stats()
    print(f"\n{'='*60}")
    print(f"Treino concluído!")
    print(f"  Final test accuracy: {test_acc:.4f}")
    print(f"  Total weights: {format_number(stats['total_weights'])}")
    print(f"  +{stats['total_positive']} 0:{stats['total_zero']} -{stats['total_negative']}")
    print(f"  Results: {results_dir}")
    print(f"{'='*60}")

    return net, metrics


# ---------------------------------------------------------------------------
# Testes unitários
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testes — ternary_b1.py")
    print("=" * 60)

    # --- Teste 1: train_sample_online ---
    print("\n[1] train_sample_online")

    set_seed(42)
    net_test = TernaryNet(seed=42)
    cm_test = CounterManager([(128, 784), (64, 128), (10, 64)])

    x_sample = np.random.randn(784).astype(np.float32)
    target_sample = 5

    sample_metrics = train_sample_online(
        net_test, cm_test, x_sample, target_sample,
        scale=1.0, radius=1.0, levels=5,
    )

    assert "loss" in sample_metrics, "Metrics missing 'loss'"
    assert "fired_neurons" in sample_metrics, "Metrics missing 'fired_neurons'"
    assert "updated_weights" in sample_metrics, "Metrics missing 'updated_weights'"
    assert sample_metrics["loss"] >= 0.0, "Loss deve ser não-negativo"
    print(f"   ✓ sample online: loss={sample_metrics['loss']:.4f}, "
          f"fired={sample_metrics['fired_neurons']}, "
          f"updated={sample_metrics['updated_weights']}")

    # --- Teste 2: train_batch ---
    print("\n[2] train_batch")

    net_test2 = TernaryNet(seed=42)
    cm_test2 = CounterManager([(128, 784), (64, 128), (10, 64)])

    batch_x = np.random.randn(8, 784).astype(np.float32)
    batch_y = np.random.randint(0, 10, size=8)

    batch_metrics = train_batch(
        net_test2, cm_test2, batch_x, batch_y,
        scale=1.0, radius=1.0, levels=5,
    )

    assert "loss" in batch_metrics
    assert batch_metrics["batch_size"] == 8
    assert batch_metrics["loss"] >= 0.0
    print(f"   ✓ batch: loss={batch_metrics['loss']:.4f}, "
          f"size={batch_metrics['batch_size']}")

    # --- Teste 3: evaluate ---
    print("\n[3] evaluate")

    net_test3 = TernaryNet(seed=42)
    _, _, test_x, test_y = load_mnist(normalize=True)

    acc = evaluate(net_test3, test_x, test_y, batch_size=128)
    print(f"   ✓ evaluate: accuracy={acc:.4f} (rede não treinada)")
    assert 0.0 <= acc <= 1.0, "Acurácia fora de [0, 1]"

    # --- Teste 4: Mini treino (2 epochs, 50 samples) ---
    print("\n[4] Mini treino (2 epochs, 50 samples)")

    set_seed(42)
    train_data_mini, train_labels_mini, test_mini_x, test_mini_y = load_mnist(normalize=True)
    # Usar apenas 50 amostras para teste rápido
    train_data_mini = train_data_mini[:50]
    train_labels_mini = train_labels_mini[:50]

    # Treino manual com subconjunto
    net_mini = TernaryNet(seed=42)
    cm_mini = CounterManager([(128, 784), (64, 128), (10, 64)])
    mini_metrics = {"epoch_metrics": []}

    for epoch in range(1, 3):
        perm = np.random.permutation(50)
        shuffled_x = train_data_mini[perm]
        shuffled_y = train_labels_mini[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, 50, 8):
            end = min(start + 8, 50)
            bm = train_batch(net_mini, cm_mini, shuffled_x[start:end], shuffled_y[start:end])
            epoch_loss += bm["loss"]
            n_batches += 1
        
        acc = evaluate(net_mini, test_mini_x, test_mini_y, batch_size=128)
        mini_metrics["epoch_metrics"].append({"epoch": epoch, "loss": epoch_loss/n_batches, "test_accuracy": acc})
        print(f"  Epoch {epoch}/2 | Loss: {epoch_loss/n_batches:.4f} | Acc: {acc:.4f}")

    assert len(mini_metrics["epoch_metrics"]) == 2
    print(f"   ✓ 2 epochs com 50 samples, final acc={mini_metrics['epoch_metrics'][-1]['test_accuracy']:.4f}")

    assert len(mini_metrics["epoch_metrics"]) == 2
    assert "test_accuracy" in mini_metrics["epoch_metrics"][0]
    assert "loss" in mini_metrics["epoch_metrics"][0]

    final_acc = mini_metrics["epoch_metrics"][-1]["test_accuracy"]
    print(f"   ✓ 2 epochs completados, final acc={final_acc:.4f}")

    # --- Teste 5: Reprodutibilidade (1 epoch, 16 samples) ---
    print("\n[5] Reprodutibilidade")

    set_seed(99)
    train_data_r, train_labels_r, _, _ = load_mnist(normalize=True)
    train_data_r = train_data_r[:16]
    train_labels_r = train_labels_r[:16]

    def repro_train(seed_val):
        set_seed(seed_val)
        n = len(train_data_r)
        net_r = TernaryNet(seed=seed_val)
        cm_r = CounterManager([(128, 784), (64, 128), (10, 64)])
        for _ in range(1):
            perm = np.random.permutation(n)
            for start in range(0, n, 8):
                end = min(start + 8, n)
                train_batch(net_r, cm_r, train_data_r[perm[start:end]], train_labels_r[perm[start:end]])
        return net_r

    net_r1 = repro_train(99)
    net_r2 = repro_train(99)

    for i in range(3):
        assert np.array_equal(net_r1.layers[i].weights, net_r2.layers[i].weights), \
            f"Camada {i} não reprodutível"
    print("   ✓ reprodutível com seed")

    print("\n" + "=" * 60)
    print("Todos os testes passaram!")
    print("=" * 60)
