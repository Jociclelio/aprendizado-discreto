"""
[007] Treino B1 — regra por neurônio + reset error-feedback
[012] Reparo do contrato matemático (ver CONTRATO_012 abaixo).

Integra `ternary_net.py`, `counter_rules.py`, `reset_policies.py`,
`fake_gradient.py` e `utils.py` em um loop de treino completo para MNIST.

CONTRATO_012 (convenção única para todas as camadas):
    z_l = W_l @ x_{l-1} + b_l        (pré-ativação; W int8, b int32)
    a_l = step(z_l), step(z)=1 se z>0 else 0   (ocultas em {0,1})
    scores = z_out                   (pré-ativação da saída, ANTES do degrau)
    pred = argmax(scores)            (evaluate consome scores)
    C acumula GRADIENTE: delta_out = sign(a_out − target) * q_fake(z_out),
        deltas ocultos por regra da cadeia com surrogate quantizado.
    Atualização (descida):  W <- clip(W − sign(C), −1, +1),
                            b <- clip(b − sign(C_b), −N, N−1).
        Sem multiplicação extra por sign(−error) da amostra corrente;
        cada camada usa seu próprio deltas[l] (C).
    Seleção B1: máscara única por camada,
        mask = |C| > max(1, mean(|C_cam|)), aplicada à atualização E ao
        reset (error-feedback). Sem gate de disparo: neurônio inativo
        pode se recuperar via evidência acumulada; `fired` é diagnóstico.
    Saturação: tentativa em peso saturado (±1 pedindo mesmo sentido) é
        BLOQUEADA (peso não muda) mas o contador é CONSUMIDO (resetado);
        métricas separam selecionados/tentativas/bloqueios/transições.
    Leak [012]: simétrico, zero ponto fixo —
        gamma: sign(C)*floor(gamma*|C|);
        shift: sign(C)*(|C|−step), step=max(1, |C|>>s) p/ |C|>0.
    Quantização (H): surrogate quantizado também nas ocultas
        (propagate_delta_with_fake_quantized) e acúmulo por
        round(delta_float) — nunca truncamento direto.

Fluxo por amostra:
    1. Forward único via forward_with_scores (scores + binárias)
    2. Erro L1 por neurônio de saída
    3. Fake gradient triangular quantizado (saída + ocultas)
    4. Propagação de deltas[l] por camada (cada camada usa o seu)
    5. Acumulação nos contadores (pesos E bias, com round)
    6. Máscara única de seleção por camada
    7. Atualização de pesos E bias (W−sign(C))
    8. Reset error-feedback na MESMA máscara

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
    propagate_delta_with_fake_quantized,
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
    "decay_rate": None,   # gamma em (0,1) para leaky, ou None
    "decay_shift": None,  # shift 3/4/5 para bitshift, ou None (011)
    "decay_per_batch": True,  # se True aplica decay a cada batch
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
    [012] Treina uma única amostra com regra B1 + error-feedback.

    Contrato: ver docstring do módulo (CONTRATO_012).

    Args:
        net: TernaryNet.
        cm: CounterManager com contadores.
        x: input, shape (784,).
        target: label verdadeiro (0-9).
        scale: amplitude do fake gradient.
        radius: largura da base do fake gradient.
        levels: níveis de quantização.

    Returns:
        Dict com métricas da amostra (inclui selecionados/tentativas/
        bloqueios/transições por pesos e bias, e sinal pré/pós-quant).
    """
    # 1. Forward ÚNICO (A): scores + binárias, degrau em TODAS as ocultas.
    scores, a_out, pre_activations, activations = net.forward_with_scores(
        x.reshape(1, -1).astype(np.float32)
    )
    # scores/a_out: (1, 10); activations: [x, a1, a2, a_out]; pre: [z1,z2,z3]
    out = activations[-1]  # (1, 10) binária

    # 2. One-hot target + erro (gradiente do L1 médio a menos de constante)
    one_hot = np.zeros((1, net.output_dim), dtype=np.float32)
    one_hot[0, target] = 1.0
    error = out - one_hot  # (1, 10)
    loss = float(np.mean(np.abs(error)))

    def _radius_for(layer_input_dim: int) -> float:
        if adaptive_radius:
            return get_adaptive_radius(layer_input_dim, scale_factor)
        return radius

    # 3. Fake gradient na saída (quantizado) → delta próprio da camada (C)
    z_output = pre_activations[-1]  # (1, 10)
    r_out = _radius_for(net.layers[2].input_dim)
    fake_grad_output = triangular_fake_grad(z_output, scale=scale, radius=r_out)
    q_fake_grad = quantize_fake_grad(fake_grad_output, levels=levels)
    deltas = [None, None, None]
    deltas[2] = (np.sign(error) * q_fake_grad).astype(np.float64)  # (1, 10)

    # 4. Propagação com surrogate QUANTIZADO (H); cada camada guarda o seu (C)
    r1 = _radius_for(net.layers[1].input_dim)
    deltas[1] = propagate_delta_with_fake_quantized(
        deltas[2], net.layers[2].weights.T, pre_activations[-2],
        scale=scale, radius=r1, levels=levels,
    )
    r0 = _radius_for(net.layers[0].input_dim)
    deltas[0] = propagate_delta_with_fake_quantized(
        deltas[1], net.layers[1].weights.T, pre_activations[-3],
        scale=scale, radius=r0, levels=levels,
    )

    # 5. Acumulação (pesos E bias) com round coerente (H, E)
    quant_stats = {}
    for l in range(3):
        act_in = activations[l]  # (1, fan_in) — 0/1 nas ocultas, pixels [0,1] em l=0
        dw_float = deltas[l].T @ act_in  # (out_dim, fan_in) float
        db_float = deltas[l][0]  # (out_dim,) float
        # Medir sinal antes/depois da conversão inteira (H)
        nz_before = int(np.sum(dw_float != 0))
        dw_int = np.round(dw_float).astype(np.int64)
        db_int = np.round(db_float).astype(np.int64)
        quant_stats[l] = {
            "dw_nz_float": nz_before,
            "dw_nz_int": int(np.sum(dw_int != 0)),
            "dw_mean_abs_float": float(np.mean(np.abs(dw_float))),
            "dw_mean_abs_int": float(np.mean(np.abs(dw_int))),
        }
        cm.accumulate(l, dw_int, db_int)

    # 6+7+8. Máscara ÚNICA por camada → atualização → reset na mesma máscara (D)
    total_selected = 0
    total_attempts = 0
    total_blocked = 0
    total_transitions = 0
    bias_selected = 0
    bias_blocked = 0
    bias_transitions = 0

    for l, layer in enumerate(net.layers):
        Cw = cm.weight_counters[l]
        thr_w = max(1.0, float(np.mean(np.abs(Cw))))
        mask_w = np.abs(Cw) > thr_w  # máscara única (D)
        rows, cols = np.where(mask_w)
        selected = len(rows)
        blocked = 0
        transitions = 0
        for r, c in zip(rows.tolist(), cols.tolist()):
            step = -int(np.sign(Cw[r, c]))  # (B) descida, sem sign(−error) extra
            if step != 0:
                w_old = int(layer.weights[r, c])
                w_new = int(np.clip(w_old + step, -1, 1))
                if w_new == w_old:
                    blocked += 1
                else:
                    layer.weights[r, c] = np.int8(w_new)
                    transitions += 1
        # Reset error-feedback na MESMA máscara (D), inclusive bloqueados
        # (saturação consome o contador — ver contrato).
        if selected:
            reset_error_feedback(
                Cw,
                np.array(rows, dtype=np.int64),
                np.array(cols, dtype=np.int64),
            )

        # Bias (E): mesma disciplina — máscara própria, descida, faixa [-N, N-1]
        Cb = cm.bias_counters[l]
        thr_b = max(1.0, float(np.mean(np.abs(Cb)))) if Cb.size else 1.0
        mask_b = np.abs(Cb) > thr_b
        b_idx = np.where(mask_b)[0]
        for bi in b_idx.tolist():
            bstep = -int(np.sign(Cb[bi]))
            if bstep != 0:
                b_old = int(layer.bias[bi])
                b_new = int(np.clip(b_old + bstep, -layer.input_dim, layer.input_dim - 1))
                if b_new == b_old:
                    bias_blocked += 1
                else:
                    layer.bias[bi] = np.int32(b_new)
                    bias_transitions += 1
        n_bsel = len(b_idx)
        bias_selected += n_bsel
        if n_bsel:
            Cb[b_idx] = 0  # consome contador de bias selecionado

        total_selected += selected
        total_attempts += selected
        total_blocked += blocked
        total_transitions += transitions

    metrics = {
        "loss": loss,
        # Diagnóstico de disparo (firing é diagnóstico, não gate)
        "fired_neurons": int(np.sum(out[0] > 0)),
        "firing_rates": [float(np.mean(a)) for a in activations[1:]],
        # Pesos: separação P2
        "selected_weights": total_selected,
        "attempts": total_attempts,
        "blocked_weights": total_blocked,
        "transitions": total_transitions,
        "updated_weights": total_transitions,  # legado = transições efetivas
        "total_updates": sum(l.input_dim * l.output_dim for l in net.layers),
        # Bias (E)
        "bias_selected": bias_selected,
        "bias_blocked": bias_blocked,
        "bias_transitions": bias_transitions,
        # Quantização (H)
        "quant_stats": quant_stats,
    }
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
    decay_per_batch: bool = False,
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
    # Decaimento temporal leaky (011): aplica uma vez por batch antes das amostras
    if decay_per_batch and (cm.decay_rate is not None or cm.decay_shift is not None):
        cm.decay()

    batch_size = batch_x.shape[0]
    total_loss = 0.0
    total_fired = 0
    total_selected = 0
    total_blocked = 0
    total_transitions = 0
    total_updates = 0

    for i in range(batch_size):
        sample_metrics = train_sample_online(
            net, cm, batch_x[i], int(batch_targets[i]),
            scale=scale, radius=radius, levels=levels,
            adaptive_radius=adaptive_radius, scale_factor=scale_factor,
        )
        total_loss += sample_metrics["loss"]
        total_fired += sample_metrics["fired_neurons"]
        total_selected += sample_metrics["selected_weights"]
        total_blocked += sample_metrics["blocked_weights"]
        total_transitions += sample_metrics["transitions"]
        total_updates += sample_metrics["total_updates"]

    return {
        "loss": total_loss / batch_size,
        "fired_neurons": total_fired,
        "selected_weights": total_selected,
        "blocked_weights": total_blocked,
        "transitions": total_transitions,
        "updated_weights": total_transitions,  # legado = transições
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
    [012] Avalia acurácia: argmax(scores) do forward unificado.

    Consome `forward_with_scores` em batches — sem reimplementar camadas,
    com degrau em todas as ocultas (corrige achado A).

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

        scores, _, _, _ = net.forward_with_scores(batch_x)
        preds = np.argmax(scores, axis=1)
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
    adaptive_radius: bool = True,
    scale_factor: float = 1.0,
    decay_rate: Optional[float] = None,
    decay_shift: Optional[int] = None,
    decay_per_batch: bool = True,
) -> Tuple[TernaryNet, Dict]:
    """
    Treina a Rede B com regra B1 + error-feedback.

    [012] Default `adaptive_radius=True` alinhado a DEFAULT_HYPERPARAMS.

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

    # Inicializar rede e contadores (com leaky se configurado)
    net = TernaryNet(seed=seed)
    cm = CounterManager([
        (128, 784), (64, 128), (10, 64)
    ], decay_rate=decay_rate, decay_shift=decay_shift)

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
            "decay_rate": decay_rate,
            "decay_shift": decay_shift,
            "decay_per_batch": decay_per_batch,
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
                decay_per_batch=decay_per_batch,
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

    # Salvar pesos + bias [012] (inferência com bias treinado) + contadores
    np.savez(
        str(results_dir / "weights.npz"),
        **{f"layer_{i}": net.layers[i].weights
           for i in range(len(net.layers))},
        **{f"bias_{i}": net.layers[i].bias
           for i in range(len(net.layers))},
        **{f"counter_{i}": cm.weight_counters[i]
           for i in range(len(net.layers))},
        **{f"counter_bias_{i}": cm.bias_counters[i]
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
