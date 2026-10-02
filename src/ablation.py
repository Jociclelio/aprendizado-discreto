"""
[015] Ablacoes — margem, homeostase e leak sobre o B1 corrigido.

Cada mecanismo e uma hipotese distinta, testada isoladamente e depois
combinada, com orcamento e conjunto de selecao fixos (validacao de 014).

Mecanismos:
    - Margem: perda multiclasse L = max(0, margin + max_{j!=y} z_j - z_y)
      sobre z_out. Subgradiente ja e em relacao ao score; NAO multiplicar
      novamente pelo surrogate do degrau de saida. Surrogate continua
      nas ocultas discretas.
    - Homeostase: ajuste do bias para manter taxa de disparo perto de
      um alvo (ex.: 0.25). Frequencia, tolerancia e interacao com o
      bias supervisionado explicitos.
    - Leak: gamma/shift de 011, reinterpretados com metricas de
      oscilacao + desempenho (ja suportado via CounterManager).

Uso:
    from src.ablation import train_sample_ablation, apply_homeostasis
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.counter_rules import CounterManager
from src.fake_gradient import (
    get_adaptive_radius,
    propagate_delta_with_fake_quantized,
    quantize_fake_grad,
    triangular_fake_grad,
)
from src.reset_policies import reset_error_feedback
from src.ternary_net import TernaryNet


# ---------------------------------------------------------------------------
# Margem multiclasse
# ---------------------------------------------------------------------------
def margin_loss(z_out: np.ndarray, target: int, margin: float) -> tuple:
    """
    Perda multiclasse com margem sobre z_out.

    L = max(0, margin + max_{j!=y} z_j - z_y)

    Args:
        z_out: pre-ativacoes de saida, shape (1, 10).
        target: classe verdadeira.
        margin: margem desejada.

    Returns:
        (loss, delta_out) — delta_out e o subgradiente em relacao a z_out.
    """
    z_y = z_out[0, target]
    z_other = z_out[0].copy()
    z_other[target] = -np.inf
    j_max = int(np.argmax(z_other))
    z_max_other = z_other[j_max]

    loss = max(0.0, margin + z_max_other - z_y)

    delta_out = np.zeros((1, z_out.shape[1]), dtype=np.float64)
    if loss > 0:
        delta_out[0, target] = -1.0   # aumentar z_y
        delta_out[0, j_max] = +1.0    # diminuir z_max_other

    return loss, delta_out


# ---------------------------------------------------------------------------
# Homeostase
# ---------------------------------------------------------------------------
def apply_homeostasis(
    net: TernaryNet,
    layer_idx: int,
    firing_rates: list,
    target_rate: float = 0.25,
    tolerance: float = 0.05,
    step_size: int = 1,
) -> Dict:
    """
    Ajusta o bias de uma camada para manter a taxa de disparo perto do alvo.

    Se firing_rate > target + tolerance: bias -= step (reduz disparos)
    Se firing_rate < target - tolerance: bias += step (aumenta disparos)

    A homeostase NAO substitui o bias supervisionado; ela atua como um
    mecanismo de regulacao complementar.

    Args:
        net: TernaryNet.
        layer_idx: camada a ajustar.
        firing_rates: taxa de disparo por camada (de forward_with_scores).
        target_rate: taxa de disparo alvo.
        tolerancia: banda morta ao redor do alvo.
        step_size: incremento do bias por ajuste.

    Returns:
        Dict com informacoes do ajuste.
    """
    layer = net.layers[layer_idx]
    fr = firing_rates[layer_idx]
    adjustment = 0

    if fr > target_rate + tolerance:
        adjustment = -step_size
    elif fr < target_rate - tolerance:
        adjustment = +step_size

    if adjustment != 0:
        layer.bias = np.clip(
            layer.bias + adjustment,
            -layer.input_dim,
            layer.input_dim - 1,
        ).astype(np.int32)

    return {
        "layer": layer_idx,
        "firing_rate": fr,
        "target": target_rate,
        "adjustment": adjustment,
    }


# ---------------------------------------------------------------------------
# Treino com ablacoes
# ---------------------------------------------------------------------------
def train_sample_ablation(
    net: TernaryNet,
    cm: CounterManager,
    x: np.ndarray,
    target: int,
    scale: float = 1.0,
    radius: float = 1.0,
    levels: int = 5,
    adaptive_radius: bool = False,
    scale_factor: float = 1.0,
    margin: Optional[float] = None,
    homeostasis_target: Optional[float] = None,
    homeostasis_tolerance: float = 0.05,
    homeostasis_step: int = 1,
) -> Dict:
    """
    Treina uma amostra com B1 + ablacoes opcionais (margem, homeostase).

    Contrato [012] mantido: C acumula GRADIENTE => W <- clip(W - sign(C)).

    Args:
        net: TernaryNet.
        cm: CounterManager.
        x: input, shape (784,).
        target: label verdadeiro (0-9).
        scale, radius, levels: parametros do fake gradient.
        adaptive_radius: se True, usa radius por camada = c*sqrt(fan_in).
        scale_factor: fator c para radius adaptativo.
        margin: margem para perda multiclasse (None = sem margem).
        homeostasis_target: taxa de disparo alvo (None = sem homeostase).
        homeostasis_tolerance: banda morta da homeostase.
        homeostasis_step: incremento do bias por ajuste.

    Returns:
        Dict com metricas da amostra.
    """
    scores, a_out, pre_activations, activations = net.forward_with_scores(
        x.reshape(1, -1).astype(np.float32)
    )
    out = activations[-1]

    # Perda e delta de saida
    if margin is not None:
        loss, delta_out = margin_loss(pre_activations[-1], target, margin)
    else:
        one_hot = np.zeros((1, net.output_dim), dtype=np.float32)
        one_hot[0, target] = 1.0
        error = out - one_hot
        loss = float(np.mean(np.abs(error)))
        z_output = pre_activations[-1]
        r_out = get_adaptive_radius(net.layers[2].input_dim, scale_factor) if adaptive_radius else radius
        fake_grad_output = triangular_fake_grad(z_output, scale=scale, radius=r_out)
        q_fake_grad = quantize_fake_grad(fake_grad_output, levels=levels)
        delta_out = (np.sign(error) * q_fake_grad).astype(np.float64)

    # Propagacao com surrogate quantizado nas ocultas
    deltas = [None, None, None]
    deltas[2] = delta_out

    for l in [1, 0]:
        r = get_adaptive_radius(net.layers[l].input_dim, scale_factor) if adaptive_radius else radius
        deltas[l] = propagate_delta_with_fake_quantized(
            deltas[l + 1], net.layers[l + 1].weights.T, pre_activations[l],
            scale=scale, radius=r, levels=levels,
        )

    # Acumulacao nos contadores
    for l in range(3):
        act_in = activations[l]
        dw_float = deltas[l].T @ act_in
        db_float = deltas[l][0]
        dw_int = np.round(dw_float).astype(np.int64)
        db_int = np.round(db_float).astype(np.int64)
        cm.accumulate(l, dw_int, db_int)

    # Atualizacao (mesma logica de train_sample_online)
    total_selected = 0
    total_blocked = 0
    total_transitions = 0
    bias_selected = 0
    bias_blocked = 0
    bias_transitions = 0

    for l, layer in enumerate(net.layers):
        Cw = cm.weight_counters[l]
        thr_w = max(1.0, float(np.mean(np.abs(Cw))))
        mask_w = np.abs(Cw) > thr_w
        rows, cols = np.where(mask_w)
        selected = len(rows)
        blocked = 0
        transitions = 0
        for r, c in zip(rows.tolist(), cols.tolist()):
            step = -int(np.sign(Cw[r, c]))
            if step != 0:
                w_old = int(layer.weights[r, c])
                w_new = int(np.clip(w_old + step, -1, 1))
                if w_new == w_old:
                    blocked += 1
                else:
                    layer.weights[r, c] = np.int8(w_new)
                    transitions += 1
        if selected:
            reset_error_feedback(
                Cw,
                np.array(rows, dtype=np.int64),
                np.array(cols, dtype=np.int64),
            )

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
            Cb[b_idx] = 0

        total_selected += selected
        total_blocked += blocked
        total_transitions += transitions

    # Homeostase (apos atualizacao)
    homeostasis_info = None
    if homeostasis_target is not None:
        firing_rates = [float(np.mean(a)) for a in activations[1:]]
        homeostasis_info = []
        for l in range(3):
            info = apply_homeostasis(
                net, l, firing_rates,
                target_rate=homeostasis_target,
                tolerance=homeostasis_tolerance,
                step_size=homeostasis_step,
            )
            homeostasis_info.append(info)

    metrics = {
        "loss": float(loss),
        "fired_neurons": int(np.sum(out[0] > 0)),
        "firing_rates": [float(np.mean(a)) for a in activations[1:]],
        "selected_weights": total_selected,
        "attempts": total_selected,
        "blocked_weights": total_blocked,
        "transitions": total_transitions,
        "updated_weights": total_transitions,
        "total_updates": sum(l.input_dim * l.output_dim for l in net.layers),
        "bias_selected": bias_selected,
        "bias_blocked": bias_blocked,
        "bias_transitions": bias_transitions,
    }
    if homeostasis_info is not None:
        metrics["homeostasis"] = homeostasis_info

    return metrics


# ---------------------------------------------------------------------------
# Testes unitarios
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testes — ablation.py")
    print("=" * 60)

    # Teste 1: margin_loss
    print("\n[1] margin_loss")
    z = np.array([[1.0, 2.0, 0.5]])
    loss, delta = margin_loss(z, target=0, margin=1.0)
    assert loss > 0, "Loss deveria ser positiva"
    assert delta[0, 0] == -1.0, "delta_y deveria ser -1"
    assert delta[0, 1] == 1.0, "delta_max_other deveria ser +1"
    print(f"   ✓ loss={loss:.4f}, delta={delta}")

    # Teste 2: margin_loss sem violacao
    z2 = np.array([[5.0, 2.0, 1.0]])
    loss2, delta2 = margin_loss(z2, target=0, margin=1.0)
    assert loss2 == 0.0, "Loss deveria ser zero"
    assert np.all(delta2 == 0), "delta deveria ser zero"
    print(f"   ✓ sem violacao: loss={loss2}")

    # Teste 3: apply_homeostasis
    print("\n[3] apply_homeostasis")
    net = TernaryNet(seed=42)
    fr = [0.8, 0.1, 0.5]
    info = apply_homeostasis(net, 0, fr, target_rate=0.25, tolerance=0.05)
    assert info["adjustment"] == -1, "bias deveria diminuir"
    print(f"   ✓ firing=0.8 > 0.25+0.05 => adjustment={info['adjustment']}")

    info2 = apply_homeostasis(net, 1, fr, target_rate=0.25, tolerance=0.05)
    assert info2["adjustment"] == 1, "bias deveria aumentar"
    print(f"   ✓ firing=0.1 < 0.25-0.05 => adjustment={info2['adjustment']}")

    info3 = apply_homeostasis(net, 2, [0.8, 0.1, 0.25], target_rate=0.25, tolerance=0.05)
    assert info3["adjustment"] == 0, "bias nao deveria mudar"
    print(f"   ✓ firing=0.25 dentro da banda => adjustment={info3['adjustment']}")

    # Teste 4: train_sample_ablation
    print("\n[4] train_sample_ablation")
    net2 = TernaryNet(seed=42)
    cm2 = CounterManager([(128, 784), (64, 128), (10, 64)])
    x = np.random.randn(784).astype(np.float32)
    m = train_sample_ablation(net2, cm2, x, 5, scale=1.0, radius=1.0, levels=5)
    assert "loss" in m
    assert "transitions" in m
    print(f"   ✓ loss={m['loss']:.4f}, transitions={m['transitions']}")

    # Teste 5: train_sample_ablation com margem
    print("\n[5] train_sample_ablation com margem")
    net3 = TernaryNet(seed=42)
    cm3 = CounterManager([(128, 784), (64, 128), (10, 64)])
    m2 = train_sample_ablation(net3, cm3, x, 5, margin=2.0)
    assert "loss" in m2
    print(f"   ✓ loss={m2['loss']:.4f}, transitions={m2['transitions']}")

    # Teste 6: train_sample_ablation com homeostase
    print("\n[6] train_sample_ablation com homeostase")
    net4 = TernaryNet(seed=42)
    cm4 = CounterManager([(128, 784), (64, 128), (10, 64)])
    m3 = train_sample_ablation(net4, cm4, x, 5, homeostasis_target=0.25)
    assert "homeostasis" in m3
    print(f"   ✓ homeostasis={m3['homeostasis']}")

    print("\n" + "=" * 60)
    print("Todos os testes passaram!")
    print("=" * 60)
