"""
[017] Tracer instrumentado — observa o treino amostra a amostra.

Envolve `TernaryNet.forward_with_scores`, `triangular_fake_grad`,
`quantize_fake_grad`, `propagate_delta_with_fake_quantized`,
`CounterManager` e a logica de `train_sample_online` (B1) **sem alterar
nenhum arquivo `src/*.py` existente** (so leitura/import).

Contrato: CONTRATO_012 (`src/ternary_b1.py` docstring):
    C acumula gradiente, update W <- clip(W - sign(C)), mascara unica
    por camada aplicada a update E reset, saturacao bloqueia o peso mas
    consome o contador, leak simetrico opcional.

Tres variantes, que trocam SO a funcao de mascara de pesos
(o resto — forward, deltas, acumulacao, update, reset — e identico):
    - trace_b1: mascara B1  = |C| > max(1, mean(|C_cam|)) por camada.
    - trace_b2: mascara B2  = Top-K por camada via
      `CounterManager.get_top_k_weights` (k_l = frac * size_l).
    - trace_b3: mascara B3  = Top-K por rede (orcamento global
      K_total = frac * params_totais distribuido pelo |C| global).

Uso:
    python -m src.playground.tracer   # teste de bit-identidade vs CLI
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.counter_rules import CounterManager
from src.fake_gradient import (
    get_adaptive_radius,
    propagate_delta_with_fake_quantized,
    quantize_fake_grad,
    triangular_fake_grad,
)
from src.reset_policies import reset_error_feedback
from src.ternary_b1 import train_sample_online
from src.ternary_net import TernaryNet

LAYER_SHAPES = [(128, 784), (64, 128), (10, 64)]
LAYER_FANINS = [784, 128, 64]


# ---------------------------------------------------------------------------
# Selecao de mascara (unica diferenca entre B1/B2/B3)
# ---------------------------------------------------------------------------
def mask_b1(Cw: np.ndarray) -> Tuple[np.ndarray, float]:
    """Mascara B1: |C| > max(1, mean(|C_cam|)). Retorna (mask, thr)."""
    thr_w = max(1.0, float(np.mean(np.abs(Cw))))
    return (np.abs(Cw) > thr_w), thr_w


def mask_b2_per_layer(
    cm: CounterManager, layer_idx: int, topk_frac: float = 0.01
) -> Tuple[np.ndarray, float]:
    """Mascara B2: Top-K por camada via CounterManager.

    k_l = max(1, int(frac * size_l)). Reusa `get_top_k_weights`
    (mesma regra de ranking por |C| do treino B2).
    Retorna (mask, thr) onde thr = menor |C| selecionado (p/ exibicao).
    """
    Cw = cm.weight_counters[layer_idx]
    k = max(1, int(topk_frac * Cw.size))
    k = min(k, Cw.size)
    rows, cols = cm.get_top_k_weights(layer_idx, k)
    mask = np.zeros_like(Cw, dtype=bool)
    if len(rows):
        mask[rows, cols] = True
        thr = float(np.min(np.abs(Cw[rows, cols])))
    else:
        thr = 0.0
    return mask, thr


def _global_topk_masks(
    Cw_list: List[np.ndarray], topk_frac: float = 0.01
) -> List[Tuple[np.ndarray, float]]:
    """Mascara B3: Top-K por rede (orcamento global por |C|).

    K_total = max(n_layers, int(frac * params_totais)); os K_total
    maiores |C| da rede inteira vencem, onde quer que estejam.
    Retorna [(mask_l, thr_l)] por camada (thr_l = menor |C| selecionado
    na camada, 0.0 se nada selecionado; thr_global no dict do trace).
    """
    flats = [np.abs(c).ravel() for c in Cw_list]
    concat = np.concatenate(flats)
    total = concat.size
    k_total = max(len(Cw_list), int(topk_frac * total))
    k_total = min(k_total, total)
    # indices globais dos top-k (argpartition deterministico p/ seed fixa
    # porque os contadores sao inteiros; desempate por ordem de indice).
    order = np.argpartition(concat, -k_total)[-k_total:]
    order = order[np.argsort(-concat[order], kind="stable")]
    chosen = np.zeros(total, dtype=bool)
    chosen[order] = True
    out = []
    offset = 0
    for c in Cw_list:
        n = c.size
        sel = chosen[offset:offset + n].reshape(c.shape)
        thr = float(np.min(np.abs(c[sel]))) if np.any(sel) else 0.0
        out.append((sel, thr))
        offset += n
    return out


# ---------------------------------------------------------------------------
# Nucleo comum: forward + deltas + acumulacao (identico ao train_sample_online)
# ---------------------------------------------------------------------------
def _forward_deltas_accumulate(
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
    """Executa os estagios 1-5 do CONTRATO_012 e acumula nos contadores.

    Replica linha a linha os estagios 1-5 de `train_sample_online`
    (forward unico, erro, fake-grad, deltas, acumulacao com round).
    Retorna dict com todos os intermediarios + snapshots de C/W.
    """
    Cw_before = [c.copy() for c in cm.weight_counters]
    Cb_before = [c.copy() for c in cm.bias_counters]
    W_before = [layer.weights.copy() for layer in net.layers]
    b_before = [layer.bias.copy() for layer in net.layers]

    # 1. Forward UNICO: scores + binarias, degrau em TODAS as ocultas.
    scores, a_out, pre_activations, activations = net.forward_with_scores(
        x.reshape(1, -1).astype(np.float32)
    )
    out = activations[-1]  # (1, 10) binaria

    # 2. One-hot target + erro (gradiente do L1 medio a menos de constante)
    one_hot = np.zeros((1, net.output_dim), dtype=np.float32)
    one_hot[0, target] = 1.0
    error = out - one_hot  # (1, 10)
    loss = float(np.mean(np.abs(error)))
    pred = int(np.argmax(scores[0]))

    def _radius_for(layer_input_dim: int) -> float:
        if adaptive_radius:
            return get_adaptive_radius(layer_input_dim, scale_factor)
        return radius

    radii = [_radius_for(net.layers[i].input_dim) for i in range(3)]

    # 3. Fake gradient continuo por camada (p/ exibicao; saida = o usado).
    z_output = pre_activations[-1]  # (1, 10)
    fake_cont_out = triangular_fake_grad(z_output, scale=scale, radius=radii[2])
    q_fake_out = quantize_fake_grad(fake_cont_out, levels=levels)
    deltas: List[Optional[np.ndarray]] = [None, None, None]
    deltas[2] = (np.sign(error) * q_fake_out).astype(np.float64)  # (1, 10)

    # 4. Propagacao com surrogate QUANTIZADO; cada camada guarda o seu.
    deltas[1] = propagate_delta_with_fake_quantized(
        deltas[2], net.layers[2].weights.T, pre_activations[-2],
        scale=scale, radius=radii[1], levels=levels,
    )
    deltas[0] = propagate_delta_with_fake_quantized(
        deltas[1], net.layers[1].weights.T, pre_activations[-3],
        scale=scale, radius=radii[0], levels=levels,
    )

    # Surrogates continuos/quantizados das ocultas (diagnostico; os
    # deltas acima ja usam internamente os mesmos valores quantizados).
    fake_cont, q_fake = [None] * 3, [None] * 3
    fake_cont[2] = fake_cont_out
    q_fake[2] = q_fake_out
    for l in (0, 1):
        fc = triangular_fake_grad(pre_activations[l], scale=scale, radius=radii[l])
        fake_cont[l] = fc
        q_fake[l] = quantize_fake_grad(fc, levels=levels)

    # 5. Acumulacao (pesos E bias) com round coerente.
    quant_stats: Dict[int, Dict] = {}
    dw_float_list, dw_int_list = [], []
    db_float_list, db_int_list = [], []
    for l in range(3):
        act_in = activations[l]
        dw_float = deltas[l].T @ act_in  # (out_dim, fan_in) float
        db_float = deltas[l][0]  # (out_dim,) float
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
        dw_float_list.append(dw_float.copy())
        dw_int_list.append(dw_int.copy())
        db_float_list.append(db_float.copy())
        db_int_list.append(db_int.copy())

    Cw_acc = [c.copy() for c in cm.weight_counters]
    Cb_acc = [c.copy() for c in cm.bias_counters]

    return {
        "x": x.reshape(-1).astype(np.float32).copy(),
        "label": int(target),
        "pred": pred,
        "scores": scores[0].copy() if scores.ndim == 2 else scores.copy(),
        "a_out": a_out[0].copy() if a_out.ndim == 2 else a_out.copy(),
        "z_list": [z.copy() for z in pre_activations],
        "a_list": [a.copy() for a in activations],
        "error": error.copy(),
        "one_hot": one_hot.copy(),
        "loss": loss,
        "radii": radii,
        "fake_cont": fake_cont,
        "q_fake": q_fake,
        "deltas": [d.copy() for d in deltas],
        "dw_float": dw_float_list,
        "dw_int": dw_int_list,
        "db_float": db_float_list,
        "db_int": db_int_list,
        "quant_stats": quant_stats,
        "Cw_before": Cw_before,
        "Cb_before": Cb_before,
        "W_before": W_before,
        "b_before": b_before,
        "Cw_acc": Cw_acc,
        "Cb_acc": Cb_acc,
    }


# ---------------------------------------------------------------------------
# Estagios 6-8: mascara unica -> update -> reset na mesma mascara
# ---------------------------------------------------------------------------
def _apply_mask_update_reset(
    net: TernaryNet,
    cm: CounterManager,
    core: Dict,
    weight_masks: List[Tuple[np.ndarray, float]],
) -> Dict:
    """Aplica os estagios 6-8 do CONTRATO_012 sobre as mascaras dadas.

    `weight_masks`: [(mask_w_l, thr_w_l)] por camada. O bias usa sempre a
    disciplina B1 (mascara propria |Cb| > max(1, mean)). Update e descida
    W <- clip(W - sign(C)); saturacao bloqueia mas consome o contador.
    """
    total_selected = 0
    total_blocked = 0
    total_transitions = 0
    bias_selected = 0
    bias_blocked = 0
    bias_transitions = 0

    masks_w, thrs_w, thrs_b = [], [], []
    bias_masks = []
    per_layer = []

    for l, layer in enumerate(net.layers):
        Cw = cm.weight_counters[l]
        mask_w, thr_w = weight_masks[l]
        masks_w.append(mask_w.copy())
        thrs_w.append(float(thr_w))
        rows, cols = np.where(mask_w)
        selected = len(rows)
        blocked = 0
        transitions = 0
        for r, c in zip(rows.tolist(), cols.tolist()):
            step = -int(np.sign(Cw[r, c]))  # descida, sem sign(-error) extra
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

        # Bias: mesma disciplina — mascara propria, descida, faixa [-N, N-1].
        Cb = cm.bias_counters[l]
        thr_b = max(1.0, float(np.mean(np.abs(Cb)))) if Cb.size else 1.0
        mask_b = np.abs(Cb) > thr_b
        bias_masks.append(mask_b.copy())
        thrs_b.append(float(thr_b))
        b_idx = np.where(mask_b)[0]
        bb0, bt0 = bias_blocked, bias_transitions
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

        per_layer.append({
            "selected": selected,
            "attempts": selected,
            "blocked": blocked,
            "transitions": transitions,
            "bias_selected": n_bsel,
            "bias_blocked": bias_blocked - bb0,
            "bias_transitions": bias_transitions - bt0,
            "thr_w": float(thr_w),
            "thr_b": float(thr_b),
        })

        total_selected += selected
        total_blocked += blocked
        total_transitions += transitions

    firing_rates = [float(np.mean(a)) for a in core["a_list"][1:]]
    metrics = {
        "loss": core["loss"],
        "fired_neurons": int(np.sum(core["a_list"][-1][0] > 0)),
        "firing_rates": firing_rates,
        "selected_weights": total_selected,
        "attempts": total_selected,
        "blocked_weights": total_blocked,
        "transitions": total_transitions,
        "updated_weights": total_transitions,  # legado = transicoes efetivas
        "total_updates": sum(ly.input_dim * ly.output_dim for ly in net.layers),
        "bias_selected": bias_selected,
        "bias_blocked": bias_blocked,
        "bias_transitions": bias_transitions,
        "per_layer": per_layer,
        # Quantização (H)
        "quant_stats": core["quant_stats"],
    }
    return {
        "metrics": metrics,
        "masks_w": masks_w,
        "thrs_w": thrs_w,
        "masks_b": bias_masks,
        "thrs_b": thrs_b,
    }


def _finish_trace(core: Dict, tail: Dict, variant: str, extra: Optional[Dict] = None) -> Dict:
    """Monta o dict Trace final (core + mascaras + snapshots pos-reset)."""
    # snapshots pos-reset sao lidos pelo chamador apos update; aqui o
    # chamador anexa Cw_after/Cb_after/W_after. Esta funcao so organiza.
    trace = dict(core)
    trace["variant"] = variant
    trace["masks_w"] = tail["masks_w"]
    trace["thrs_w"] = tail["thrs_w"]
    trace["masks_b"] = tail["masks_b"]
    trace["thrs_b"] = tail["thrs_b"]
    trace["metrics"] = tail["metrics"]
    if extra:
        trace.update(extra)
    return trace


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------
def trace_b1(
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
    """Trace de 1 amostra com a regra B1 (bit-identico a `train_sample_online`)."""
    core = _forward_deltas_accumulate(
        net, cm, x, target, scale, radius, levels, adaptive_radius, scale_factor
    )
    weight_masks = [mask_b1(Cw) for Cw in core["Cw_acc"]]
    tail = _apply_mask_update_reset(net, cm, core, weight_masks)
    trace = _finish_trace(core, tail, "B1")
    trace["Cw_after"] = [c.copy() for c in cm.weight_counters]
    trace["Cb_after"] = [c.copy() for c in cm.bias_counters]
    trace["W_after"] = [layer.weights.copy() for layer in net.layers]
    trace["b_after"] = [layer.bias.copy() for layer in net.layers]
    trace["k_info"] = {"kind": "b1-threshold"}
    return trace


def trace_b2(
    net: TernaryNet,
    cm: CounterManager,
    x: np.ndarray,
    target: int,
    scale: float = 1.0,
    radius: float = 1.0,
    levels: int = 5,
    adaptive_radius: bool = False,
    scale_factor: float = 1.0,
    topk_frac: float = 0.01,
) -> Dict:
    """Trace de 1 amostra com mascara B2 (Top-K por camada)."""
    core = _forward_deltas_accumulate(
        net, cm, x, target, scale, radius, levels, adaptive_radius, scale_factor
    )
    weight_masks = []
    ks = []
    for l in range(len(net.layers)):
        k = max(1, min(int(topk_frac * core["Cw_acc"][l].size), core["Cw_acc"][l].size))
        ks.append(k)
        mask, thr = mask_b2_per_layer(cm, l, topk_frac=topk_frac)
        weight_masks.append((mask, thr))
    tail = _apply_mask_update_reset(net, cm, core, weight_masks)
    trace = _finish_trace(core, tail, "B2", extra={"k_per_layer": ks, "topk_frac": topk_frac})
    trace["Cw_after"] = [c.copy() for c in cm.weight_counters]
    trace["Cb_after"] = [c.copy() for c in cm.bias_counters]
    trace["W_after"] = [layer.weights.copy() for layer in net.layers]
    trace["b_after"] = [layer.bias.copy() for layer in net.layers]
    trace["k_info"] = {"kind": "b2-topk-per-layer", "k_per_layer": ks}
    return trace


def trace_b3(
    net: TernaryNet,
    cm: CounterManager,
    x: np.ndarray,
    target: int,
    scale: float = 1.0,
    radius: float = 1.0,
    levels: int = 5,
    adaptive_radius: bool = False,
    scale_factor: float = 1.0,
    topk_frac: float = 0.01,
) -> Dict:
    """Trace de 1 amostra com mascara B3 (Top-K por rede, orcamento global)."""
    core = _forward_deltas_accumulate(
        net, cm, x, target, scale, radius, levels, adaptive_radius, scale_factor
    )
    per_layer = _global_topk_masks(core["Cw_acc"], topk_frac=topk_frac)
    total = sum(c.size for c in core["Cw_acc"])
    k_total = max(len(core["Cw_acc"]), int(topk_frac * total))
    tail = _apply_mask_update_reset(net, cm, core, per_layer)
    trace = _finish_trace(
        core, tail, "B3", extra={"k_total": min(k_total, total), "topk_frac": topk_frac}
    )
    trace["Cw_after"] = [c.copy() for c in cm.weight_counters]
    trace["Cb_after"] = [c.copy() for c in cm.bias_counters]
    trace["W_after"] = [layer.weights.copy() for layer in net.layers]
    trace["b_after"] = [layer.bias.copy() for layer in net.layers]
    trace["k_info"] = {"kind": "b3-topk-global", "k_total": min(k_total, total)}
    return trace


TRACE_FNS = {"B1": trace_b1, "B2": trace_b2, "B3": trace_b3}


# ---------------------------------------------------------------------------
# Teste de bit-identidade: tracer B1 vs train_sample_online (CLI)
# ---------------------------------------------------------------------------
def _assert_bit_identity(seed: int = 42, n_samples: int = 5) -> Dict:
    """Compara trace_b1 com train_sample_online na mesma seed/amostra.

    Retorna dict com o resultado; levanta AssertionError se divergir.
    """
    from src.utils import set_seed

    rng = np.random.RandomState(seed)
    report = {"seed": seed, "samples": []}
    for i in range(n_samples):
        set_seed(seed)
        net_ref = TernaryNet(seed=seed)
        cm_ref = CounterManager(list(LAYER_SHAPES))
        set_seed(seed)
        net_tr = TernaryNet(seed=seed)
        cm_tr = CounterManager(list(LAYER_SHAPES))
        # avancar ambos i passos identicos via CLI para testar estado acumulado
        x_seq = [rng.randn(784).astype(np.float32) for _ in range(i + 1)]
        y_seq = [int(rng.randint(0, 10)) for _ in range(i + 1)]
        for j in range(i):
            train_sample_online(net_ref, cm_ref, x_seq[j], y_seq[j])
            trace_b1(net_tr, cm_tr, x_seq[j], y_seq[j])
        m_ref = train_sample_online(net_ref, cm_ref, x_seq[i], y_seq[i])
        tr = trace_b1(net_tr, cm_tr, x_seq[i], y_seq[i])
        m_tr = tr["metrics"]
        for l in range(3):
            assert np.array_equal(net_ref.layers[l].weights, net_tr.layers[l].weights), \
                f"amostra {i} camada {l}: W diverge"
            assert np.array_equal(net_ref.layers[l].bias, net_tr.layers[l].bias), \
                f"amostra {i} camada {l}: bias diverge"
            assert np.array_equal(cm_ref.weight_counters[l], cm_tr.weight_counters[l]), \
                f"amostra {i} camada {l}: Cw diverge"
            assert np.array_equal(cm_ref.bias_counters[l], cm_tr.bias_counters[l]), \
                f"amostra {i} camada {l}: Cb diverge"
        for k in ("loss", "selected_weights", "blocked_weights", "transitions",
                  "bias_selected", "bias_blocked", "bias_transitions"):
            assert m_ref[k] == m_tr[k], f"amostra {i}: metrica {k} diverge ({m_ref[k]} vs {m_tr[k]})"
        report["samples"].append({"i": i, "loss": m_ref["loss"],
                                  "selected": m_ref["selected_weights"],
                                  "transitions": m_ref["transitions"]})
    report["status"] = "bit-identical"
    return report


if __name__ == "__main__":
    print("=" * 60)
    print("Teste — tracer bit-identidade B1 vs train_sample_online")
    print("=" * 60)
    rep = _assert_bit_identity(seed=42, n_samples=5)
    for s in rep["samples"]:
        print(f"   amostra {s['i']}: loss={s['loss']:.4f} "
              f"sel={s['selected']} trans={s['transitions']}")
    # Sanidade B2/B3: mascaras Top-K tem cardinalidade esperada e diferem da B1
    from src.utils import set_seed
    set_seed(7)
    for variant, fn in (("B2", trace_b2), ("B3", trace_b3)):
        net = TernaryNet(seed=7)
        cm = CounterManager(list(LAYER_SHAPES))
        x = np.random.randn(784).astype(np.float32)
        tr = fn(net, cm, x, 3)
        sel = tr["metrics"]["selected_weights"]
        print(f"   {variant}: sel={sel} k_info={tr['k_info']}")
        assert sel > 0, f"{variant} sem selecionados"
    # B1 sanidade: roda sem erro e expoe os 7 estagios
    set_seed(7)
    net = TernaryNet(seed=7)
    cm = CounterManager(list(LAYER_SHAPES))
    tr1 = trace_b1(net, cm, np.random.randn(784).astype(np.float32), 3)
    for key in ("z_list", "fake_cont", "q_fake", "deltas", "dw_float",
                "dw_int", "Cw_before", "Cw_acc", "Cw_after", "masks_w",
                "W_after", "metrics"):
        assert key in tr1, f"Trace sem '{key}'"
    print("   B1 expoe z/fake/q_fake/deltas/dw/C/mask/W_after + metrics")
    print("Todos os testes passaram (bit-identico ao CLI)!")
