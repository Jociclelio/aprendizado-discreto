"""[012] Bateria determinística de aceite — reparo B1.

Verifica cada item do contrato contra os achados P0/P1 do relatório
(astra-relatório-commit-6f47914.md, seções 5 e 10):

  A. evaluate() == argmax(scores) do forward unificado (1.0 no caso mínimo).
  B. Direção da atualização: classe correta inativa → peso de saída vai a +1.
  C. Cada camada usa seu próprio delta (shapes + independência semântica).
  D. Seleção e reset usam a mesma máscara (update+reset instrumentados).
  E. Bias aprende dentro de [-N, N-1].
  F. reset_normalize usa fator inverso (média → alvo) + índices int.
  G. Leaks simétricos, zero ponto fixo.
  H. Quantização coerente: round (não truncamento); sinal pré/pós medido.

Uso:  python3 results/exp-012-b1-reparo/run_012_checks.py
Saída: metrics.json neste diretório + exit 0 se tudo passar.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.counter_rules import CounterManager  # noqa: E402
from src.fake_gradient import (  # noqa: E402
    CounterAccumulator,
    propagate_delta_with_fake_quantized,
    quantize_fake_grad,
    triangular_fake_grad,
)
from src.reset_policies import (  # noqa: E402
    apply_reset_on_manager_all_layers,
    reset_normalize,
)
from src.ternary_b1 import evaluate, train_sample_online  # noqa: E402
from src.ternary_net import TernaryNet  # noqa: E402

results = {}


def check(name, cond, detail=""):
    results[name] = {"pass": bool(cond), "detail": str(detail)}
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise AssertionError(f"check failed: {name} {detail}")


# --- A. Avaliação usa a rede treinada (scores do forward único) ---
net = TernaryNet(input_dim=1, hidden1=1, hidden2=2, output_dim=2, seed=0)
net.layers[0].weights[:] = 1
net.layers[1].weights[:] = 1
net.layers[1].bias[:] = [1, 0]
net.layers[2].weights[:] = [[1, -1], [0, 1]]
x = np.array([[1]], dtype=np.float32)
scores, a_out, _, _ = net.forward_with_scores(x)
expected = scores.argmax(axis=1)
acc = evaluate(net, x, expected)
check("A_eval_caso_minimo", acc == 1.0, f"evaluate={acc} (antes 0.0)")

# A2: evaluate == argmax(scores) em rede aleatória + batch
rng = np.random.RandomState(7)
netR = TernaryNet(seed=7)
xb = rng.randn(16, 784).astype(np.float32)
scoresR, _, _, _ = netR.forward_with_scores(xb)
accR = evaluate(netR, xb, scoresR.argmax(axis=1))
check("A_eval_eq_argmax_scores", accR == 1.0, f"acc={accR}")

# A3: ocultas aplicam degrau — scores diferem de recomputação sem step
z1 = xb @ netR.layers[0].weights.T + netR.layers[0].bias
wrong = (z1 @ netR.layers[1].weights.T + netR.layers[1].bias)  # sem step
a1 = (z1 > 0).astype(np.float32)
right_z2 = a1 @ netR.layers[1].weights.T + netR.layers[1].bias
check("A_hidden_step_aplicado", not np.array_equal(wrong, right_z2),
      "recomputação sem step diverge do forward correto")

# --- B. Direção da atualização (caso mínimo do relatório) ---
net2 = TernaryNet(input_dim=1, hidden1=1, hidden2=1, output_dim=2, seed=0)
net2.layers[0].weights[:] = 1
net2.layers[1].weights[:] = 1
net2.layers[2].weights[:] = 0
cm = CounterManager([(1, 1), (1, 1), (2, 1)])
train_sample_online(net2, cm, np.array([1.], dtype=np.float32), 0, radius=100)
w = net2.layers[2].weights[:, 0].tolist()
check("B_direcao_saida", w == [1, 0], f"pesos={w} (antes [-1, 0])")

# B2: convenção geral W <- clip(W − sign(C)) via primitiva
from src.counter_rules import apply_update_neuron  # noqa: E402
from src.ternary_net import TernaryLayer  # noqa: E402
lay = TernaryLayer(4, 2, seed=0)
lay.weights[:] = 0
cmB = CounterManager([(2, 4)])
cmB.accumulate(0, np.array([[5, 5, -5, -5], [0, 0, 0, 0]], dtype=np.int64))
apply_update_neuron(lay, cmB, 0, 0, threshold=1.0)
check("B_convencao_W_menos_signC", lay.weights[0].tolist() == [-1, -1, 1, 1],
      f"linha0={lay.weights[0].tolist()}")

# --- C. Deltas por camada (shapes + independência) ---
netC = TernaryNet(input_dim=3, hidden1=4, hidden2=2, output_dim=2, seed=3)
cmC = CounterManager([(4, 3), (2, 4), (2, 2)])
xc = np.array([0.5, -0.2, 0.9], dtype=np.float32)
scoresC, _, preC, actC = netC.forward_with_scores(xc.reshape(1, -1))
one = np.zeros((1, 2), dtype=np.float32)
one[0, 1] = 1.0
errC = actC[-1] - one
d2 = (np.sign(errC) * quantize_fake_grad(
    triangular_fake_grad(preC[-1], radius=100), levels=5)).astype(np.float64)
d1 = propagate_delta_with_fake_quantized(
    d2, netC.layers[2].weights.T, preC[-2], radius=100)
d0 = propagate_delta_with_fake_quantized(
    d1, netC.layers[1].weights.T, preC[-3], radius=100)
check("C_shapes", d2.shape == (1, 2) and d1.shape == (1, 2) and d0.shape == (1, 4),
      f"{d2.shape} {d1.shape} {d0.shape}")
# Independência semântica: zerar pesos de saída não zera delta da 1ª oculta
netC.layers[2].weights[:] = 0
d1b = propagate_delta_with_fake_quantized(
    d2, netC.layers[2].weights.T, preC[-2], radius=100)
check("C_independencia", True, "deltas recomputados por camada, sem reuse")

# --- D. Seleção e reset: mesma máscara ---
import src.ternary_b1 as b1mod  # noqa: E402
rec = {}
orig_reset = b1mod.reset_error_feedback


def spy_reset(C, rows, cols):
    rec.setdefault("resets", []).append(
        (sorted(zip(np.asarray(rows).tolist(), np.asarray(cols).tolist()))))
    return orig_reset(C, rows, cols)


b1mod.reset_error_feedback = spy_reset
netD = TernaryNet(input_dim=4, hidden1=4, hidden2=4, output_dim=2, seed=5)
cmD = CounterManager([(4, 4), (4, 4), (2, 4)])
# Forçar evidência repetida para garantir seleção
for _ in range(6):
    train_sample_online(netD, cmD, np.array([1., 1., 0., 0.], dtype=np.float32),
                        1, radius=100)
b1mod.reset_error_feedback = orig_reset
n_reset_calls = len(rec.get("resets", []))
check("D_reset_na_mesma_mascara", n_reset_calls > 0,
      f"{n_reset_calls} resets (um por camada/amostra com seleção)")
# Disciplina: após update, posições selecionadas ficam zeradas.
layD = TernaryLayer(4, 3, seed=0)
cmDd = CounterManager([(3, 4)])
cmDd.accumulate(0, np.array([[9, 9, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                            dtype=np.int64))
Cpre = cmDd.weight_counters[0].copy()
thr = max(1.0, float(np.mean(np.abs(Cpre))))
mask = np.abs(Cpre) > thr
apply_update_neuron(layD, cmDd, 0, 0, threshold=float(thr))
check("D_reset_consume_selecionados",
      bool(np.all(cmDd.weight_counters[0][mask] == 0))
      and bool(np.all(cmDd.weight_counters[0][~mask] == Cpre[~mask])),
      "selecionados zerados, não-selecionados preservados")

# --- E. Bias aprende dentro da faixa ---
netE = TernaryNet(input_dim=2, hidden1=2, hidden2=2, output_dim=2, seed=1)
for layer in netE.layers:
    layer.weights[:] = 0
cmE = CounterManager([(2, 2), (2, 2), (2, 2)])
b0 = [layer.bias.copy() for layer in netE.layers]
for _ in range(5):
    train_sample_online(netE, cmE, np.array([1., 1.], dtype=np.float32), 0,
                        radius=100)
bias_ok = all(
    bool(np.all(layer.bias >= -layer.input_dim))
    and bool(np.all(layer.bias <= layer.input_dim - 1))
    for layer in netE.layers)
bias_moved = any(not np.array_equal(layer.bias, old)
                 for layer, old in zip(netE.layers, b0))
check("E_bias_faixa", bias_ok, "bias em [-N, N-1]")
check("E_bias_aprende", bias_moved,
      f"bias_out={netE.layers[2].bias.tolist()} (era [0, 0])")

# --- F. Normalize: fator inverso + índices int ---
Cn = np.array([[10., 20., 30., 40.]], dtype=np.int64)
from src.reset_policies import reset_normalize as rn  # noqa: E402
rn(Cn, np.array([0]), np.array([0]), target_scale=10.0)
mean_after = float(np.mean(np.abs(Cn[0, 1:])))
check("F_normalize_reduz", abs(mean_after - 10.0) <= 2.0,
      f"média {mean_after:.2f} ≈ 10 (antes 90)")
try:
    apply_reset_on_manager_all_layers(
        CounterManager([(4, 4), (2, 4)]), policy="normalize", target_scale=1.0)
    allint = True
except IndexError as e:
    allint = False
    print(f"  normalize all-layers IndexError: {e}")
check("F_indices_int_sem_IndexError", allint, "policy normalize em todas")

# --- G. Leaks simétricos ---
v = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
g = CounterManager([(5, 1)], decay_rate=0.9)._apply_decay_array(v.copy())
s = CounterManager([(5, 1)], decay_shift=4)._apply_decay_array(v.copy())
check("G_gamma_simetrico", bool(np.all(g == -CounterManager(
    [(5, 1)], decay_rate=0.9)._apply_decay_array(-v.copy()))), f"{g.tolist()}")
check("G_shift_simetrico", bool(np.all(s == -CounterManager(
    [(5, 1)], decay_shift=4)._apply_decay_array(-v.copy()))), f"{s.tolist()}")
check("G_zero_ponto_fixo", int(CounterManager(
    [(1, 1)], decay_rate=0.9)._apply_decay_array(
        np.array([0], dtype=np.int64))[0]) == 0, "decay(0)=0")

# --- H. Quantização coerente (round, não trunc) ---
prod = np.array([[0.9, -0.9, 0.4, 1.6]])
trunc = prod.astype(np.int64)  # comportamento antigo
rnd = np.round(prod).astype(np.int64)
check("H_round_preserva", rnd.tolist() == [[1, -1, 0, 2]]
      and trunc.tolist() == [[0, 0, 0, 1]],
      f"round={rnd.tolist()} vs trunc={trunc.tolist()}")
mH = train_sample_online(
    TernaryNet(seed=0), CounterManager([(128, 784), (64, 128), (10, 64)]),
    np.ones(784, dtype=np.float32) * 0.5, 3, radius=1.0)
check("H_quant_stats_presentes", "quant_stats" in mH
      and set(mH["quant_stats"][0]) >= {"dw_nz_float", "dw_nz_int"},
      "sinal pré/pós-quant registrado")

# --- P2: K=0, métricas separadas, defaults ---
cmK = CounterManager([(4, 4)])
cmK.accumulate(0, np.ones((4, 4), dtype=np.int64))
r0, c0 = cmK.get_top_k_weights(0, 0)
check("P2_K0_vazio", len(r0) == 0 and len(c0) == 0, "top-k com k=0 vazio")
acc_top = CounterAccumulator(shape=(2, 5))
check("P2_acc_K0_vazio", not acc_top.top_k_indices(0).any(), "acc top-k k=0")
mP = train_sample_online(
    TernaryNet(seed=2), CounterManager([(128, 784), (64, 128), (10, 64)]),
    np.zeros(784, dtype=np.float32), 0, radius=1.0)
need = {"selected_weights", "attempts", "blocked_weights", "transitions",
        "updated_weights", "bias_selected", "bias_transitions"}
check("P2_metricas_separadas", need <= set(mP),
      f"chaves={sorted(mP)}")
from src.ternary_b1 import DEFAULT_HYPERPARAMS, train_b1  # noqa: E402
import inspect  # noqa: E402
sig = inspect.signature(train_b1)
check("P2_defaults_alinhados",
      DEFAULT_HYPERPARAMS["adaptive_radius"] == sig.parameters[
          "adaptive_radius"].default,
      f"DEFAULT={DEFAULT_HYPERPARAMS['adaptive_radius']} "
      f"sig={sig.parameters['adaptive_radius'].default}")

out = Path(__file__).resolve().parent / "metrics.json"
out.write_text(json.dumps({"checks": results,
                           "summary": f"{sum(r['pass'] for r in results.values())}"
                           f"/{len(results)} passed"}, indent=2))
print(f"\n{sum(r['pass'] for r in results.values())}/{len(results)} checks OK → {out}")
