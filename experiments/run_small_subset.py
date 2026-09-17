"""
[013] Prova de capacidade de ajuste em subconjunto pequeno.

Demonstra (ou refuta, com diagnostico) que o B1 corrigido [012] aprende
de fato num subconjunto pequeno e fixo de MNIST, com metricas auditaveis.

Uso:
    python experiments/run_small_subset.py --seed 42 --n 64
    python experiments/run_small_subset.py --seed 7 --n 32 --epochs 3

Contrato [012] (reutilizado, nao redefinido):
    z_l = W_l @ x_{l-1} + b_l, a_l = step(z_l), scores = z_out,
    C acumula GRADIENTE => W <- clip(W - sign(C)), b <- clip(b - sign(C_b)).
    Usa a API unificada: TernaryNet.forward_with_scores (scores + binarias)
    e train_sample_online de src.ternary_b1 (deltas por camada, mascara
    unica de selecao, contadores separados selecionados/tentativas/
    bloqueios/transicoes, quant_stats pre/pos-quantizacao).

O que o script faz:
    1. Fixa subconjunto de N amostras via semente (indices salvos em JSON).
    2. Mede acuracia ANTES (rede inicial) no proprio subconjunto + baseline
       preditor majoritario.
    3. Treina B1 por orcamento pequeno e fixo (epochs sobre o subconjunto).
    4. Mede acuracia DEPOIS no mesmo subconjunto.
    5. A cada diagnostico (a cada `diag_every` passos + fronteiras de epoch)
       registra: scores z_out por amostra, distribuicao de preds, matriz de
       confusao, taxa de atividade por camada, surrogate continuo vs
       quantizado (fracao nao-nula por camada), deltas por camada
       (amostra-sonda), incremento float vs int (via quant_stats), e os
       contadores separados do train_sample_online.
    6. Detecta o PRIMEIRO ponto de falha, se houver: colapso de diversidade
       (todas as preds iguais), morte do surrogate de saida (q_fake(z_out)
       todo zero => deltas zero => treino congela), epoch congelada
       (0 transicoes efetivas).
    7. Salva tudo em results/exp-013-small-subset/ + REPORT.md com veredito
       PASS/FAIL segundo os criterios da task.

Criterio de sucesso (task 013):
    dacc > 0 E acc_final > acc_majoritario E acc_final > acc_inicial.
    Se falhar, o REPORT.md aponta o primeiro ponto onde sinal/diversidade
    some (surrogate zero => escala/raio; delta cancelado => sinal;
    incremento truncado => quantizacao), sem introduzir margem/homeostase
    (isso e 015).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.counter_rules import CounterManager
from src.fake_gradient import (
    get_adaptive_radius,
    propagate_delta_with_fake_quantized,
    quantize_fake_grad,
    triangular_fake_grad,
)
from src.ternary_b1 import evaluate, train_sample_online
from src.ternary_net import TernaryNet
from src.utils import ensure_dir, load_mnist, save_metrics, set_seed

ARCH_SHAPES = [(128, 784), (64, 128), (10, 64)]
FAN_INS = [784, 128, 64]


# ---------------------------------------------------------------------------
# Utilidades de diagnostico (somente leitura — nao alteram pesos/contadores)
# ---------------------------------------------------------------------------
def confusion_matrix(preds: np.ndarray, labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Matriz de confusao (linhas=true, colunas=pred) como int."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def diagnose(net: TernaryNet, sub_x: np.ndarray, sub_y: np.ndarray,
             scale: float, scale_factor: float, levels: int,
             probe_idx: int = 0) -> dict:
    """Passada de diagnostico sobre o subconjunto (sem treinar).

    Retorna dict JSON-serializavel com scores, distribuicao, confusao,
    atividade, |z| por camada, surrogate cont/quant e deltas da amostra-sonda.
    """
    radii = [get_adaptive_radius(d, scale_factor) for d in FAN_INS]
    scores, _, pre, acts = net.forward_with_scores(sub_x.astype(np.float32))
    preds = np.argmax(scores, axis=1)
    firing = [float(np.mean(a)) for a in acts[1:]]

    layers = []
    for l in range(3):
        fc = triangular_fake_grad(pre[l], scale=scale, radius=radii[l])
        qf = quantize_fake_grad(fc, levels=levels)
        layers.append({
            "radius": radii[l],
            "z_mean_abs": float(np.mean(np.abs(pre[l]))),
            "z_max_abs": float(np.max(np.abs(pre[l]))),
            "surrogate_cont_nz_frac": float(np.mean(fc != 0)),
            "surrogate_quant_nz_frac": float(np.mean(qf != 0)),
        })

    # Deltas da amostra-sonda (mesma conta do train_sample_online, read-only).
    sc, ao, pr, ac = net.forward_with_scores(
        sub_x[probe_idx].reshape(1, -1).astype(np.float32))
    tgt = int(sub_y[probe_idx])
    oh = np.zeros((1, net.output_dim), dtype=np.float32)
    oh[0, tgt] = 1.0
    err = ao - oh
    qf_out = quantize_fake_grad(
        triangular_fake_grad(pr[-1], scale=scale, radius=radii[2]),
        levels=levels).astype(np.float64)
    d2 = (np.sign(err) * qf_out).astype(np.float64)
    d1 = propagate_delta_with_fake_quantized(
        d2, net.layers[2].weights.T, pr[-2],
        scale=scale, radius=radii[1], levels=levels)
    d0 = propagate_delta_with_fake_quantized(
        d1, net.layers[1].weights.T, pr[-3],
        scale=scale, radius=radii[0], levels=levels)
    probe = {
        "idx": probe_idx, "target": tgt,
        "error_nz": int(np.sum(err != 0)),
        "d_out_nz": int(np.sum(d2 != 0)),
        "d_h2_nz": int(np.sum(d1 != 0)),
        "d_h1_nz": int(np.sum(d0 != 0)),
    }

    return {
        "pred_dist": np.bincount(preds, minlength=10).tolist(),
        "collapsed_single_class": bool(len(set(preds.tolist())) == 1),
        "confusion": confusion_matrix(preds, sub_y).tolist(),
        "firing_rates": firing,
        "layers": layers,
        "probe": probe,
    }


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def run_experiment(seed: int = 42, n: int = 64, epochs: int = 3,
                   scale: float = 1.0, scale_factor: float = 1.0,
                   levels: int = 5, diag_every: int = 8,
                   outdir: str = "results/exp-013-small-subset") -> dict:
    set_seed(seed)
    out = ensure_dir(Path(outdir))

    # 1. Subconjunto fixo pela semente (indices salvos => reproduzivel).
    print("Carregando MNIST...")
    train_data, train_labels, _, _ = load_mnist(normalize=True)
    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(train_data.shape[0], size=n, replace=False))
    sub_x = train_data[indices]
    sub_y = train_labels[indices]
    counts = np.bincount(sub_y, minlength=10)
    maj_class = int(np.argmax(counts))
    maj_acc = float(counts.max() / n)
    with open(out / "subset.json", "w") as f:
        json.dump({"seed": seed, "n": n, "indices": indices.tolist(),
                   "labels": sub_y.tolist(),
                   "label_counts": counts.tolist(),
                   "majority_class": maj_class,
                   "majority_accuracy": maj_acc}, f, indent=2)
    print(f"  Subconjunto: n={n} seed={seed} "
          f"maj=classe {maj_class} acc={maj_acc:.4f}")

    # 2. Rede inicial + acuracia ANTES + diagnostico ANTES.
    net = TernaryNet(seed=seed)
    cm = CounterManager(ARCH_SHAPES)
    acc_before = float(evaluate(net, sub_x, sub_y))
    diag_before = diagnose(net, sub_x, sub_y, scale, scale_factor, levels)
    scores_before, _, _, _ = net.forward_with_scores(
        sub_x.astype(np.float32))
    np.save(out / "scores_before.npy", np.asarray(scores_before))
    print(f"  Acc ANTES: {acc_before:.4f} "
          f"(preds: {diag_before['pred_dist']})")

    # 3. Treino por orcamento fixo, com log por amostra e diagnosticos.
    step_log = []     # 1 linha por amostra treinada
    diag_trail = [{"step": 0, "epoch": 0, **diag_before}]
    first_collapse_step = None
    first_output_surrogate_death = None
    if diag_before["collapsed_single_class"]:
        first_collapse_step = 0
    if diag_before["layers"][2]["surrogate_quant_nz_frac"] == 0.0:
        first_output_surrogate_death = 0

    epoch_records = []
    step = 0
    for ep in range(1, epochs + 1):
        perm = np.random.RandomState(seed + ep - 1).permutation(n)
        ep_sel = ep_blk = ep_trans = 0
        ep_bsel = ep_bblk = ep_btrans = 0
        ep_loss = 0.0
        for i in perm:
            m = train_sample_online(
                net, cm, sub_x[i], int(sub_y[i]),
                scale=scale, radius=1.0, levels=levels,
                adaptive_radius=True, scale_factor=scale_factor)
            step += 1
            ep_sel += m["selected_weights"]
            ep_blk += m["blocked_weights"]
            ep_trans += m["transitions"]
            ep_bsel += m["bias_selected"]
            ep_bblk += m["bias_blocked"]
            ep_btrans += m["bias_transitions"]
            ep_loss += m["loss"]
            step_log.append({
                "step": step, "epoch": ep, "sample": int(i),
                "target": int(sub_y[i]), "loss": m["loss"],
                "firing_rates": m["firing_rates"],
                "selected": m["selected_weights"],
                "attempts": m["attempts"],
                "blocked": m["blocked_weights"],
                "transitions": m["transitions"],
                "bias_selected": m["bias_selected"],
                "bias_blocked": m["bias_blocked"],
                "bias_transitions": m["bias_transitions"],
                "quant": {str(l): {
                    "dw_nz_float": v["dw_nz_float"],
                    "dw_nz_int": v["dw_nz_int"],
                    "dw_mean_abs_float": v["dw_mean_abs_float"],
                    "dw_mean_abs_int": v["dw_mean_abs_int"],
                } for l, v in m["quant_stats"].items()},
            })
            if step % diag_every == 0:
                d = diagnose(net, sub_x, sub_y, scale, scale_factor, levels)
                d = {"step": step, "epoch": ep, **d}
                diag_trail.append(d)
                if first_collapse_step is None and d["collapsed_single_class"]:
                    first_collapse_step = step
                if (first_output_surrogate_death is None
                        and d["layers"][2]["surrogate_quant_nz_frac"] == 0.0):
                    first_output_surrogate_death = step
        acc_now = float(evaluate(net, sub_x, sub_y))
        d_end = diagnose(net, sub_x, sub_y, scale, scale_factor, levels)
        if first_collapse_step is None and d_end["collapsed_single_class"]:
            first_collapse_step = step
        if (first_output_surrogate_death is None
                and d_end["layers"][2]["surrogate_quant_nz_frac"] == 0.0):
            first_output_surrogate_death = step
        epoch_records.append({
            "epoch": ep, "loss": ep_loss / n, "accuracy": acc_now,
            "selected": ep_sel, "blocked": ep_blk,
            "transitions": ep_trans, "bias_selected": ep_bsel,
            "bias_blocked": ep_bblk, "bias_transitions": ep_btrans,
            "firing_rates": d_end["firing_rates"],
            "pred_dist": d_end["pred_dist"],
            "surrogate_quant_nz": [l["surrogate_quant_nz_frac"]
                                   for l in d_end["layers"]],
        })
        print(f"  Epoch {ep}/{epochs} | loss={ep_loss / n:.4f} | "
              f"acc={acc_now:.4f} | sel={ep_sel} blk={ep_blk} "
              f"trans={ep_trans} | preds={d_end['pred_dist']}")

    # 4. Acuracia DEPOIS + diagnostico DEPOIS.
    acc_after = float(evaluate(net, sub_x, sub_y))
    diag_after = diagnose(net, sub_x, sub_y, scale, scale_factor, levels)
    scores_after, _, _, _ = net.forward_with_scores(
        sub_x.astype(np.float32))
    np.save(out / "scores_after.npy", np.asarray(scores_after))

    dacc = acc_after - acc_before
    frozen_epochs = sum(1 for r in epoch_records if r["transitions"] == 0)
    passed = bool(dacc > 0 and acc_after > maj_acc and acc_after > acc_before)
    failure_point = None
    if not passed:
        cands = []
        if first_output_surrogate_death is not None:
            cands.append(
                f"morte do surrogate de saida no passo "
                f"{first_output_surrogate_death} "
                f"(q_fake(z_out)==0 em todo o subconjunto => deltas zero => "
                f"treino congela; causa: escala/raio — |z_out| medio "
                f"{diag_after['layers'][2]['z_mean_abs']:.1f} >> raio)")
        if first_collapse_step is not None:
            cands.append(
                f"colapso de diversidade no passo {first_collapse_step} "
                f"(todas as {n} preds -> classe "
                f"{int(np.argmax(diag_after['pred_dist']))})")
        if frozen_epochs:
            cands.append(f"{frozen_epochs} epoch(s) congelada(s) "
                         f"(0 transicoes efetivas)")
        if not cands:
            cands.append("sem melhora e sem ponto singular detectado; "
                         "ver trilha de diagnosticos")
        failure_point = "; ".join(cands)

    metrics = {
        "experiment": "exp-013-small-subset",
        "architecture": net.get_stats()["architecture"],
        "hyperparams": {"seed": seed, "n_subset": n, "epochs": epochs,
                        "budget_steps": epochs * n, "scale": scale,
                        "radius": 1.0, "adaptive_radius": True,
                        "scale_factor": scale_factor, "levels": levels},
        "baseline": {"initial_accuracy": acc_before,
                     "majority_class": maj_class,
                     "majority_accuracy": maj_acc},
        "final": {"accuracy": acc_after, "delta_accuracy": dacc,
                  "passed": passed},
        "totals": {
            "selected": sum(r["selected"] for r in epoch_records),
            "blocked": sum(r["blocked"] for r in epoch_records),
            "transitions": sum(r["transitions"] for r in epoch_records),
            "bias_transitions": sum(r["bias_transitions"]
                                    for r in epoch_records),
            "frozen_epochs": frozen_epochs},
        "failure_point": failure_point,
        "first_collapse_step": first_collapse_step,
        "first_output_surrogate_death_step": first_output_surrogate_death,
        "epoch_records": epoch_records,
        "diag_before": diag_before,
        "diag_after": diag_after,
        "diag_trail": diag_trail,
    }
    with open(out / "step_log.json", "w") as f:
        json.dump(step_log, f, indent=1)
    save_metrics(metrics, str(out / "metrics.json"),
                 experiment="exp-013-small-subset")

    # Pesos finais (pesos + bias + contadores, padrao 012).
    layer_w = {f"layer_{i}": net.layers[i].weights
               for i in range(len(net.layers))}
    layer_b = {f"bias_{i}": net.layers[i].bias
               for i in range(len(net.layers))}
    cc_w = {f"counter_{i}": cm.weight_counters[i]
            for i in range(len(net.layers))}
    cc_b = {f"counter_bias_{i}": cm.bias_counters[i]
            for i in range(len(net.layers))}
    np.savez(out / "weights.npz", **layer_w, **layer_b, **cc_w, **cc_b)

    # REPORT.md — veredito auditavel.
    status = "PASS" if passed else "FAIL"
    lines = [
        "# exp-013 — subconjunto pequeno (B1 corrigido)",
        "",
        f"**Veredito: {status}** — "
        f"acc {acc_before:.4f} -> {acc_after:.4f} "
        f"(Δ={dacc:+.4f}) vs majoritario {maj_acc:.4f} "
        f"(n={n}, seed={seed}, {epochs} epochs = {epochs * n} passos).",
        "",
        "## Diagnostico",
        "",
        f"- Preds antes: `{diag_before['pred_dist']}` "
        f"-> depois: `{diag_after['pred_dist']}`.",
        f"- Atividade antes: "
        f"{[round(v, 3) for v in diag_before['firing_rates']]} -> depois: "
        f"{[round(v, 3) for v in diag_after['firing_rates']]}.",
        f"- Surrogate quantizado nao-nulo (z1/z2/zout) antes: "
        f"{[round(l['surrogate_quant_nz_frac'], 3) for l in diag_before['layers']]} "
        f"-> depois: "
        f"{[round(l['surrogate_quant_nz_frac'], 3) for l in diag_after['layers']]}.",
        f"- |z| medio (z1/z2/zout) antes: "
        f"{[round(l['z_mean_abs'], 1) for l in diag_before['layers']]} "
        f"-> depois: "
        f"{[round(l['z_mean_abs'], 1) for l in diag_after['layers']]}.",
        f"- Transicoes efetivas por epoch: "
        f"{[r['transitions'] for r in epoch_records]} "
        f"(bloqueios: {[r['blocked'] for r in epoch_records]}).",
    ]
    if failure_point:
        lines += ["", "## Primeiro ponto de falha",
                  "", f"{failure_point}.",
                  "", "Proximo passo sugerido: investigar em 015 "
                  "(margem/homeostase/leak) como manter |z| dentro do cone "
                  "do surrogate e preservar diversidade de preds; "
                  "nao adicionar mecanismos nesta task."]
    else:
        lines += ["", "O mecanismo aprende o subconjunto acima do "
                  "majoritario e da rede inicial."]
    lines += ["", "Artefatos: `subset.json`, `metrics.json`, "
              "`step_log.json`, `scores_before/after.npy`, "
              "`weights.npz`, `diag_trail` dentro de `metrics.json`."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")

    print(f"\n[{status}] acc {acc_before:.4f} -> {acc_after:.4f} "
          f"(Δ={dacc:+.4f}) vs maj {maj_acc:.4f}")
    if failure_point:
        print(f"  Falha: {failure_point}")
    print(f"  Resultados: {out}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=64, choices=[32, 64, 128])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--scale-factor", type=float, default=1.0)
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--diag-every", type=int, default=8)
    p.add_argument("--outdir", type=str,
                   default="results/exp-013-small-subset")
    args = p.parse_args()
    run_experiment(seed=args.seed, n=args.n, epochs=args.epochs,
                   scale=args.scale, scale_factor=args.scale_factor,
                   levels=args.levels, diag_every=args.diag_every,
                   outdir=args.outdir)


if __name__ == "__main__":
    main()
