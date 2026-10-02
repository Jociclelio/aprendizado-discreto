"""
[015] Ablacoes — margem, homeostase e leak sobre o B1 corrigido.

Cada mecanismo e testado isoladamente e depois combinado, com orcamento
e conjunto de selecao fixos (validacao de 014).

Uso:
    python experiments/run_ablations.py
    python experiments/run_ablations.py --quick  # 1 seed, 2 epochs

Resultados em results/exp-015-ablacoes/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ablation import train_sample_ablation
from src.counter_rules import CounterManager
from src.ternary_b1 import evaluate
from src.ternary_net import TernaryNet
from src.utils import ensure_dir, load_mnist, save_metrics, set_seed

ARCH_SHAPES = [(128, 784), (64, 128), (10, 64)]

# Orcamento fixo para todas as ablacoes
DEFAULT_CONFIG = {
    "scale": 1.0,
    "adaptive_radius": True,
    "scale_factor": 1.0,
    "levels": 5,
    "batch_size": 32,
    "epochs": 3,
    "train_size": 10000,
    "val_size": 5000,
    "test_size": 10000,
    "seeds": [42],
}

# Grade de ablacoes
ABLATIONS = [
    {"name": "base", "margin": None, "homeostasis": None, "decay_rate": None, "decay_shift": None},
    {"name": "leak_g0.90", "margin": None, "homeostasis": None, "decay_rate": 0.90, "decay_shift": None},
    {"name": "leak_g0.95", "margin": None, "homeostasis": None, "decay_rate": 0.95, "decay_shift": None},
    {"name": "leak_g0.99", "margin": None, "homeostasis": None, "decay_rate": 0.99, "decay_shift": None},
    {"name": "leak_s3", "margin": None, "homeostasis": None, "decay_rate": None, "decay_shift": 3},
    {"name": "leak_s4", "margin": None, "homeostasis": None, "decay_rate": None, "decay_shift": 4},
    {"name": "leak_s5", "margin": None, "homeostasis": None, "decay_rate": None, "decay_shift": 5},
    {"name": "margin_1", "margin": 1.0, "homeostasis": None, "decay_rate": None, "decay_shift": None},
    {"name": "margin_2", "margin": 2.0, "homeostasis": None, "decay_rate": None, "decay_shift": None},
    {"name": "margin_5", "margin": 5.0, "homeostasis": None, "decay_rate": None, "decay_shift": None},
    {"name": "homeo_0.25", "margin": None, "homeostasis": 0.25, "decay_rate": None, "decay_shift": None},
    {"name": "margin_2+homeo", "margin": 2.0, "homeostasis": 0.25, "decay_rate": None, "decay_shift": None},
    {"name": "margin_2+homeo+leak_g0.95", "margin": 2.0, "homeostasis": 0.25, "decay_rate": 0.95, "decay_shift": None},
]


def run_ablation(config: dict, abl: dict, split: dict, seed: int) -> dict:
    """Executa uma ablacao com orcamento fixo."""
    set_seed(seed)
    net = TernaryNet(seed=seed)
    cm = CounterManager(
        ARCH_SHAPES,
        decay_rate=abl["decay_rate"],
        decay_shift=abl["decay_shift"],
    )

    train_x, train_y = split["train_x"], split["train_y"]
    val_x, val_y = split["val_x"], split["val_y"]
    n = train_x.shape[0]
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    best_weights = None
    best_bias = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = np.random.RandomState(seed + ep - 1).permutation(n)
        ep_loss = 0.0
        ep_trans = 0
        ep_sel = 0
        ep_blk = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            if cm.decay_rate is not None or cm.decay_shift is not None:
                cm.decay()
            for i in perm[start:end]:
                m = train_sample_ablation(
                    net, cm, train_x[i], int(train_y[i]),
                    scale=config["scale"], radius=1.0,
                    levels=config["levels"],
                    adaptive_radius=config["adaptive_radius"],
                    scale_factor=config["scale_factor"],
                    margin=abl["margin"],
                    homeostasis_target=abl["homeostasis"],
                )
                ep_loss += m["loss"]
                ep_trans += m["transitions"]
                ep_sel += m["selected_weights"]
                ep_blk += m["blocked_weights"]
            n_batches += 1

        train_acc = float(evaluate(net, train_x[:5000], train_y[:5000]))
        val_acc = float(evaluate(net, val_x, val_y))

        rec = {
            "epoch": ep,
            "loss": ep_loss / (n_batches * batch_size),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "transitions": ep_trans,
            "selected": ep_sel,
            "blocked": ep_blk,
        }
        history.append(rec)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            best_weights = [l.weights.copy() for l in net.layers]
            best_bias = [l.bias.copy() for l in net.layers]

    train_time = time.time() - t0

    # Restaurar melhor checkpoint
    for i, layer in enumerate(net.layers):
        layer.weights = best_weights[i].copy()
        layer.bias = best_bias[i].copy()

    # Avaliacao de teste — unica vez
    test_acc = float(evaluate(net, split["test_x"], split["test_y"]))

    # Metricas de oscilacao: taxa de transicao por epoch
    trans_per_epoch = [r["transitions"] for r in history]

    return {
        "name": abl["name"],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_time_s": round(train_time, 2),
        "history": history,
        "transitions_per_epoch": trans_per_epoch,
        "final_weights": [l.weights.copy() for l in net.layers],
        "final_bias": [l.bias.copy() for l in net.layers],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true",
                   help="1 seed, 2 epochs (teste rapido)")
    p.add_argument("--outdir", type=str, default="results/exp-015-ablacoes")
    args = p.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.quick:
        config["epochs"] = 2
        config["seeds"] = [42]

    outdir = ensure_dir(Path(args.outdir))

    print("=" * 60)
    print("[015] Ablacoes — margem, homeostase e leak")
    print(f"  Config: {json.dumps(config, indent=2)}")
    print(f"  Ablacoes: {len(ABLATIONS)}")
    print(f"  Outdir: {outdir}")
    print("=" * 60)

    print("\nCarregando MNIST...")
    train_data, train_labels, test_data, test_labels = load_mnist(normalize=True)

    all_results = []
    for abl in ABLATIONS:
        for seed in config["seeds"]:
            print(f"\n--- {abl['name']} (seed={seed}) ---")
            rng = np.random.RandomState(seed)
            perm = rng.permutation(train_data.shape[0])
            train_idx = perm[:config["train_size"]]
            val_idx = perm[config["train_size"]:config["train_size"] + config["val_size"]]
            split = {
                "train_x": train_data[train_idx], "train_y": train_labels[train_idx],
                "val_x": train_data[val_idx], "val_y": train_labels[val_idx],
                "test_x": test_data, "test_y": test_labels,
            }

            res = run_ablation(config, abl, split, seed)
            all_results.append(res)
            print(f"  best_val={res['best_val_acc']:.4f} | "
                  f"test={res['test_acc']:.4f} | "
                  f"trans={res['transitions_per_epoch']}")

    # Consolidar resultados
    summary = {
        "experiment": "exp-015-ablacoes",
        "config": config,
        "ablations": [
            {
                "name": r["name"],
                "seed": r["seed"],
                "best_val_acc": r["best_val_acc"],
                "test_acc": r["test_acc"],
                "transitions_per_epoch": r["transitions_per_epoch"],
                "train_time_s": r["train_time_s"],
            }
            for r in all_results
        ],
        "best_config": None,
    }

    # Encontrar melhor configuracao por validacao
    best = max(all_results, key=lambda r: r["best_val_acc"])
    summary["best_config"] = {
        "name": best["name"],
        "best_val_acc": best["best_val_acc"],
        "test_acc": best["test_acc"],
    }

    save_metrics(summary, str(outdir / "metrics.json"),
                 experiment="exp-015-ablacoes")

    # Salvar pesos da melhor configuracao
    np.savez(
        str(outdir / "best_weights.npz"),
        **{f"layer_{i}": best["final_weights"][i] for i in range(3)},
        **{f"bias_{i}": best["final_bias"][i] for i in range(3)},
    )

    print("\n" + "=" * 60)
    print("RESUMO")
    for r in all_results:
        print(f"  {r['name']:30s} val={r['best_val_acc']:.4f} test={r['test_acc']:.4f}")
    print(f"\n  Melhor: {best['name']} (val={best['best_val_acc']:.4f})")
    print(f"  Resultados: {outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
