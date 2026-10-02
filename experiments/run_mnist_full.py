"""
[014] Ensaio MNIST auditavel — proveniencia total e avaliacao limpa.

Executa o B1 corrigido [012] no MNIST completo com:
    - Split 3-vias: 50k treino / 10k validacao / 10k teste
    - Selecao de checkpoint por validacao; teste avaliado uma unica vez
    - Multiplas sementes (42, 123, 999), todas registradas
    - Proveniencia completa (commit, versoes, splits, config, tempo)
    - Unidades padronizadas: acuracia em fracao [0,1]
    - Curvas treino/validacao/teste (PNG + CSV)
    - Matriz de confusao final (teste, unica vez)

Uso:
    python experiments/run_mnist_full.py
    python experiments/run_mnist_full.py --config experiments/config_014.json
    python experiments/run_mnist_full.py --seeds 42 123 --epochs 5

Contrato [012] (reutilizado):
    z_l = W_l @ x_{l-1} + b_l, a_l = step(z_l), scores = z_out,
    C acumula GRADIENTE => W <- clip(W - sign(C)), b <- clip(b - sign(C_b)).
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.counter_rules import CounterManager
from src.fake_gradient import get_adaptive_radius
from src.ternary_b1 import evaluate, train_sample_online
from src.ternary_net import TernaryNet
from src.utils import ensure_dir, load_mnist, save_metrics, set_seed

ARCH_SHAPES = [(128, 784), (64, 128), (10, 64)]
FAN_INS = [784, 128, 64]
DEFAULT_CONFIG = {
    "scale": 1.0,
    "adaptive_radius": True,
    "scale_factor": 1.0,
    "levels": 5,
    "batch_size": 32,
    "epochs": 5,
    "decay_rate": None,
    "decay_shift": None,
    "decay_per_batch": True,
    "seeds": [42, 123, 999],
    "train_size": 10000,
    "val_size": 5000,
    "test_size": 10000,
}


# ---------------------------------------------------------------------------
# Proveniencia
# ---------------------------------------------------------------------------
def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def get_git_status() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def get_versions() -> dict:
    import numpy as np
    import torch
    import torchvision
    return {
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": sys.version.split()[0],
    }


def build_provenance(config: dict, seed: int, split_indices: dict) -> dict:
    return {
        "git_commit": get_git_commit(),
        "git_status": get_git_status(),
        "versions": get_versions(),
        "seed": seed,
        "config": config,
        "split_indices": split_indices,
        "architecture": "784->128->64->10",
        "units": "accuracy in fraction [0,1]",
        "preprocessing": {
            "baseline": "MNIST standardization (0.1307, 0.3081)",
            "rede_B": "pixels [0,1] (normalize=True)",
            "nota": "mesma topologia NAO isola o efeito da quantizacao "
                    "(tambem mudam ativacao/perda/entrada)",
        },
    }


# ---------------------------------------------------------------------------
# Split 3-vias
# ---------------------------------------------------------------------------
def make_split(train_data, train_labels, test_data, test_labels,
               train_size, val_size, test_size, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(train_data.shape[0])
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = np.arange(test_data.shape[0])
    return {
        "train_x": train_data[train_idx], "train_y": train_labels[train_idx],
        "val_x": train_data[val_idx], "val_y": train_labels[val_idx],
        "test_x": test_data[test_idx], "test_y": test_labels[test_idx],
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }


# ---------------------------------------------------------------------------
# Treino + avaliacao
# ---------------------------------------------------------------------------
def train_one_seed(config: dict, seed: int, split: dict, outdir: Path) -> dict:
    set_seed(seed)
    radii = [get_adaptive_radius(d, config["scale_factor"]) for d in FAN_INS]

    net = TernaryNet(seed=seed)
    cm = CounterManager(
        ARCH_SHAPES,
        decay_rate=config.get("decay_rate"),
        decay_shift=config.get("decay_shift"),
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
            if config.get("decay_per_batch") and (
                cm.decay_rate is not None or cm.decay_shift is not None
            ):
                cm.decay()
            for i in perm[start:end]:
                m = train_sample_online(
                    net, cm, train_x[i], int(train_y[i]),
                    scale=config["scale"], radius=1.0,
                    levels=config["levels"],
                    adaptive_radius=config["adaptive_radius"],
                    scale_factor=config["scale_factor"],
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
        print(f"  Epoch {ep}/{epochs} | loss={rec['loss']:.4f} | "
              f"train={train_acc:.4f} | val={val_acc:.4f} | "
              f"trans={ep_trans}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            best_weights = [l.weights.copy() for l in net.layers]
            best_bias = [l.bias.copy() for l in net.layers]

    train_time = time.time() - t0

    # Restaurar melhor checkpoint (selecao por validacao)
    for i, layer in enumerate(net.layers):
        layer.weights = best_weights[i].copy()
        layer.bias = best_bias[i].copy()

    # Avaliacao de teste — UNICA vez, no fim
    test_acc = float(evaluate(net, split["test_x"], split["test_y"]))

    # Matriz de confusao (teste, unica vez)
    cm_conf = np.zeros((10, 10), dtype=np.int64)
    for start in range(0, split["test_x"].shape[0], 256):
        end = min(start + 256, split["test_x"].shape[0])
        scores, _, _, _ = net.forward_with_scores(
            split["test_x"][start:end].astype(np.float32))
        preds = np.argmax(scores, axis=1)
        for t, p in zip(split["test_y"][start:end], preds):
            cm_conf[int(t), int(p)] += 1

    # Distribuicao de predicoes no teste
    pred_dist = np.bincount(np.argmax(
        net.forward_with_scores(split["test_x"].astype(np.float32))[0], axis=1
    ), minlength=10).tolist()

    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_time_s": round(train_time, 2),
        "history": history,
        "confusion_matrix": cm_conf.tolist(),
        "test_pred_dist": pred_dist,
        "final_weights": [l.weights.copy() for l in net.layers],
        "final_bias": [l.bias.copy() for l in net.layers],
    }

    # Salvar pesos do melhor checkpoint
    np.savez(
        str(outdir / f"weights_seed{seed}.npz"),
        **{f"layer_{i}": best_weights[i] for i in range(3)},
        **{f"bias_{i}": best_bias[i] for i in range(3)},
    )

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(all_results: list, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for res in all_results:
        epochs = [r["epoch"] for r in res["history"]]
        axes[0].plot(epochs, [r["train_acc"] for r in res["history"]],
                     "o-", label=f"seed {res['seed']}")
        axes[1].plot(epochs, [r["val_acc"] for r in res["history"]],
                     "s-", label=f"seed {res['seed']}")
        axes[2].plot(epochs, [r["transitions"] for r in res["history"]],
                     "^-", label=f"seed {res['seed']}")

    axes[0].set_title("Treino")
    axes[0].set_ylabel("Acuracia (fracao)")
    axes[1].set_title("Validacao")
    axes[2].set_title("Transicoes por epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ensaio MNIST Auditavel — B1 corrigido [014]")
    fig.tight_layout()
    fig.savefig(str(outdir / "curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_csv(all_results: list, outdir: Path):
    with open(outdir / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "epoch", "loss", "train_acc", "val_acc",
                     "transitions", "selected", "blocked"])
        for res in all_results:
            for r in res["history"]:
                w.writerow([res["seed"], r["epoch"], r["loss"],
                            r["train_acc"], r["val_acc"], r["transitions"],
                            r["selected"], r["blocked"]])


def plot_confusion(cm: np.ndarray, outdir: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(str(outdir / "confusion_matrix.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--outdir", type=str, default="results/exp-014-mnist-full")
    args = p.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    if args.seeds:
        config["seeds"] = args.seeds
    if args.epochs:
        config["epochs"] = args.epochs

    outdir = ensure_dir(Path(args.outdir))

    print("=" * 60)
    print("[014] Ensaio MNIST auditavel — B1 corrigido")
    print(f"  Config: {json.dumps(config, indent=2)}")
    print(f"  Outdir: {outdir}")
    print("=" * 60)

    print("\nCarregando MNIST...")
    train_data, train_labels, test_data, test_labels = load_mnist(normalize=True)
    print(f"  Train: {train_data.shape}, Test: {test_data.shape}")

    all_results = []
    for seed in config["seeds"]:
        print(f"\n--- Seed {seed} ---")
        split = make_split(
            train_data, train_labels, test_data, test_labels,
            config["train_size"], config["val_size"], config["test_size"],
            seed,
        )
        prov = build_provenance(config, seed, {
            "train_idx_hash": int(split["train_idx"][0]),
            "val_idx_hash": int(split["val_idx"][0]),
            "test_idx_hash": int(split["test_idx"][0]),
            "train_size": config["train_size"],
            "val_size": config["val_size"],
            "test_size": config["test_size"],
        })
        with open(outdir / f"provenance_seed{seed}.json", "w") as f:
            json.dump(prov, f, indent=2, default=str)

        res = train_one_seed(config, seed, split, outdir)
        all_results.append(res)
        print(f"  Seed {seed}: best_val={res['best_val_acc']:.4f} "
              f"(epoch {res['best_epoch']}) | test={res['test_acc']:.4f}")

    # Agregacao
    test_accs = [r["test_acc"] for r in all_results]
    val_accs = [r["best_val_acc"] for r in all_results]
    summary = {
        "experiment": "exp-014-mnist-full",
        "config": config,
        "seeds": config["seeds"],
        "results": [
            {"seed": r["seed"], "best_epoch": r["best_epoch"],
             "best_val_acc": r["best_val_acc"], "test_acc": r["test_acc"],
             "train_time_s": r["train_time_s"],
             "test_pred_dist": r["test_pred_dist"]}
            for r in all_results
        ],
        "aggregate": {
            "val_acc_mean": float(np.mean(val_accs)),
            "val_acc_std": float(np.std(val_accs)),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
        },
        "units": "accuracy in fraction [0,1]",
        "test_evaluated_once": True,
        "checkpoint_selection": "validation",
    }
    save_metrics(summary, str(outdir / "metrics.json"),
                 experiment="exp-014-mnist-full")

    # Curvas e CSV
    plot_curves(all_results, outdir)
    save_csv(all_results, outdir)

    # Matriz de confusao (media das sementes)
    avg_cm = np.mean([np.array(r["confusion_matrix"]) for r in all_results],
                     axis=0).astype(int)
    plot_confusion(avg_cm, outdir, "Matriz de Confusao — Teste (media seeds)")

    print("\n" + "=" * 60)
    print("RESUMO")
    print(f"  Validacao: {np.mean(val_accs):.4f} +/- {np.std(val_accs):.4f}")
    print(f"  Teste:     {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
    print(f"  Resultados: {outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
