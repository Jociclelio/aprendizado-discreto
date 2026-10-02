"""
[016] Benchmark final — Baseline A vs. B1 corrigido vs. B+ refinada.

Avalia se ha ganho (sem pressupor) e valida a proposta de eficiencia:
memoria, bits por parametro, frequencia de transicao, tempo.

Uso:
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --quick  # 1 seed, 2 epochs

Resultados em results/exp-016-benchmark/.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from src.counter_rules import CounterManager
from src.ternary_b1 import evaluate, train_sample_online
from src.ternary_net import TernaryNet
from src.ternary_refined import TernaryRefined, train_refined
from src.utils import ensure_dir, load_mnist, save_metrics, set_seed

ARCH_SHAPES = [(128, 784), (64, 128), (10, 64)]
TOTAL_WEIGHTS = 109184
TOTAL_BIAS = 202
TOTAL_PARAMS = TOTAL_WEIGHTS + TOTAL_BIAS

# Memoria (bytes) — secao 7 do relatorio
MEM_BASELINE = TOTAL_PARAMS * 4  # float32
MEM_B_WEIGHTS = TOTAL_WEIGHTS * 1 + TOTAL_BIAS * 4  # int8 + int32
MEM_B_COUNTERS = TOTAL_WEIGHTS * 8 + TOTAL_BIAS * 8  # int64
MEM_B_TRAINING = MEM_B_WEIGHTS + MEM_B_COUNTERS


# ---------------------------------------------------------------------------
# Rede A — Baseline float32 (001)
# ---------------------------------------------------------------------------
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def train_baseline(seed: int, epochs: int, train_size: int, val_size: int,
                   batch_size: int = 128) -> dict:
    """Treina a baseline float32 com split treino/validacao."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    # Split treino/validacao
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_dataset))
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]

    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = BaselineMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = []
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_acc = correct / total

        train_loss = running_loss / len(train_subset)
        history.append({"epoch": ep, "loss": train_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0

    # Restaurar melhor checkpoint e avaliar teste
    model.load_state_dict(best_state)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_acc = correct / total

    return {
        "name": "A (baseline)",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_time_s": round(train_time, 2),
        "history": history,
        "bits_per_param": 32,
        "memory_params_bytes": MEM_BASELINE,
        "memory_training_bytes": MEM_BASELINE * 3,  # params + grads + moments (Adam)
    }


# ---------------------------------------------------------------------------
# Rede B — B1 corrigido (012)
# ---------------------------------------------------------------------------
def train_b1(seed: int, epochs: int, train_size: int, val_size: int,
             batch_size: int = 32) -> dict:
    """Treina B1 corrigido com split treino/validacao."""
    set_seed(seed)
    train_data, train_labels, test_data, test_labels = load_mnist(normalize=True)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(train_data.shape[0])
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]

    train_x, train_y = train_data[train_idx], train_labels[train_idx]
    val_x, val_y = train_data[val_idx], train_labels[val_idx]

    net = TernaryNet(seed=seed)
    cm = CounterManager(ARCH_SHAPES)

    n = train_x.shape[0]
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    best_weights = None
    best_bias = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm_ep = np.random.RandomState(seed + ep - 1).permutation(n)
        ep_loss = 0.0
        ep_trans = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            for i in perm_ep[start:end]:
                m = train_sample_online(
                    net, cm, train_x[i], int(train_y[i]),
                    scale=1.0, radius=1.0, levels=5,
                    adaptive_radius=True, scale_factor=1.0,
                )
                ep_loss += m["loss"]
                ep_trans += m["transitions"]
            n_batches += 1

        train_acc = float(evaluate(net, train_x[:5000], train_y[:5000]))
        val_acc = float(evaluate(net, val_x, val_y))

        history.append({
            "epoch": ep, "loss": ep_loss / (n_batches * batch_size),
            "train_acc": train_acc, "val_acc": val_acc,
            "transitions": ep_trans,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            best_weights = [l.weights.copy() for l in net.layers]
            best_bias = [l.bias.copy() for l in net.layers]

    train_time = time.time() - t0

    for i, layer in enumerate(net.layers):
        layer.weights = best_weights[i].copy()
        layer.bias = best_bias[i].copy()

    test_acc = float(evaluate(net, test_data, test_labels))

    return {
        "name": "B1 (corrigido)",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_time_s": round(train_time, 2),
        "history": history,
        "bits_per_param": 1.58,
        "memory_params_bytes": MEM_B_WEIGHTS,
        "memory_training_bytes": MEM_B_TRAINING,
    }


# ---------------------------------------------------------------------------
# Rede B+ — Refinada (015)
# ---------------------------------------------------------------------------
def train_bplus(seed: int, epochs: int, train_size: int, val_size: int,
                batch_size: int = 32) -> dict:
    """Treina B+ refinada com split treino/validacao."""
    set_seed(seed)
    train_data, train_labels, test_data, test_labels = load_mnist(normalize=True)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(train_data.shape[0])
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]

    train_x, train_y = train_data[train_idx], train_labels[train_idx]
    val_x, val_y = train_data[val_idx], train_labels[val_idx]

    model = TernaryRefined(seed=seed)
    n = train_x.shape[0]
    history = []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm_ep = np.random.RandomState(seed + ep - 1).permutation(n)
        ep_loss = 0.0
        ep_trans = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            for i in perm_ep[start:end]:
                m = model.train_sample(train_x[i], int(train_y[i]))
                ep_loss += m["loss"]
                ep_trans += m["transitions"]
            n_batches += 1

        train_acc = model.evaluate(train_x[:5000], train_y[:5000])
        val_acc = model.evaluate(val_x, val_y)

        history.append({
            "epoch": ep, "loss": ep_loss / (n_batches * batch_size),
            "train_acc": train_acc, "val_acc": val_acc,
            "transitions": ep_trans,
        })

    train_time = time.time() - t0

    test_acc = model.evaluate(test_data, test_labels)
    best_val_acc = max(h["val_acc"] for h in history)
    best_epoch = int(np.argmax([h["val_acc"] for h in history])) + 1

    return {
        "name": "B+ (refinada)",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_time_s": round(train_time, 2),
        "history": history,
        "bits_per_param": 1.58,
        "memory_params_bytes": MEM_B_WEIGHTS,
        "memory_training_bytes": MEM_B_TRAINING,
    }


# ---------------------------------------------------------------------------
# Largura de contador
# ---------------------------------------------------------------------------
def test_counter_width(seed: int, epochs: int, train_size: int,
                       val_size: int) -> dict:
    """Ensaiar contadores de 8/16 bits com politica de saturacao."""
    set_seed(seed)
    train_data, train_labels, test_data, test_labels = load_mnist(normalize=True)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(train_data.shape[0])
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]

    train_x, train_y = train_data[train_idx], train_labels[train_idx]
    val_x, val_y = train_data[val_idx], train_labels[val_idx]

    results = {}
    for bits in [8, 16]:
        max_val = 2 ** (bits - 1) - 1
        net = TernaryNet(seed=seed)
        cm = CounterManager(ARCH_SHAPES)

        n = train_x.shape[0]
        for ep in range(1, epochs + 1):
            perm_ep = np.random.RandomState(seed + ep - 1).permutation(n)
            for start in range(0, n, 32):
                end = min(start + 32, n)
                for i in perm_ep[start:end]:
                    m = train_sample_online(
                        net, cm, train_x[i], int(train_y[i]),
                        scale=1.0, radius=1.0, levels=5,
                        adaptive_radius=True, scale_factor=1.0,
                    )
                    # Saturacao: clipa contadores no range do tipo
                    for l in range(3):
                        cm.weight_counters[l] = np.clip(
                            cm.weight_counters[l], -max_val, max_val
                        )
                        cm.bias_counters[l] = np.clip(
                            cm.bias_counters[l], -max_val, max_val
                        )

        val_acc = float(evaluate(net, val_x, val_y))
        test_acc = float(evaluate(net, test_data, test_labels))
        results[f"{bits}bits"] = {
            "val_acc": val_acc,
            "test_acc": test_acc,
            "max_counter": max_val,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_benchmark(all_results: list, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for res in all_results:
        epochs = [h["epoch"] for h in res["history"]]
        axes[0, 0].plot(epochs, [h["val_acc"] for h in res["history"]],
                        "o-", label=res["name"])
        axes[0, 1].plot(epochs, [h["loss"] for h in res["history"]],
                        "o-", label=res["name"])
        if "transitions" in res["history"][0]:
            axes[1, 0].plot(epochs, [h["transitions"] for h in res["history"]],
                            "o-", label=res["name"])

    axes[0, 0].set_title("Validacao")
    axes[0, 0].set_ylabel("Acuracia (fracao)")
    axes[0, 1].set_title("Perda")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_title("Transicoes por epoch")
    axes[1, 0].set_ylabel("Transicoes")
    axes[1, 1].axis("off")

    # Tabela de memoria
    names = [r["name"] for r in all_results]
    mem_params = [r["memory_params_bytes"] / 1024 for r in all_results]
    mem_train = [r["memory_training_bytes"] / 1024 for r in all_results]
    x = np.arange(len(names))
    width = 0.35
    axes[1, 1].bar(x - width/2, mem_params, width, label="Parametros (KB)")
    axes[1, 1].bar(x + width/2, mem_train, width, label="Treinamento (KB)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=15, ha="right")
    axes[1, 1].set_title("Memoria")
    axes[1, 1].legend()

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Benchmark — A vs B1 vs B+ [016]")
    fig.tight_layout()
    fig.savefig(str(outdir / "benchmark.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--outdir", type=str, default="results/exp-016-benchmark")
    args = p.parse_args()

    config = {
        "epochs": 3,
        "train_size": 10000,
        "val_size": 5000,
        "seeds": [42, 123, 999],
    }
    if args.quick:
        config["epochs"] = 2
        config["seeds"] = [42]

    outdir = ensure_dir(Path(args.outdir))

    print("=" * 60)
    print("[016] Benchmark — A vs B1 vs B+")
    print(f"  Config: {json.dumps(config, indent=2)}")
    print(f"  Outdir: {outdir}")
    print("=" * 60)

    all_results = []
    for seed in config["seeds"]:
        print(f"\n--- Seed {seed} ---")

        print("  Treinando A (baseline)...")
        res_a = train_baseline(seed, config["epochs"], config["train_size"],
                               config["val_size"])
        all_results.append(res_a)
        print(f"    A: val={res_a['best_val_acc']:.4f} test={res_a['test_acc']:.4f}")

        print("  Treinando B1 (corrigido)...")
        res_b1 = train_b1(seed, config["epochs"], config["train_size"],
                          config["val_size"])
        all_results.append(res_b1)
        print(f"    B1: val={res_b1['best_val_acc']:.4f} test={res_b1['test_acc']:.4f}")

        print("  Treinando B+ (refinada)...")
        res_bp = train_bplus(seed, config["epochs"], config["train_size"],
                             config["val_size"])
        all_results.append(res_bp)
        print(f"    B+: val={res_bp['best_val_acc']:.4f} test={res_bp['test_acc']:.4f}")

    # Largura de contador
    print("\n--- Largura de contador ---")
    counter_results = test_counter_width(
        config["seeds"][0], config["epochs"], config["train_size"],
        config["val_size"]
    )
    for bits, res in counter_results.items():
        print(f"  {bits}: val={res['val_acc']:.4f} test={res['test_acc']:.4f}")

    # Consolidar
    summary = {
        "experiment": "exp-016-benchmark",
        "config": config,
        "results": [
            {
                "name": r["name"],
                "seed": r["seed"],
                "best_val_acc": r["best_val_acc"],
                "test_acc": r["test_acc"],
                "train_time_s": r["train_time_s"],
                "bits_per_param": r["bits_per_param"],
                "memory_params_bytes": r["memory_params_bytes"],
                "memory_training_bytes": r["memory_training_bytes"],
            }
            for r in all_results
        ],
        "counter_width": counter_results,
        "memory_accounting": {
            "baseline_params_bytes": MEM_BASELINE,
            "B_params_bytes": MEM_B_WEIGHTS,
            "B_training_bytes": MEM_B_TRAINING,
            "params_reduction": MEM_BASELINE / MEM_B_WEIGHTS,
            "training_overhead": MEM_B_TRAINING / MEM_BASELINE,
        },
    }
    save_metrics(summary, str(outdir / "metrics.json"),
                 experiment="exp-016-benchmark")

    # CSV
    with open(outdir / "benchmark.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "seed", "best_val_acc", "test_acc",
                     "train_time_s", "bits_per_param",
                     "memory_params_bytes", "memory_training_bytes"])
        for r in all_results:
            w.writerow([r["name"], r["seed"], r["best_val_acc"],
                        r["test_acc"], r["train_time_s"],
                        r["bits_per_param"], r["memory_params_bytes"],
                        r["memory_training_bytes"]])

    # Plot
    plot_benchmark(all_results, outdir)

    # README
    lines = [
        "# exp-016 — Benchmark A vs B1 vs B+",
        "",
        "## Resultados",
        "",
        "| Rede | Seed | Val Acc | Test Acc | Tempo (s) | Bits/Param | Mem Params (KB) | Mem Treino (KB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['name']} | {r['seed']} | {r['best_val_acc']:.4f} | "
            f"{r['test_acc']:.4f} | {r['train_time_s']:.1f} | "
            f"{r['bits_per_param']:.2f} | "
            f"{r['memory_params_bytes']/1024:.1f} | "
            f"{r['memory_training_bytes']/1024:.1f} |"
        )

    lines += [
        "",
        "## Largura de contador",
        "",
        "| Bits | Val Acc | Test Acc |",
        "|---|---|---|",
    ]
    for bits, res in counter_results.items():
        lines.append(f"| {bits} | {res['val_acc']:.4f} | {res['test_acc']:.4f} |")

    lines += [
        "",
        "## Conclusoes",
        "",
        f"- Reducao de memoria de parametros: {MEM_BASELINE/MEM_B_WEIGHTS:.2f}x",
        f"- Overhead de memoria de treinamento: {MEM_B_TRAINING/MEM_BASELINE:.2f}x",
        "- 'Zero MACs float' e propriedade de uma implementacao especializada, nao do codigo atual.",
        "- A comparacao de memoria de treinamento e incompleta sem considerar momentos do Adam e temporarios.",
        "- Se a Rede B nao alcancar a baseline, a contribucao e o trade-off (bits x transicoes x memoria), nao a acuracia.",
        "",
        "## Ressalvas",
        "",
        "- O benchmark NAO deve concluir 'a Rede B e superior' sem evidencia.",
        "- Tempo de treino com loops Python por amostra nao e representativo de implementacao otimizada.",
        "- Hardware/empacotamento: avaliar somente depois de existir um modelo treinado confiavel.",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n")

    print("\n" + "=" * 60)
    print("RESUMO")
    for r in all_results:
        print(f"  {r['name']:20s} val={r['best_val_acc']:.4f} test={r['test_acc']:.4f}")
    print(f"  Resultados: {outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
