"""
[002] Utilitários compartilhados para experimentos MNIST

Funções reutilizáveis pelas Redes A (baseline) e B (ternária):
- load_mnist()          : carrega e normaliza MNIST em numpy
- plot_accuracy_curve() : gera gráfico de acurácia treino/teste
- save_metrics()        : salva métricas como JSON
- count_ternary_weights(): conta pesos ternários {−1, 0, +1}

Uso:
    from src.utils import load_mnist, plot_accuracy_curve, save_metrics
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ---------------------------------------------------------------------------
# Loader MNIST
# ---------------------------------------------------------------------------
def load_mnist(
    data_dir: Optional[Path] = None,
    normalize: bool = True,
    return_tensors: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega MNIST e retorna arrays numpy.

    Args:
        data_dir: diretório onde os dados serão baixados/armazenados.
                  Padrão: ``<repo_root>/data``.
        normalize: se True, normaliza pixels para [0, 1].
        return_tensors: se True, retorna torch.Tensor em vez de np.ndarray.

    Returns:
        (train_data, train_labels, test_data, test_labels)
        - train_data:  (60000, 784)  float32
        - train_labels: (60000,)     int64
        - test_data:    (10000, 784) float32
        - test_labels:  (10000,)     int64
    """
    if data_dir is None:
        data_dir = DATA_DIR

    transform_train = transforms.Compose(
        [transforms.ToTensor()]
        + ([transforms.Normalize((0.1307,), (0.3081,))] if normalize else [])
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor()]
        + ([transforms.Normalize((0.1307,), (0.3081,))] if normalize else [])
    )

    train_dataset = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform_test
    )

    # Convert to numpy: (N, 784) e (N,)
    train_data = train_dataset.data.numpy().astype(np.float32)  # (60000, 28, 28)
    train_labels = train_dataset.targets.numpy().astype(np.int64)
    test_data = test_dataset.data.numpy().astype(np.float32)
    test_labels = test_dataset.targets.numpy().astype(np.int64)

    # Flatten para (N, 784)
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)

    if normalize:
        # Se usou Normalize do torchvision, os dados já estão normalizados
        # (média 0.1307, std 0.3081). Se quiser [0,1] puro, descomente:
        # train_data /= 255.0
        # test_data /= 255.0
        pass

    if return_tensors:
        train_data = torch.from_numpy(train_data)
        train_labels = torch.from_numpy(train_labels)
        test_data = torch.from_numpy(test_data)
        test_labels = torch.from_numpy(test_labels)

    return train_data, train_labels, test_data, test_labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_accuracy_curve(
    epochs: Union[List[int], np.ndarray],
    train_acc: Union[List[float], np.ndarray],
    test_acc: Union[List[float], np.ndarray],
    title: str = "Acurácia ao longo do Treino",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Gera gráfico de acurácia treino/teste e salva como PNG.

    Args:
        epochs: lista de números de epoch.
        train_acc: acurácia de treino por epoch (%).
        test_acc: acurácia de teste por epoch (%).
        title: título do gráfico.
        save_path: caminho do arquivo PNG. Se None, salva em
                   ``results/exp-NNN-nome/accuracy_curve.png``.
        show: se True, exibe o gráfico (interativo).

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = np.asarray(epochs)
    train_acc = np.asarray(train_acc)
    test_acc = np.asarray(test_acc)

    ax.plot(epochs, train_acc, "o-", label="Treino", color="#2196F3", linewidth=2)
    ax.plot(epochs, test_acc, "s-", label="Teste", color="#4CAF50", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Acurácia (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(epochs.min(), epochs.max())

    # Destacar melhor teste
    best_idx = np.argmax(test_acc)
    ax.annotate(
        f"{test_acc[best_idx]:.2f}%",
        xy=(epochs[best_idx], test_acc[best_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color="#4CAF50",
        fontweight="bold",
    )

    if save_path is None:
        save_path = BASE_DIR / "results" / "accuracy_curve_default.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return fig


def plot_comparison_curves(
    datasets: List[Dict],
    title: str = "Comparação de Curvas",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plota múltiplas curvas de acurácia lado a lado para comparação.

    Args:
        datasets: lista de dicts com chaves:
                  'epochs', 'train_acc', 'test_acc', 'label'
        title: título do gráfico.
        save_path: caminho do PNG.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

    for i, ds in enumerate(datasets):
        color = colors[i % len(colors)]
        ax.plot(
            ds["epochs"],
            ds["train_acc"],
            "--",
            label=f"{ds['label']} (train)",
            color=color,
            alpha=0.7,
        )
        ax.plot(
            ds["epochs"],
            ds["test_acc"],
            "-",
            label=f"{ds['label']} (test)",
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Acurácia (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path is None:
        save_path = BASE_DIR / "results" / "comparison_curve_default.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Salvamento de métricas
# ---------------------------------------------------------------------------
def save_metrics(
    metrics: Dict,
    save_path: Optional[Union[str, Path]] = None,
    experiment: str = "exp-unknown",
) -> Path:
    """
    Salva um dict de métricas como JSON com timestamp.

    Args:
        metrics: dict com chaves-string e valores serializáveis.
        save_path: caminho do arquivo JSON. Se None, usa
                   ``results/{experiment}/metrics.json``.
        experiment: nome do experimento (usado no path).

    Returns:
        Path para o arquivo salvo.
    """
    if save_path is None:
        save_path = BASE_DIR / "results" / experiment / "metrics.json"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Adicionar metadados automáticos
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        **metrics,
    }

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return save_path


def load_metrics(save_path: Union[str, Path]) -> Dict:
    """Carrega métricas de um arquivo JSON."""
    save_path = Path(save_path)
    with open(save_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers para Rede B (ternária)
# ---------------------------------------------------------------------------
def count_ternary_weights(model: nn.Module) -> Dict[str, int]:
    """
    Conta pesos ternários em um modelo PyTorch.

    Analiza todos os parâmetros Linear (ou convoluções) e verifica
    quantos pesos estão exatamente em {-1, 0, +1}.

    Args:
        model: modelo PyTorch com camadas nn.Linear.

    Returns:
        Dict com:
          - 'total': total de pesos
          - 'positive': peso == +1
          - 'zero': peso == 0
          - 'negative': peso == -1
          - 'ternary_fraction': fração em {-1, 0, +1}
    """
    total = 0
    positive = 0
    zero = 0
    negative = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            w = param.detach().cpu().numpy()
            total += w.size
            positive += int(np.sum(w == 1.0))
            zero += int(np.sum(w == 0.0))
            negative += int(np.sum(w == -1.0))
        elif "bias" in name:
            total += param.numel()  # biases não contam como ternários

    ternary_count = positive + zero + negative
    fraction = ternary_count / total if total > 0 else 0.0

    return {
        "total": total,
        "positive": positive,
        "zero": zero,
        "negative": negative,
        "ternary_fraction": round(fraction, 4),
    }


def count_nonzero_weights(model: nn.Module) -> Dict[str, int]:
    """
    Conta pesos não-zero em um modelo (útil para sparsity analysis).

    Returns:
        Dict com 'total', 'nonzero', 'zero', 'sparsity'.
    """
    total = 0
    nonzero = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            w = param.detach().cpu().numpy()
            total += w.size
            nonzero += int(np.sum(w != 0.0))

    zero = total - nonzero
    sparsity = zero / total if total > 0 else 0.0

    return {
        "total": total,
        "nonzero": nonzero,
        "zero": zero,
        "sparsity": round(sparsity, 4),
    }


# ---------------------------------------------------------------------------
# Utilitários gerais
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Define sementes de reprodutibilidade."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Garante que um diretório existe, cria se necessário."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_number(n: int) -> str:
    """Formata número com separadores de milhar."""
    return f"{n:,}"


# ---------------------------------------------------------------------------
# CLI de teste rápido
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Teste rápido de utils.py")
    print("=" * 60)

    # 1. load_mnist
    print("\n[1] Carregando MNIST...")
    train_data, train_labels, test_data, test_labels = load_mnist()
    print(f"   train_data:  shape={train_data.shape}, dtype={train_data.dtype}")
    print(f"   train_labels: shape={train_labels.shape}, dtype={train_labels.dtype}")
    print(f"   test_data:   shape={test_data.shape}, dtype={test_data.dtype}")
    print(f"   test_labels:  shape={test_labels.shape}, dtype={test_labels.dtype}")
    assert train_data.shape == (60000, 784), "train_data shape incorreto"
    assert test_data.shape == (10000, 784), "test_data shape incorreto"
    print("   ✓ load_mnist OK")

    # 2. plot_accuracy_curve
    print("\n[2] Gerando gráfico de acurácia...")
    fake_epochs = list(range(1, 21))
    fake_train = [90 + i * 0.5 + np.random.randn() * 0.3 for i in range(20)]
    fake_test = [95 + i * 0.2 + np.random.randn() * 0.3 for i in range(20)]
    plot_accuracy_curve(
        fake_epochs, fake_train, fake_test, "Teste — Curva de Acurácia"
    )
    print("   ✓ plot_accuracy_curve OK")

    # 3. save_metrics
    print("\n[3] Salvando métricas...")
    metrics_path = save_metrics(
        {"accuracy": 97.97, "epochs": 20, "optimizer": "adam"},
        experiment="test-002",
    )
    print(f"   → {metrics_path}")
    loaded = load_metrics(metrics_path)
    assert loaded["accuracy"] == 97.97, "métrica não carregada corretamente"
    print("   ✓ save_metrics / load_metrics OK")

    # 4. count_ternary_weights
    print("\n[4] Testando count_ternary_weights...")

    class DummyTernaryNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = DummyTernaryNet()
    counts = count_ternary_weights(model)
    print(f"   Total:   {counts['total']}")
    print(f"   Positive: {counts['positive']}")
    print(f"   Zero:     {counts['zero']}")
    print(f"   Negative: {counts['negative']}")
    print(f"   Ternary:  {counts['ternary_fraction']}")
    print("   ✓ count_ternary_weights OK")

    print("\n" + "=" * 60)
    print("Todos os testes passaram!")
    print("=" * 60)
