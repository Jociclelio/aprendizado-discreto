"""
[001] MLP baseline para MNIST

Treina um MLP float32 com arquitetura 784 → 128 → 64 → 10
para servir como referência fixa (vara de medir) para a Rede B.

Uso:
    python src/baseline.py
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3  # Adam
MOMENTUM = 0.9  # SGD fallback
OPTIMIZER = "adam"  # "adam" or "sgd"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "exp-001-baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class BaselineMLP(nn.Module):
    """MLP: 784 → 128 → ReLU → 64 → ReLU → 10 → softmax."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten (28x28 → 784)
        return self.net(x)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def get_dataloaders(batch_size: int = 128):
    """Retorna train_loader e test_loader para MNIST."""
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=BASE_DIR / "data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.MNIST(
        root=BASE_DIR / "data", train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Treino e avaliação
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_accuracy_curve(train_accs: list, test_accs: list):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_accs) + 1)

    ax.plot(epochs, train_accs, "o-", label="Treino", color="#2196F3")
    ax.plot(epochs, test_accs, "s-", label="Teste", color="#4CAF50")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("MLP Baseline — Curva de Acurácia")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = RESULTS_DIR / "accuracy_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Curva salva em {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Optimizer: {OPTIMIZER} (lr={LR})")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print()

    # Modelo
    model = BaselineMLP().to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros totais: {param_count:,}")
    print(model)
    print()

    # Dados
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    print(f" amostras de treino, {len(test_loader.dataset)} amostras de teste")
    print()

    # Perda e otimizador
    criterion = nn.CrossEntropyLoss()
    if OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LR)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # Treino
    train_accs, test_accs = [], []
    best_test_acc = 0.0
    checkpoint_path = RESULTS_DIR / "baseline_checkpoint.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_acc = evaluate(model, test_loader)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch:2d}/{EPOCHS}  "
            f"Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.2f}%  "
            f"Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                },
                checkpoint_path,
            )

    # Salvar curva
    plot_accuracy_curve(train_accs, test_accs)

    # Salvar CSV
    csv_path = RESULTS_DIR / "accuracy_curve.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_acc", "test_acc"])
        for i, (ta, te) in enumerate(zip(train_accs, test_accs), start=1):
            writer.writerow([i, round(ta, 4), round(te, 4)])
    print(f"  → CSV salvo em {csv_path}")

    # Resumo
    print()
    print("=" * 60)
    print(f"RESULTADO FINAL")
    print(f"  Melhor acurácia de teste: {best_test_acc:.2f}%")
    print(f"  Acurácia final de teste:  {test_accs[-1]:.2f}%")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Arquivos: {list(RESULTS_DIR.iterdir())}")
    print("=" * 60)

    return best_test_acc


if __name__ == "__main__":
    best = main()
    if best < 97.0:
        print(f"\n⚠️  Atenção: acurácia {best:.2f}% < 97% — critério não atingido.")
    else:
        print(f"\n✓ Critério atingido: acurácia {best:.2f}% ≥ 97%")
