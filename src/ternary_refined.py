"""
[015] Rede B Refinada (B+) — melhor configuracao das ablacoes.

Consolida a melhor configuracao encontrada em results/exp-015-ablacoes/:
    - Margem multiclasse (margin=2.0) sobre z_out
    - Homeostase de disparos (target_rate=0.25)
    - Sem leak (nao demonstrou ganho isolado)

A margem melhora a validacao (0.1022 vs 0.0932 base) e mantem transicoes
efetivas por mais epochs. A homeostase e incluida como regulacao
complementar, embora seu efeito isolado seja neutro/negativo.

Contrato [012] mantido: C acumula GRADIENTE => W <- clip(W - sign(C)).

Uso:
    from src.ternary_refined import TernaryRefined, train_refined
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ablation import train_sample_ablation
from src.counter_rules import CounterManager
from src.ternary_b1 import evaluate
from src.ternary_net import TernaryNet
from src.utils import set_seed

# Melhor configuracao das ablações (ver results/exp-015-ablacoes/)
REFINED_CONFIG = {
    "scale": 1.0,
    "adaptive_radius": True,
    "scale_factor": 1.0,
    "levels": 5,
    "margin": 2.0,
    "homeostasis_target": 0.25,
    "homeostasis_tolerance": 0.05,
    "homeostasis_step": 1,
    "decay_rate": None,
    "decay_shift": None,
}


class TernaryRefined:
    """
    Rede B Refinada (B+) — B1 corrigido + margem + homeostase.

    Wrapper sobre TernaryNet com configuracao de ablacoes consolidada.
    """

    def __init__(self, seed: int = 42, config: Optional[Dict] = None):
        self.config = config or REFINED_CONFIG
        self.net = TernaryNet(seed=seed)
        self.cm = CounterManager(
            [(128, 784), (64, 128), (10, 64)],
            decay_rate=self.config.get("decay_rate"),
            decay_shift=self.config.get("decay_shift"),
        )

    def train_sample(self, x: np.ndarray, target: int) -> Dict:
        """Treina uma amostra com a configuracao refinada."""
        return train_sample_ablation(
            self.net, self.cm, x, target,
            scale=self.config["scale"],
            radius=1.0,
            levels=self.config["levels"],
            adaptive_radius=self.config["adaptive_radius"],
            scale_factor=self.config["scale_factor"],
            margin=self.config.get("margin"),
            homeostasis_target=self.config.get("homeostasis_target"),
            homeostasis_tolerance=self.config.get("homeostasis_tolerance", 0.05),
            homeostasis_step=self.config.get("homeostasis_step", 1),
        )

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Avalia acuracia (fracao [0,1])."""
        return float(evaluate(self.net, data, labels))

    def forward_with_scores(self, x: np.ndarray):
        """Forward unificado (scores + binarias)."""
        return self.net.forward_with_scores(x)

    def get_weights(self):
        return self.net.get_weights()

    def get_biases(self):
        return self.net.get_biases()


def train_refined(
    epochs: int = 3,
    batch_size: int = 32,
    seed: int = 42,
    train_data: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    val_data: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
) -> tuple:
    """
    Treina a Rede B Refinada com orcamento fixo.

    Args:
        epochs: numero de epochs.
        batch_size: tamanho do batch.
        seed: semente.
        train_data, train_labels: dados de treino (N, 784) e (N,).
        val_data, val_labels: dados de validacao (opcional).
        config: configuracao customizada (ou REFINED_CONFIG).

    Returns:
        (model, history) — TernaryRefined treinado e historico.
    """
    set_seed(seed)
    model = TernaryRefined(seed=seed, config=config)

    if train_data is None:
        from src.utils import load_mnist
        train_data, train_labels, _, _ = load_mnist(normalize=True)
        train_data = train_data[:10000]
        train_labels = train_labels[:10000]

    n = train_data.shape[0]
    history = []

    for ep in range(1, epochs + 1):
        perm = np.random.RandomState(seed + ep - 1).permutation(n)
        ep_loss = 0.0
        ep_trans = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            for i in perm[start:end]:
                m = model.train_sample(train_data[i], int(train_labels[i]))
                ep_loss += m["loss"]
                ep_trans += m["transitions"]
            n_batches += 1

        train_acc = model.evaluate(train_data[:5000], train_labels[:5000])
        val_acc = model.evaluate(val_data, val_labels) if val_data is not None else train_acc

        rec = {
            "epoch": ep,
            "loss": ep_loss / (n_batches * batch_size),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "transitions": ep_trans,
        }
        history.append(rec)
        print(f"  Epoch {ep}/{epochs} | loss={rec['loss']:.4f} | "
              f"train={train_acc:.4f} | val={val_acc:.4f} | trans={ep_trans}")

    return model, history


# ---------------------------------------------------------------------------
# Testes unitarios
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testes — ternary_refined.py")
    print("=" * 60)

    # Teste 1: TernaryRefined
    print("\n[1] TernaryRefined")
    model = TernaryRefined(seed=42)
    x = np.random.randn(784).astype(np.float32)
    m = model.train_sample(x, 5)
    assert "loss" in m
    assert "transitions" in m
    print(f"   ✓ loss={m['loss']:.4f}, transitions={m['transitions']}")

    # Teste 2: train_refined
    print("\n[2] train_refined (mini)")
    from src.utils import load_mnist
    train_data, train_labels, _, _ = load_mnist(normalize=True)
    train_data = train_data[:100]
    train_labels = train_labels[:100]
    model2, hist = train_refined(epochs=1, batch_size=32, seed=42,
                                 train_data=train_data, train_labels=train_labels)
    assert len(hist) == 1
    print(f"   ✓ 1 epoch, loss={hist[0]['loss']:.4f}")

    print("\n" + "=" * 60)
    print("Todos os testes passaram!")
    print("=" * 60)
