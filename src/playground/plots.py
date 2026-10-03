"""
[017] Figuras do playground — so matplotlib (Agg, padrao do repo).

Heatmaps sempre AMOSTRADOS (L0 tem ~100k conexoes: nunca matriz cheia;
crop 32x32 ou top-|C| deterministico). Todas as funcoes retornam `fig`
(o chamador faz `st.pyplot(fig)` e `plt.close(fig)`).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

Tuple2D = Tuple[np.ndarray, str]

LAYER_NAMES = ["L0 784->128", "L1 128->64", "L2 64->10"]


def _sample_matrix(mat: np.ndarray, size: int = 32) -> Tuple2D:
    """Amostra deterministica size x size: linhas/colunas de maior media |.|.

    Retorna (amostra, titulo_sufixo). Deterministico: desempate por indice.
    """
    r, c = mat.shape
    if r <= size and c <= size:
        return mat, "cheia"
    row_score = np.mean(np.abs(mat.astype(np.float64)), axis=1)
    col_score = np.mean(np.abs(mat.astype(np.float64)), axis=0)
    rows = np.argsort(-row_score, kind="stable")[: min(size, r)]
    cols = np.argsort(-col_score, kind="stable")[: min(size, c)]
    rows = np.sort(rows)
    cols = np.sort(cols)
    return mat[np.ix_(rows, cols)], "top-|.| 32x32"


def fig_digit(img28: np.ndarray, label: int, pred: int) -> plt.Figure:
    """Imagem 28x28 do digito + label/pred no titulo."""
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    ax.imshow(img28, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(f"label={label} pred={pred}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_z_hist(z_list, radii, bins: int = 40) -> plt.Figure:
    """Histograma de z por camada + cone do raio adaptativo (±raio)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i, ax in enumerate(axes):
        z = np.asarray(z_list[i]).ravel()
        ax.hist(z, bins=bins, color="#2196F3", alpha=0.8)
        ax.axvline(-radii[i], color="red", linestyle="--", label=f"-raio ({radii[i]:.1f})")
        ax.axvline(radii[i], color="red", linestyle="--", label=f"+raio ({radii[i]:.1f})")
        frac_in = float(np.mean(np.abs(z) < radii[i]))
        ax.set_title(f"{LAYER_NAMES[i]}\n|z|<raio: {frac_in:.1%}")
        ax.set_xlabel("z")
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Pre-ativacoes z com cone do raio adaptativo")
    fig.tight_layout()
    return fig


def fig_surrogate(z, fake_cont, radius: float, scale: float, layer_name: str) -> plt.Figure:
    """Perfil do triangulo + pontos (z, fake) amostrados (ate 500)."""
    z = np.asarray(z).ravel().astype(np.float64)
    f = np.asarray(fake_cont).ravel().astype(np.float64)
    n = min(500, z.size)
    sel = np.linspace(0, z.size - 1, n).astype(int)
    xs = np.linspace(-1.2 * max(radius, 1e-9), 1.2 * max(radius, 1e-9), 200)
    tri = scale * np.maximum(0.0, 1.0 - np.abs(xs) / radius)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xs, tri, color="red", label="triangulo")
    ax.scatter(z[sel], f[sel], s=6, alpha=0.5, color="#2196F3", label="amostras")
    ax.set_title(f"Surrogate {layer_name} (raio={radius:.1f})")
    ax.set_xlabel("z")
    ax.set_ylabel("fake(z)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_heatmap_sampled(mat: np.ndarray, title: str, vmin=None, vmax=None,
                        cmap: str = "RdBu") -> plt.Figure:
    """Heatmap amostrado (nunca matriz cheia) + sufixo do metodo."""
    sample, how = _sample_matrix(np.asarray(mat), size=32)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(sample, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    ax.set_title(f"{title}\n[{how}] shape real {np.asarray(mat).shape}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_weight_dist(layer_stats) -> plt.Figure:
    """Distribuicao +1/0/-1 por camada via `net.get_stats()`."""
    labels = LAYER_NAMES
    pos = [s["positive"] for s in layer_stats]
    zer = [s["zero"] for s in layer_stats]
    neg = [s["negative"] for s in layer_stats]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(x, pos, label="+1", color="#4CAF50")
    ax.bar(x, zer, bottom=pos, label="0", color="#9E9E9E")
    ax.bar(x, neg, bottom=np.array(pos) + np.array(zer), label="-1", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("pesos")
    ax.set_title("Distribuicao de pesos +1/0/-1 por camada")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_counter_hist(Cw_list) -> plt.Figure:
    """Histograma de |C| (contadores de pesos) por camada + limiar B1."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i, ax in enumerate(axes):
        c = np.abs(np.asarray(Cw_list[i]).ravel())
        ax.hist(c, bins=30, color="#FF9800", alpha=0.85)
        thr = max(1.0, float(np.mean(c)))
        ax.axvline(thr, color="red", linestyle="--", label=f"thr B1={thr:.2f}")
        ax.set_title(f"{LAYER_NAMES[i]}  max|C|={c.max():.0f}")
        ax.set_xlabel("|C|")
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Contadores de pesos |C| por camada")
    fig.tight_layout()
    return fig


def fig_runner_curve(batches, loss, acc=None, trans=None) -> plt.Figure:
    """Curva do runner curto: loss (+acc/trans no eixo duplo opcional)."""
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.plot(batches, loss, "o-", color="#2196F3", label="loss/batch")
    ax1.set_xlabel("batch")
    ax1.set_ylabel("loss", color="#2196F3")
    lines, labels = ax1.get_legend_handles_labels()
    if acc is not None:
        ax2 = ax1.twinx()
        ax2.plot(batches, acc, "s-", color="#4CAF50", label="acc")
        ax2.set_ylabel("acc", color="#4CAF50")
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += lb2
    if trans is not None:
        ax1.bar(batches, np.asarray(trans) / max(1, np.max(trans)) * (max(loss) or 1.0),
                alpha=0.2, color="gray", label="trans (norm)")
    ax1.legend(lines, labels, fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
