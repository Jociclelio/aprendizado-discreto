"""
[017] Playground web — stepper amostra a amostra B1/B2/B3 sobre MNIST.

Execucao:
    streamlit run src/playground/app.py

Nenhum arquivo `src/*.py` existente e modificado: o app so le/importa
(TernaryNet, CounterManager, train_batch/evaluate, fake-grad,
counter_rules, plots de utils) e delega a matematica ao tracer
(`src/playground/tracer.py`), bit-identico ao CLI.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.playground import plots
from src.playground.state import (
    clamp_idx,
    config_line,
    ensure_state,
    get_mnist,
    reset_net,
    trace_fn_for,
)

LAYER_NAMES = ["L0 784->128", "L1 128->64", "L2 64->10"]
N_CAP = 2000  # cap interativo; full MNIST fica no CLI

st.set_page_config(page_title="Rede B — Playground 017", layout="wide")
st.title("Rede B — Playground do treino (017)")
st.caption(
    "Stepper amostra a amostra B1/B2/B3 sobre digitos reais do MNIST. "
    "Matematica = CONTRATO_012, bit-identica ao CLI (`src/playground/tracer.py`)."
)

# ---------------------------------------------------------------------------
# Sidebar (hiperparametros + navegacao da amostra)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuracao")
    variant = st.selectbox("Variante", ["B1", "B2", "B3"])
    scale = st.number_input("scale (amplitude fake-grad)", 0.1, 5.0, 1.0, 0.1)
    adaptive = st.checkbox("raio adaptativo c*sqrt(fan-in)", value=True)
    if adaptive:
        scale_factor = st.slider("c (fator do raio)", 0.25, 2.0, 1.0, 0.05)
        radius = 1.0
    else:
        scale_factor = 1.0
        radius = st.number_input("radius (base do triangulo)", 0.5, 60.0, 1.0, 0.5)
    levels = st.selectbox("levels (quantizacao)", [3, 5, 7], index=1)
    topk_frac = st.slider("topk_frac (B2/B3)", 0.001, 0.10, 0.01, 0.001)
    seed = st.number_input("seed", 0, 99999, 42, 1)
    batch_size = st.number_input("batch_size (runner)", 8, 256, 32, 8)
    st.divider()
    st.header("Amostra MNIST")
    idx_in = st.number_input(
        "indice (treino)", 0, 59999, int(st.session_state.get("sample_idx", 0)), 1
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        prev_b = st.button("Prev")
    with c2:
        next_b = st.button("Next")
    with c3:
        rand_b = st.button("Aleatorio")
    if st.button("Reset rede (seed)", use_container_width=True):
        reset_net(int(seed))
        st.success(f"Rede restaurada da seed {int(seed)}.")

params = {
    "variant": variant,
    "scale": float(scale),
    "adaptive_radius": bool(adaptive),
    "scale_factor": float(scale_factor),
    "radius": float(radius),
    "levels": int(levels),
    "seed": int(seed),
    "topk_frac": float(topk_frac),
    "batch_size": int(batch_size),
}
ensure_state(params)

train_data, train_labels, test_data, test_labels = get_mnist()
n_train = train_data.shape[0]

if prev_b:
    st.session_state["sample_idx"] = clamp_idx(int(idx_in) - 1, n_train)
    st.rerun()
if next_b:
    st.session_state["sample_idx"] = clamp_idx(int(idx_in) + 1, n_train)
    st.rerun()
if rand_b:
    st.session_state["sample_idx"] = int(np.random.randint(0, n_train))
    st.rerun()
st.session_state["sample_idx"] = clamp_idx(int(idx_in), n_train)
idx = int(st.session_state["sample_idx"])

net = st.session_state["net"]
cm = st.session_state["cm"]
steps_done = int(st.session_state.get("steps_done", 0))

st.info(f"`{config_line(params, idx)}` | steps nesta sessao: {steps_done}")

tab_step, tab_insp, tab_run, tab_cmp = st.tabs(
    ["1 Stepper", "2 Inspetor de camada", "3 Runner curto", "4 Comparador B1/B2/B3"]
)


def do_step(sample_idx: int):
    """Roda 1 trace na amostra corrente (mesmos numeros do CLI)."""
    trace_fn = trace_fn_for(params["variant"])
    kw = dict(
        scale=params["scale"],
        radius=params["radius"],
        levels=params["levels"],
        adaptive_radius=params["adaptive_radius"],
        scale_factor=params["scale_factor"],
    )
    if params["variant"] in ("B2", "B3"):
        kw["topk_frac"] = params["topk_frac"]
    x = train_data[sample_idx]
    y = int(train_labels[sample_idx])
    tr = trace_fn(net, cm, x, y, **kw)
    st.session_state["trace"] = tr
    st.session_state["steps_done"] = int(st.session_state.get("steps_done", 0)) + 1
    return tr


def _qstats(m, l):
    """Acesso tolerante a quant_stats (chave int ou str apos JSON)."""
    qs = m["quant_stats"]
    return qs[l] if l in qs else qs[str(l)]


# ---------------------------------------------------------------------------
# Aba 1 — Stepper: os 7 estagios do pipeline na ordem do CONTRATO_012
# ---------------------------------------------------------------------------
with tab_step:
    st.header("Stepper amostra a amostra")
    b_step, b_replay = st.columns(2)
    with b_step:
        if st.button("Step — treinar 1 amostra", type="primary", use_container_width=True):
            do_step(idx)
            st.rerun()
    with b_replay:
        if st.button("Replay — Reset + Step", use_container_width=True):
            reset_net(params["seed"])
            do_step(idx)
            st.rerun()

    tr = st.session_state.get("trace")
    x = train_data[idx]
    y = int(train_labels[idx])

    if tr is None or tr.get("label") != y or not np.array_equal(tr.get("x"), x):
        scores, _, _, _ = net.forward_with_scores(x.reshape(1, -1).astype(np.float32))
        pred0 = int(np.argmax(scores[0]))
        st.warning("Nenhum Step nesta amostra ainda — forward atual (sem treinar).")
        c_img, c_fw = st.columns([1, 2])
        with c_img:
            fig = plots.fig_digit(x.reshape(28, 28), y, pred0)
            st.pyplot(fig)
            plt.close(fig)
        with c_fw:
            st.write(f"**scores (z_out):** `{np.round(scores[0], 1).tolist()}`")
            st.write(f"**pred={pred0} · label={y}** · acerto: {pred0 == y}")
    else:
        m = tr["metrics"]
        st.subheader("1 · Forward — z, ativacoes, scores, pred")
        c_img, c_fw = st.columns([1, 2])
        with c_img:
            fig = plots.fig_digit(x.reshape(28, 28), y, tr["pred"])
            st.pyplot(fig)
            plt.close(fig)
        with c_fw:
            st.write(f"**scores (z_out):** `{np.round(tr['scores'], 1).tolist()}`")
            st.write(f"**pred={tr['pred']} · label={tr['label']}** · "
                     f"acerto: {tr['pred'] == tr['label']}")
            st.write(f"firing por camada: {[f'{v:.3f}' for v in m['firing_rates']]} "
                     f"(saida ativa: {m['fired_neurons']})")
        fig = plots.fig_z_hist(tr["z_list"], tr["radii"])
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("2 · Erro — one-hot, error, loss L1")
        st.write(f"error (a_out − one_hot): `{tr['error'][0].tolist()}`")
        st.write(f"**loss = {m['loss']:.4f}** (media |erro| por neuronio de saida)")
        st.subheader("3 · Fake-grad triangular continuo")
        for l in range(3):
            fc = np.asarray(tr["fake_cont"][l])
            st.write(f"{LAYER_NAMES[l]}: raio={tr['radii'][l]:.2f} · "
                     f"fake medio={fc.mean():.4f} · fracao>0: {np.mean(fc > 0):.1%}")
        fig = plots.fig_surrogate(tr["z_list"][2], tr["fake_cont"][2],
                                  tr["radii"][2], params["scale"], "saida L2")
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("4 · Fake-grad quantizado (surrogate inteiro)")
        for l in range(3):
            q = np.asarray(tr["q_fake"][l])
            vals, counts = np.unique(q, return_counts=True)
            st.write(f"{LAYER_NAMES[l]}: {dict(zip(vals.tolist(), counts.tolist()))}")
        if np.all(np.asarray(tr["q_fake"][2]) == 0):
            st.error("`q_fake==0` na saida: surrogate zerou (|z_out| >> raio) — "
                     "rede CONGELA nesta amostra (0 deltas, 0 transicoes). Ver task 013.")
        st.subheader("5 · Deltas por camada + acumulacao (round)")
        for l in range(3):
            d = np.asarray(tr["deltas"][l]).ravel()
            qs = _qstats(m, l)
            st.write(f"{LAYER_NAMES[l]}: delta max|.|={np.abs(d).max():.4f} "
                     f"media|.|={np.mean(np.abs(d)):.4f} · "
                     f"dw nz float→int: {qs['dw_nz_float']}→{qs['dw_nz_int']}")
        st.subheader("6 · Mascara unica por camada (update E reset)")
        for l, pl in enumerate(m["per_layer"]):
            st.write(f"{LAYER_NAMES[l]}: thr_w={pl['thr_w']:.2f} thr_b={pl['thr_b']:.2f} · "
                     f"pesos sel={pl['selected']} · bias sel={pl['bias_selected']}")
        st.write(f"k_info: `{tr.get('k_info')}`")
        st.subheader("7 · Update W<-clip(W−sign(C)) + reset (saturacao consome contador)")
        st.table([
            {"camada": LAYER_NAMES[l], "sel": pl["selected"], "bloq": pl["blocked"],
             "trans": pl["transitions"], "bias_sel": pl["bias_selected"],
             "bias_trans": pl["bias_transitions"]}
            for l, pl in enumerate(m["per_layer"])
        ])
        st.write(f"**Total pesos:** sel={m['selected_weights']} "
                 f"bloq={m['blocked_weights']} trans={m['transitions']} · "
                 f"**Bias:** sel={m['bias_selected']} bloq={m['bias_blocked']} "
                 f"trans={m['bias_transitions']}")

# ---------------------------------------------------------------------------
# Aba 2 — Inspetor de camada L0/L1/L2 (heatmaps sempre amostrados)
# ---------------------------------------------------------------------------
with tab_insp:
    st.header("Inspetor de camada")
    tr = st.session_state.get("trace")
    if tr is not None:
        z_list, radii = tr["z_list"], tr["radii"]
        Cw_show = tr["Cw_acc"]
        st.caption("z e C exibidos do ultimo Step (pos-acumulacao, pre-reset).")
    else:
        from src.fake_gradient import get_layer_radii

        _, _, pre, _ = net.forward_with_scores(x.reshape(1, -1).astype(np.float32))
        radii = get_layer_radii([784, 128, 64], scale_factor=params["scale_factor"]) \
            if params["adaptive_radius"] else [params["radius"]] * 3
        z_list = pre
        Cw_show = cm.weight_counters
        st.caption("Sem Step ainda: forward atual (sem treinar).")
    stats = net.get_stats()
    fig = plots.fig_weight_dist(stats["layer_stats"])
    st.pyplot(fig)
    plt.close(fig)
    fig = plots.fig_z_hist(z_list, radii)
    st.pyplot(fig)
    plt.close(fig)
    layer_sel = st.radio("Camada", [0, 1, 2],
                         format_func=lambda i: LAYER_NAMES[i], horizontal=True)
    l = int(layer_sel)
    c_w, c_c = st.columns(2)
    with c_w:
        fig = plots.fig_heatmap_sampled(net.layers[l].weights.astype(np.float64),
                                        f"W {LAYER_NAMES[l]}", vmin=-1, vmax=1)
        st.pyplot(fig)
        plt.close(fig)
    with c_c:
        fig = plots.fig_heatmap_sampled(np.asarray(Cw_show[l]).astype(np.float64),
                                        f"C (contadores) {LAYER_NAMES[l]}")
        st.pyplot(fig)
        plt.close(fig)
    fig = plots.fig_counter_hist(Cw_show)
    st.pyplot(fig)
    plt.close(fig)
    from src.fake_gradient import triangular_fake_grad

    fig = plots.fig_surrogate(
        z_list[l],
        triangular_fake_grad(np.asarray(z_list[l]), scale=params["scale"], radius=radii[l]),
        radii[l], params["scale"], LAYER_NAMES[l])
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Aba 3 — Runner curto (cap 2000; full MNIST fica no CLI)
# ---------------------------------------------------------------------------
with tab_run:
    st.header("Runner curto")
    n_run = st.number_input("N amostras (cap 2000)", 32, N_CAP, 500, 32)
    eval_every = st.number_input("avaliar a cada K batches", 1, 50, 5, 1)
    if st.button("Rodar treino curto", type="primary"):
        from src.ternary_b1 import evaluate, train_batch
        from src.utils import ensure_dir, save_metrics

        n_run = min(int(n_run), N_CAP)
        bs = params["batch_size"]
        perm = np.random.RandomState(params["seed"]).permutation(n_train)[:n_run]
        sx, sy = train_data[perm], train_labels[perm]
        reset_net(params["seed"])
        net_r, cm_r = st.session_state["net"], st.session_state["cm"]
        kw = dict(scale=params["scale"], radius=params["radius"],
                  levels=params["levels"], adaptive_radius=params["adaptive_radius"],
                  scale_factor=params["scale_factor"])
        hist = {"loss": [], "trans": [], "acc": [], "batches": []}
        log_lines = []
        prog = st.progress(0.0)
        chart_ph = st.empty()
        n_batches = (n_run + bs - 1) // bs
        for b in range(n_batches):
            sl = slice(b * bs, min((b + 1) * bs, n_run))
            if params["variant"] == "B1":
                bm = train_batch(net_r, cm_r, sx[sl], sy[sl], **kw)
                bl, bt = float(bm["loss"]), int(bm["transitions"])
            else:
                fn = trace_fn_for(params["variant"])
                bl_acc, bt_acc = 0.0, 0
                for i in range(sl.start, sl.stop):
                    t = fn(net_r, cm_r, sx[i], int(sy[i]),
                           topk_frac=params["topk_frac"], **kw)
                    bl_acc += t["metrics"]["loss"]
                    bt_acc += t["metrics"]["transitions"]
                n_b = sl.stop - sl.start
                bl, bt = bl_acc / n_b, bt_acc
            hist["loss"].append(bl)
            hist["trans"].append(bt)
            hist["batches"].append(b + 1)
            log_lines.append(f"batch {b + 1:3d}/{n_batches} | loss={bl:.4f} | trans={bt}")
            if (b + 1) % int(eval_every) == 0 or (b + 1) == n_batches:
                acc = evaluate(net_r, test_data[:2000], test_labels[:2000])
                hist["acc"].append(float(acc))
                log_lines[-1] += f" | acc2000={acc:.4f}"
            else:
                hist["acc"].append(hist["acc"][-1] if hist["acc"] else 0.0)
            prog.progress((b + 1) / n_batches)
            fig = plots.fig_runner_curve(hist["batches"], hist["loss"],
                                         acc=hist["acc"], trans=hist["trans"])
            chart_ph.pyplot(fig)
            plt.close(fig)
        outdir = Path("results") / "exp-017-playground" / f"runner-{params['variant'].lower()}"
        ensure_dir(outdir)
        save_metrics({"variant": params["variant"], "n": n_run, "history": hist,
                      "config": config_line(params, idx)}, str(outdir / "metrics.json"))
        fig = plots.fig_runner_curve(hist["batches"], hist["loss"],
                                     acc=hist["acc"], trans=hist["trans"])
        fig.savefig(str(outdir / "curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        (outdir / "log.txt").write_text("\n".join(log_lines) + "\n")
        st.success(f"Runner concluido: loss={hist['loss'][-1]:.4f} "
                   f"acc2000={hist['acc'][-1]:.4f} -> {outdir}")
        with st.expander("Log textual (igual ao CLI)"):
            st.code("\n".join(log_lines[-20:]))

# ---------------------------------------------------------------------------
# Aba 4 — Comparador B1/B2/B3 (mesmo seed + mesmo subset)
# ---------------------------------------------------------------------------
with tab_cmp:
    st.header("Comparador B1/B2/B3")
    n_cmp = st.number_input("N amostras do subset (cap 2000)", 32, N_CAP, 300, 32)
    if st.button("Comparar variantes", type="primary"):
        from src.playground.state import make_net_cm
        from src.ternary_b1 import evaluate
        from src.utils import ensure_dir, plot_comparison_curves, save_metrics

        n_cmp = min(int(n_cmp), N_CAP)
        perm = np.random.RandomState(params["seed"]).permutation(n_train)[:n_cmp]
        sx, sy = train_data[perm], train_labels[perm]
        kw = dict(scale=params["scale"], radius=params["radius"],
                  levels=params["levels"], adaptive_radius=params["adaptive_radius"],
                  scale_factor=params["scale_factor"])
        ckpt = max(1, n_cmp // 6)
        acc_curves, table = {}, []
        prog = st.progress(0.0)
        for vi, var in enumerate(["B1", "B2", "B3"]):
            net_c, cm_c = make_net_cm(params["seed"])
            fn = trace_fn_for(var)
            steps_axis, accs = [], []
            tot_sel = tot_bloq = tot_trans = 0
            for i in range(n_cmp):
                extra = {"topk_frac": params["topk_frac"]} if var in ("B2", "B3") else {}
                t = fn(net_c, cm_c, sx[i], int(sy[i]), **extra, **kw)
                m = t["metrics"]
                tot_sel += m["selected_weights"]
                tot_bloq += m["blocked_weights"]
                tot_trans += m["transitions"]
                if (i + 1) % ckpt == 0 or (i + 1) == n_cmp:
                    acc = evaluate(net_c, test_data[:2000], test_labels[:2000])
                    steps_axis.append(i + 1)
                    accs.append(float(acc))
                prog.progress((vi * n_cmp + i + 1) / (3 * n_cmp))
            acc_curves[var] = {"steps": steps_axis, "acc": accs}
            table.append({"variante": var, "acc2000_final": round(accs[-1], 4),
                          "transicoes": tot_trans, "bloqueios": tot_bloq,
                          "%bloq": round(100 * tot_bloq / max(1, tot_sel), 2)})
        outdir = Path("results") / "exp-017-playground" / "comparison"
        ensure_dir(outdir)
        datasets = [{"epochs": acc_curves[v]["steps"], "train_acc": acc_curves[v]["acc"],
                     "test_acc": acc_curves[v]["acc"], "label": v}
                    for v in ["B1", "B2", "B3"]]
        plot_comparison_curves(datasets, title="B1 vs B2 vs B3 (mesmo seed+subset)",
                               save_path=str(outdir / "comparison.png"))
        save_metrics({"n": n_cmp, "table": table, "config": config_line(params, idx)},
                     str(outdir / "metrics.json"))
        st.image(str(outdir / "comparison.png"))
        st.table(table)
        st.success(f"Comparacao salva em {outdir}")
