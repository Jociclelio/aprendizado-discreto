# exp-013 — subconjunto pequeno (B1 corrigido)

**Veredito: FAIL** — acc 0.1250 -> 0.1562 (Δ=+0.0312) vs majoritario 0.1562 (n=64, seed=42, 3 epochs = 192 passos).

## Diagnostico

- Preds antes: `[2, 0, 43, 2, 9, 1, 0, 3, 1, 3]` -> depois: `[0, 0, 0, 0, 0, 0, 0, 0, 64, 0]`.
- Atividade antes: [0.483, 0.521, 0.397] -> depois: [0.489, 0.531, 0.6].
- Surrogate quantizado nao-nulo (z1/z2/zout) antes: [0.993, 0.832, 0.938] -> depois: [0.148, 0.0, 0.0].
- |z| medio (z1/z2/zout) antes: [6.0, 5.0, 2.4] -> depois: [39.4, 29.8, 20.6].
- Transicoes efetivas por epoch: [22263, 0, 0] (bloqueios: [9938, 0, 0]).

## Primeiro ponto de falha

morte do surrogate de saida no passo 32 (q_fake(z_out)==0 em todo o subconjunto => deltas zero => treino congela; causa: escala/raio — |z_out| medio 20.6 >> raio); colapso de diversidade no passo 8 (todas as 64 preds -> classe 8); 2 epoch(s) congelada(s) (0 transicoes efetivas).

Proximo passo sugerido: investigar em 015 (margem/homeostase/leak) como manter |z| dentro do cone do surrogate e preservar diversidade de preds; nao adicionar mecanismos nesta task.

Artefatos: `subset.json`, `metrics.json`, `step_log.json`, `scores_before/after.npy`, `weights.npz`, `diag_trail` dentro de `metrics.json`.
