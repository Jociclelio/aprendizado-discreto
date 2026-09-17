# exp-012 — Reparo do B1: notas da reparação

## Contrato matemático (fixado em `src/ternary_b1.py`, docstring CONTRATO_012)

- `z_l = W_l @ x_{l-1} + b_l`, `a_l = step(z_l)` (`1` se `z > 0`).
- `scores = z_out` (pré-ativação da saída); `pred = argmax(scores)`.
- `C` acumula **gradiente** ⇒ descida `W ← clip(W − sign(C))`,
  `b ← clip(b − sign(C_b), −N, N−1)`. Sem `sign(−error)` extra; cada camada
  usa seu próprio `deltas[l]`.
- Seleção: máscara única por camada `|C| > max(1, mean|C|)` para update **e**
  reset. Sem gate de disparo (firing é diagnóstico; inativos podem recuperar).
- Saturação: tentativa bloqueada **consome** o contador (reset) e é contada
  em `blocked`, separada de `transitions` efetivas.
- Leak simétrico, zero ponto fixo:
  `gamma: sign(C)·floor(gamma·|C|)`,
  `shift: sign(C)·(|C|−step)`, `step = max(1, |C|>>s)` p/ `|C|>0`.
  (Bitshift puro `C − (C>>s)` é floor/assimétrico — removido.)
- Quantização: surrogate quantizado também nas ocultas
  (`propagate_delta_with_fake_quantized`) + `round()` no acúmulo
  (nunca truncamento `astype(int)` direto).

## Correções por achado

| Achado | Correção | Arquivo |
|---|---|---|
| A — avaliação sem step na 2ª oculta | `forward_with_scores()` único; `evaluate()` consome `scores` | `ternary_net.py`, `ternary_b1.py` |
| B — `sign(C)·sign(−error)` invertia saída | descida pura `−sign(C)` em todas as camadas; primitivas B1/B2/B3 alinhadas (`delta_sign` depreciado/ignorado, testes atualizados) | `ternary_b1.py`, `counter_rules.py` |
| C — ocultas reusavam delta errado | `deltas[0..2]` explícitos por camada | `ternary_b1.py` |
| D — update/reset com máscaras distintas | máscara única por camada p/ update+reset; reconciliado com `apply_update_neuron()` | `ternary_b1.py` |
| E — bias não aprendia | `delta_bias` acumulado + update de `layer.bias` em `[−N, N−1]` | `ternary_b1.py` |
| F — normalize ampliava escala + `IndexError` | fator `target/mean` + `rint` inteiro + índices `int64` (incl. `all_layers`) | `reset_policies.py` |
| G — leaks assimétricos opostos | política simétrica documentada (gamma e shift), `[-2..2]→[-1,0,0,0,1]` | `counter_rules.py` |
| H — truncamento `<1→0` nas ocultas | surrogate oculto quantizado + `round()` + `quant_stats` pré/pós | `ternary_b1.py`, `fake_gradient.py` |
| P2 — `updated` misturava tentativas/bloqueios | `selected/attempts/blocked/transitions` (+ bias); `updated_weights`=transições (legado) | `ternary_b1.py`, `counter_rules.py` |
| P2 — `DEFAULT` vs assinatura | `train_b1(adaptive_radius=True)` alinhado ao `DEFAULT_HYPERPARAMS` | `ternary_b1.py` |
| P2 — loss dizia "BCE" | docstring: L1 médio + `firing_rates()` anti-colapso | `ternary_net.py` |
| P2 — `K=0` selecionava tudo | `K<=0` → seleção vazia (`CounterManager` e `CounterAccumulator`) | `counter_rules.py`, `fake_gradient.py` |
| P2 — checkpoint sem bias | `weights.npz` agora salva pesos + bias + contadores | `ternary_b1.py` |
| P2 — `normalize()` virava float | `CounterAccumulator.normalize()` int-safe (`rint`, mantém `int64`) | `fake_gradient.py` |

## Casos mínimos do relatório (seção 10) — após o reparo

- Avaliação pequena → **1.0** (antes 0.0).
- Classe correta inativa → peso de saída vai para **+1** (antes −1):
  `weights[:, 0] == [1, 0]` (classe errada com erro 0 não se move).

## Validação

- `run_012_checks.py`: **22/22** (A–H + P2). → `metrics.json`
- Existentes: `python -m src.ternary_b1`, `src.counter_rules`,
  `src.reset_policies`, `src.fake_gradient`, `src.ternary_net` — todos passam.
- Mini-treino (50 amostras, 2 épocas): loss 0.52→0.62, acc 0.11→0.12 —
  sem colapso de execução; aprendizado real é objeto da task [013]
  (prova em subconjunto), não exigido aqui.

## Decisões que a próxima task deve respeitar

- Não reintroduzir gate de disparo sem experimento (015 faz ablações).
- Margem/homeostase ficam em 015 — este diff é só correção.
- `delta_sign` nas primitivas é legado ignorado; remover na próxima
  limpeza de API se desejado.
