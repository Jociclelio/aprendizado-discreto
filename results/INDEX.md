# Índice de resultados — execução × interpretação

> Regra: **status de execução** (o experimento rodou? há artefato?) é separado de
> **resultado científico** (o que os números sustentam?).
> Auditoria: `docs/reports/revisao-f5d3a5a.md`. Tasks: `.tasks/INDEX.md`.

| Diretório | Experimento | Execução | Resultado científico | Artefatos-chave |
|---|---|---|---|---|
| `exp-001-baseline/` | baseline MLP | completa | positivo: teste até 97.97% (20 epochs) | `accuracy_curve.csv` |
| `test-002/` | (órfão, sem task) | — | — | órfão pré-Astra; não usar como evidência |
| `accuracy_curve_default.png` (raiz) | (órfão, sem task) | — | — | órfão pré-Astra; sem proveniência |
| `exp-007-b1-error-feedback/` | **ausente** | **não há** | inconclusivo | task 007 declara, diretório não existe |
| `exp-008-b2-{ef,fr,norm}/`, `exp-008-b3-{ef,fr,norm}/` | B2/B3 × resets | **não executados** (dirs vazios) | n/a | 6 dirs vazios = reserva de nomes, não evidência |
| `exp-009-comparison/` | comparação final | **não executada** (dir vazio) | n/a | dir vazio; superada por 026/027 |
| `exp-010-adaptive-radius/` | raio adaptativo | completa | negativo (9–10%, sem ganho) | `metrics.json`, `delta_fraction.png` |
| `exp-011-leaky-counters/` | leak | completa | negativo (sem evolução) | `metrics.json`, `synthetic.json`, PNGs |
| `exp-012-b1-reparo/` | reparo A–H + contrato | completa (22/22 por registro) | contrato fixado | `run_012_checks.py`, `metrics.json`, `NOTAS.md` |
| `exp-013-small-subset/` | subconjunto 64/sd42 + grade | completa (192 passos + trail) | **negativo**: empata majoritário; surrogate morto passo 32 | `REPORT.md`, `metrics.json`, `step_log.json`, `subset.json`, `scores_*.npy`, `weights.npz` |
| `exp-014-mnist-full/` | ensaio 3 seeds (10k/5k/10k) | completa | **negativo**: ~9%, colapso de predição | `metrics.json`, `curves.{csv,png}`, `provenance_seed*.json`, `confusion_matrix.png`, `weights_*.npz` |
| `exp-015-ablacoes/` | 13 configs + B+ | completa | marginal (margem ≈ base; homeostase/leak sem benefício; congela epoch 1) | `metrics.json`, `best_weights.npz` |
| `exp-016-benchmark/` | A vs B1 vs B+ (3 seeds) | completa | trade-off: −83pp por 3.98× inferência; treino 2.25×; 8=16 bits | `metrics.json`, `benchmark.{csv,png}`, `README.md` |
| `exp-017-playground/` | ferramenta (smokes) | completa (smokes) | n/a (demos, não experimento) | `tracer-bit-identity.log`, `smoke.log`, `figs/` (8 PNGs), `smoke-runner-b1/`, `smoke-comparison/` |
| `exp-017-playground/runner-b1/` | runner interativo | **não rastreado** | n/a | saída local; não é evidência versionada |
| `logs/` | logs diversos | — | — | ver conteúdo local |

## Notas de leitura

- "Completa" = script versionado + artefatos localizáveis; **não** significa hipótese confirmada.
- Números de B em 010/011/013/014/015/016 convergem para ~9–11% com congelamento após a 1ª epoch — padrão consistente com o mecanismo de 013 (surrogate de saída zera), não com 6 fenômenos independentes.
- Contadores 8 vs 16 bits idênticos (016) significam que a saturação nunca é alcançada porque o sinal morre antes — interpretação, não prova de suficiência de 8 bits.
