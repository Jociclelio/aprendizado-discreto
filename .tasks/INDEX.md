# Índice de tasks (001–017, 028) — histórico preservado + estado auditado

> 028 (ferramenta/infra, pi-task-cycle) executada fora da fila 019–027 por
> pedido explícito do usuário; a fila científica permanece reservada.

> Os arquivos `.tasks/NNN-*.md` são **histórico**: nenhum foi editado por 018.
> Este índice reconcilia o que cada task declara com as evidências existentes.
> Auditoria completa: `docs/reports/revisao-f5d3a5a.md`. Experimentos: `results/INDEX.md`.
> Fila seguinte (019–027): `.agents/plans/proxima-rodada/plan.md`.

| # | Task | Execução | Evidência | Resultado científico | Observação |
|---|---|---|---|---|---|
| 001 | baseline MLP | feita | `results/exp-001-baseline/accuracy_curve.csv` (20 epochs, teste até 97.97%) | positivo | referência; checkpoint escolhido pelo teste (limitação p/ 023) |
| 002 | utils e dados | feita | `src/utils.py`, `results/test-002/` (órfão) | ferramenta | — |
| 003 | fake gradient | feita | `src/fake_gradient.py` | ferramenta | quantização intermediária revista em 012(H) |
| 004 | núcleo ternário | feita | `src/ternary_net.py` | ferramenta | perda L1 documentada como tal em 012 |
| 005 | regras B1/B2/B3 | feita (primitivas) | `src/counter_rules.py` | parcial | primitivas existem; treinadores B2/B3 integrados **não** |
| 006 | resets | feita | `src/reset_policies.py` | parcial | normalize corrigido em 012(F) |
| 007 | B1 inicial | **parcial; declaração sem evidência** | `src/ternary_b1.py` existe; `results/exp-007-b1-error-feedback/` **ausente**; relato: 50 amostras/2 epochs ~9–10% | **inconclusivo** (avaliador P0-A posterior) | critérios `[x]` não correspondem a evidências; não reexecutar — caminho é 024 |
| 008 | B2/B3 × resets | **pendente** (todos `[ ]`) | `src/ternary_b2.py`/`ternary_b3.py` ausentes; `results/exp-008-*/` **vazios** | n/a | **substituída**: máscaras → 021; comparação de variantes → 025 (condicionada a 024); contrato → 020 |
| 009 | comparação final | **pendente** (todos `[ ]`) | `src/compare.py` ausente; `results/exp-009-comparison/` **vazio** | n/a | **substituída**: benchmark corrigido → 026; consolidação → 027 |
| 010 | raio adaptativo | feita | `results/exp-010-adaptive-radius/metrics.json` (~9–10%) | negativo (sem ganho nesses registros) | heurística mantida como default; justificativa estatística ressalvada no Astra §8 |
| 011 | leaky counters | feita | `results/exp-011-leaky-counters/metrics.json` (+ sintético) | negativo (sem evolução documentada) | reexaminado com B1 corrigido em 015 |
| 012 | repara B1 (A–H) | feita | `results/exp-012-b1-reparo/{run_012_checks.py,metrics.json,NOTAS.md}` (22/22 por registro); CONTRATO_012 em `src/ternary_b1.py` | contrato fixado | aceito por registro (estático em 018); rede de segurança em 019 |
| 013 | subconjunto | feita | `results/exp-013-small-subset/{REPORT.md,metrics.json,step_log.json,subset.json}` | **negativo explícito**: 0.1250→0.1562 = majoritário; surrogate morto passo 32 | critério de sucesso permanece `[ ]` — correto |
| 014 | ensaio auditável | feita | `experiments/run_mnist_full.py`; `results/exp-014-mnist-full/{metrics.json,curves.csv,provenance_seed*.json,confusion_matrix.png}` (3 seeds, split 10k/5k/10k) | **negativo**: ~9% com colapso (seed 42 tudo → classe 8) | processo cumprido; números são evidência negativa |
| 015 | ablações | feita | `experiments/run_ablations.py`; `results/exp-015-ablacoes/metrics.json` (13 configs); `src/ternary_refined.py` | **marginal**: margem não piora (+0.009 val); homeostase sem benefício; leak sem efeito; congelam após epoch 1 só base/leak/`homeo_0.25` (`base [19896,0,0]`, leak `~[19915,0,0]`, `homeo_0.25 [57829,0,0]`); margem/combinações transitam (ex. `margin_1 [168146,134101,131680]`) mas seguem ~9–10% | `[x]` = testado, não hipótese confirmada; reavaliação → 025 |
| 016 | benchmark | feita | `experiments/run_benchmark.py`; `results/exp-016-benchmark/{metrics.json,benchmark.csv,benchmark.png,README.md}` (3 seeds) | **trade-off**: A 92–93% vs B ~9–10%; inferência 3.98×, treino 2.25×; 8/16 bits idênticos | sem reivindicação de MACs zero; refazer com protocolo 023 em 026 |
| 017 | playground | feita | `src/playground/{tracer,app,state,plots}.py`; `results/exp-017-playground/{tracer-bit-identity.log,smoke.log,figs/,smoke-runner-b1/,smoke-comparison/}` | ferramenta (demos curtas, não experimento) | `streamlit run` não reexecutado em 018 (evidência = smoke 2026-10-03); B2/B3 só como máscara no tracer |
| 018 | organiza docs e fila (esta) | esta revisão | `docs/reports/revisao-f5d3a5a.md`, `docs/ideia.md`, este índice, `results/INDEX.md`, README fiel | documental | sem código novo; sem reescrita de evidências; textos 001–017 intactos |
| 028 | port do workflow task-cycle (opencode → pi) | feita | `~/Documentos/pidev/pi-task-cycle/` (extensão, 4 skills, pool multi-modelo, `install.sh`); testes `node --test` 16/16; e2e real sd2 (executor commitou entrega+registro; revisor deu parecer com identidade confirmada); skills visíveis em sessão nova | ferramenta (infra de agentes) | instalação global em `~/.pi/agent/settings.json`; convenção de projetos `.agents/` intercambiável com opencode |

## Destinos formais (resumo)

- **008 → 020/021/025**: contrato (020), unificação das máscaras (021), reavaliação condicionada a 024 (025).
- **009 → 026/027**: benchmark com protocolo corrigido (026), consolidação com negativos e adiamentos (027).
- **007**: sem destino de reexecução; superada por 012 (reparo) + 024 (diagnóstico com controles).
