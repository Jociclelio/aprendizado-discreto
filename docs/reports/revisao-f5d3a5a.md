# Revisão `f5d3a5a` — reconciliação de documentação, fila e evidências

**Commit revisado:** `f5d3a5ada972343fc9ea5c96a9b5b2beeae62b89` ([017] playground)
**Relatório Astra de referência:** `astra-relatório-commit-6f47914.md` (análise do commit `6f47914`; preservado, não reescrito)
**Task de origem:** `.opencode/tasks/proxima-rodada/018-organiza-documentacao-e-fila.md`
**Plano:** `.opencode/plans/proxima-rodada/plan.md`
**Método:** análise estática (leitura de tasks, código, CSV/JSON/logs versionados). Nenhum treino foi reexecutado; nenhum código de `src/` foi alterado.

## 1. O que esta revisão entrega

- Separa **proposta científica** (`docs/ideia.md`) de **estado real** (`README.md` fiel aos arquivos).
- Audita os critérios de `007/013/014/015/016/017` como **execução × evidência × resultado científico** (seções 3–4).
- Dá **destino formal** a `008/009` sem falsa declaração de execução e sem alterar seus textos históricos (seção 5).
- Cria os índices `.tasks/INDEX.md` e `results/INDEX.md` (status de execução separado de interpretação científica).
- Lista as divergências README × arquivos reais corrigidas no `README.md` (seção 6).

## 2. Fontes consultadas (por arquivo)

| Fonte | O que foi usado |
|---|---|
| `astra-relatório-commit-6f47914.md` §§ 1, 4–7, 9–10 | defeitos P0 (A–D) / P1 (E–H), contabilidade de memória, sequência recomendada Etapas 1–5 |
| `.tasks/007-b1-initial.md` | critérios marcados `[x]`; seção Resultados (50 amostras, 2 epochs, acc ~9–10%) |
| `.tasks/008-b2-b3-variants.md`, `.tasks/009-comparison.md` | critérios todos `[ ]`; escopo (treinadores B2/B3, `compare.py`, 6 configs, comparação final) |
| `.tasks/012-repara-b1-contrato-matematico.md` + `results/exp-012-b1-reparo/{NOTAS.md,metrics.json,run_012_checks.py}` | contrato CONTRATO_012, 22/22 checks, casos mínimos 1.0 / +1 |
| `.tasks/013-prova-aprendizado-subconjunto.md` + `results/exp-013-small-subset/{REPORT.md,metrics.json,step_log.json,subset.json}` | ramo FALHA explícito: 0.1250 → 0.1562 vs majoritário 0.1562; surrogate de saída zera no passo 32 |
| `.tasks/014-ensaio-mnist-auditable.md` + `results/exp-014-mnist-full/{metrics.json,curves.csv,provenance_seed{42,123,999}.json,confusion_matrix.png}` | 3 sementes, split 10k/5k/10k, colapso de predição (seed 42: 10000/10000 → classe 8) |
| `.tasks/015-ablacoes-margin-homeostasis-leak.md` + `results/exp-015-ablacoes/metrics.json` + `src/ternary_refined.py` | 13 configs; `margin_2` val 0.1022/test 0.098 vs base 0.0932/0.0974; `homeo_0.25` 0.0924/0.0849; transições `[~19900, 0, 0]` |
| `.tasks/016-benchmark-eficiencia.md` + `results/exp-016-benchmark/{metrics.json,benchmark.csv,benchmark.png,README.md}` | A ~92–93% vs B1/B+ ~9–10%; 3.98× inferência, 2.25× treino; 8/16 bits idênticos |
| `.tasks/017-playground-viz.md` + `results/exp-017-playground/{tracer-bit-identity.log,smoke.log,figs/,smoke-runner-b1/,smoke-comparison/}` + `src/playground/{tracer,app,state,plots}.py` | bit-identidade passa; smoke runner/comparador OK; B2/B3 só como máscara no tracer |
| `src/{baseline,counter_rules,fake_gradient,reset_policies,ternary_b1,ternary_net,ternary_refined,ablation,utils}.py`, `experiments/run_{small_subset,mnist_full,ablations,benchmark}.py`, `requirements.txt` | árvore real (seção 6) |
| `AGENTS.md` | convenção de fila (uma task por vez, commit `[NNN]`) respeitada: textos de `.tasks/001–017` não foram editados |

## 3. Auditoria de critérios (execução × evidência)

Legenda: **Exec** = trabalho executado com artefato; **Evid** = artefato versionado localizável; **Res** = resultado científico (positivo / negativo / inconclusivo).

### 3.1 [007] B1 inicial — declaração sem evidência correspondente

- O arquivo marca todos os critérios `[x]`, mas:
  - `results/exp-007-b1-error-feedback/` **não existe** (verificado em 2026-10-04);
  - o próprio campo Resultados relata só "mini treino (50 samples, 2 epochs): acc ~9–10%", não as 20 epochs do critério;
  - o critério "acurácia > 10%" não é sustentado pelo número relatado (~9–10%);
  - o avaliador usado à época foi depois demonstrado incorreto (Astra P0-A).
- **Veredito:** execução parcial histórica; critérios `[x]` **não** correspondem a evidências. O texto da task é **preservado** (não editado); a correção auditável é este registro + `.tasks/INDEX.md`.
- **Ação futura:** não reexecutar 007; o caminho é 024 (diagnóstico com controles).

### 3.2 [013] Subconjunto — resultado negativo explícito e válido

- Critério "Δ acima do majoritário" permanece `[ ]` — **correto**.
- Evidência: `metrics.json` + `REPORT.md` + `step_log.json` (192 passos, trilha a cada 8).
- Resultado: acc 0.1250 → 0.1562, majoritário 0.1562 (empate); colapso de diversidade no passo 8; surrogate de saída morto no passo 32 (`|z_out|` 2.4 → 20.6 >> raio 8); epochs 2–3 congeladas (transições `[22263, 0, 0]`).
- **Veredito:** execução completa, resultado **negativo** — mantém-se explícito. É o ponto de partida de 024/025, não um fracasso a esconder.

### 3.3 [014] Ensaio auditável — execução completa, resultado negativo

- Todos os critérios `[x]` com evidência: `experiments/run_mnist_full.py`, 3 provenances, `curves.csv`, matriz de confusão.
- Limitação registrada: `train_size: 10000` (subconjunto de 10k, não os 60k) — ensaio auditável, não full-MNIST literal.
- Resultado científico: B1 val ~9.2–9.3%, teste ~8.9–9.7%, predição colapsada (seed 42: tudo → classe 8). Execução ≠ sucesso científico.
- **Veredito:** critérios de processo cumpridos; números são evidência negativa, não baseline de capacidade.

### 3.4 [015] Ablações — execução completa, hipótese só parcialmente sustentada

- Todos os critérios `[x]` com evidência: 13 configs em `metrics.json`, `src/ternary_refined.py` consolidado.
- Leitura fiel dos números (seed 42): `margin_2` 0.1022/0.098 vs base 0.0932/0.0974 (ganho marginal, "não piora" sustentado no limite); `homeo_0.25` 0.0924/0.0849 (**piora**; critério "reduz colapso sem sacrificar acurácia" **não demonstrado** — transições seguem `[~19900, 0, 0]`, i.e. congela após epoch 1 em todas as configs).
- **Veredito:** o `[x]` marca *teste isolado executado*, não *hipótese confirmada*. Reavaliação condicionada a 024 fica para 025.

### 3.5 [016] Benchmark — execução completa, conclusão de trade-off (não de superioridade)

- Todos os critérios `[x]` com evidência: `run_benchmark.py`, `metrics.json`, `benchmark.csv/png`, `README.md` do exp.
- Resultado: A 92–93% vs B1/B+ 9–10% (3 sementes); inferência 3.98× menor; treino 2.25× maior (sem contar momentos do Adam — limitação já registrada no próprio README do exp); 8 vs 16 bits idênticos (saturação nunca alcançada porque o sinal morre antes — interpretação, não prova de suficiência).
- "Zero MACs float" corretamente **não** reivindicado para o código NumPy atual.
- **Veredito:** benchmark válido como *medida do estado atual*, não como validação de eficiência. Correção metodológica (seleção/checkpoints) fica para 023/026.

### 3.6 [017] Playground — execução completa (ferramenta, não experimento)

- Todos os critérios `[x]` com evidência: `tracer-bit-identity.log` ("Todos os testes passaram (bit-identico ao CLI)"), `smoke.log` ("SMOKE-017 OK"), 8 PNGs em `figs/`, `smoke-runner-b1/`, `smoke-comparison/`, `runner-b1/` (este último **não rastreado** — ver seção 7).
- Limitação desta revisão estática: `streamlit run src/playground/app.py` **não foi reexecutado aqui**; a evidência de abertura sem erro é o smoke log de 2026-10-03.
- B2/B3 no playground são **trocas de máscara no tracer**, não treinadores independentes — consistente com 008 pendente, sem contradição.
- **Veredito:** ferramenta entregue; números do runner/comparador são demonstrações curtas (cap 2000 / 500 amostras), não resultados científicos.

### 3.7 [012] Reparo B1 — execução completa, verificada por registro

- Todos os critérios `[x]`; evidência: `run_012_checks.py`, `metrics.json` (22/22 por registro), `NOTAS.md`, casos mínimos do Astra §10 com resultado esperado (1.0 / +1) por registro.
- Não reexecutado nesta revisão estática; aceito por registro auditável. Qualquer regressão será capturada por 019 (suíte de testes) e 021 (unificação).

## 4. Separação execução × resultado (regra geral)

| Exp | Execução | Resultado científico |
|---|---|---|
| 012 | completa (22/22 checks) | contrato fixado; base para medir a técnica |
| 013 | completa | **negativo**: só aprende o prior, depois congela |
| 014 | completa (3 seeds, proveniência) | **negativo**: ~9% com colapso de predição |
| 015 | completa (13 configs) | **inconclusivo/marginal**: margem não piora; homeostase sem benefício demonstrado; leak sem efeito |
| 016 | completa | **trade-off medido**: −83pp de acurácia por 3.98× menos bytes de inferência |
| 017 | completa (ferramenta) | n/a (não é experimento) |

## 5. Destino formal de 008/009 (sem reescrita histórica)

Os arquivos `.tasks/008-b2-b3-variants.md` e `.tasks/009-comparison.md` foram **preservados byte a byte** (critérios seguem `[ ]`, diretórios `results/exp-008-*/` e `results/exp-009-comparison/` seguem **vazios** — estado verificado, não falha de listagem).

- **[008] → substituída.** Treinadores B2/B3 independentes não serão implementados na forma original (treino por amostra com 6 grades ad hoc). Destino:
  - a *máscara* Top-K por camada/rede é unificada em **021** (passo de treino único, sem duplicar a matemática);
  - a *comparação entre variantes* só volta como experimento em **025**, condicionada a 024 demonstrar aprendizado (diferença operacional definida);
  - o *contrato* que as variantes devem respeitar é versionado em **020**.
- **[009] → substituída.** A comparação qualitativa A-vs-todas-as-B pressupõe variantes que aprendam, o que não existe. Destino:
  - comparação de eficiência refeita com protocolo corrigido em **026**;
  - consolidação narrativa (incluindo negativos e adiamentos) em **027**.
- Referência cruzada registrada também em `.tasks/INDEX.md`.

## 6. Divergências README × arquivos reais (corrigidas no README)

| README antes (f5d3a5a) | Estado real verificado |
|---|---|
| Árvore lista `src/ternary_b2.py`, `src/ternary_b3.py` | **não existem**; só há primitivas Top-K em `src/counter_rules.py` + máscaras no tracer (017) |
| Árvore lista `src/compare.py` | **não existe**; há `src/ablation.py` (outro papel) e `experiments/run_benchmark.py` (016) |
| Árvore lista `experiments/ ...` genérico | há exatamente 4 scripts: `run_small_subset.py`, `run_mnist_full.py`, `run_ablations.py`, `run_benchmark.py` |
| Tabela de tasks termina em 017 | acrescentadas linha 018 (esta revisão) e ponte para a fila 019–027 |
| Proposta científica misturada com estado do repo | proposta movida para `docs/ideia.md`; README descreve estado + limitações |
| Sem link para revisão/auditoria | link para este relatório + `.tasks/INDEX.md` + `results/INDEX.md` |

## 7. Itens não rastreados encontrados (não incluídos na entrega)

- `results/accuracy_curve_default.png` — artefato órfão pré-Astra, sem proveniência; listado em `results/INDEX.md` como órfão, não como evidência.
- `results/test-002/` — idem.
- `results/exp-017-playground/runner-b1/` — saída de runner interativo não versionada; os artefatos versionados de 017 são `smoke-runner-b1/` e `smoke-comparison/`.
- `.opencode/` — planejamento da próxima rodada (fonte desta task); verificado `git check-ignore` (não ignorado), mas seu versionamento é decisão fora do escopo de 018.

## 8. Limites desta revisão

- Análise **estática**: nenhum teste foi executado, nenhum número foi regenerado. Comandos de verificação propostos para 019: `python -m src.ternary_b1`, `python -m src.counter_rules`, `python -m src.reset_policies`, `python -m src.playground.tracer`.
- Nenhum código de `src/` foi migrado ou unificado (responsabilidade de **021**).
- Nenhum teste novo foi atribuído a esta revisão.
- Próximo passo: **019** (suíte de testes + ambiente reproduzível), depois 020/021.
