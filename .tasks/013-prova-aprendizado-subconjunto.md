# [013] Prova de capacidade de ajuste em subconjunto pequeno

## Objetivo

Demonstrar que o B1 corrigido (012) **aprende de fato** num subconjunto pequeno de MNIST, com métricas auditáveis, antes de qualquer treino longo ou novo mecanismo.

## Contexto

O relatório (seção 11) define o próximo marco: "um B1 matematicamente consistente que aprenda um conjunto pequeno, com scores, contadores e transições auditáveis". Treinar 20 epochs em 60k amostras com vários mecanismos novos dificulta localizar onde o sinal morre. Um subconjunto pequeno e fixo torna o experimento determinístico, barato e diagnóstico.

Esta task é a **Etapa 2** da sequência recomendada (relatório, seção 9).

## Escopo

- [x] Criar `experiments/run_small_subset.py` (script versionado, sem comandos ad hoc) que:
  - fixa um subconjunto de **32–128 amostras** por uma semente e uma lista de índices salvos;
  - mede acurácia **antes** do treino (rede inicial) no próprio subconjunto;
  - treina B1 corrigido por um orçamento pequeno e fixo;
  - mede acurácia **depois** no mesmo subconjunto.
- [x] Registrar a cada passo/epoch:
  - scores de saída (`z_out`) por amostra;
  - distribuição de classes previstas (contagem por classe) e matriz de confusão;
  - taxa de atividade (fração de disparos) por camada;
  - surrogate (fake gradient), delta e incremento **antes e depois** da quantização;
  - contadores separados: **selecionados**, **tentativas**, **bloqueios por clipping**, **transições efetivas**.
- [x] Comparar contra um **preditor majoritário** do subconjunto (baseline fraca) e contra a rede inicial.
- [x] Se não houver melhora substancial, registrar o **primeiro ponto** (camada/etapa) em que o sinal ou a diversidade desaparece (ex.: surrogate zero, delta cancelado, incremento truncado a 0).
- [x] Salvar tudo em `results/exp-013-small-subset/`.

## Especificações técnicas

- **Subconjunto**: N ∈ {32, 64, 128} amostras; índices e semente gravados em JSON.
- **Arquitetura**: `784 → 128 → 64 → 10` (invariante).
- **Orçamento**: fixo e pequeno (ex.: 1–3 épocas sobre o subconjunto, ou X passos); registrado.
- **Métricas**: acurácia inicial/final, Δacurácia, contagem de transições efetivas, taxa de atividade por camada, fração de surrogate/delta não nulo.
- **Reprodutibilidade**: `python experiments/run_small_subset.py --seed 42 --n 64` reproduz o mesmo experimento.
- **Dependências**: usa a API unificada de `forward_with_scores` e `deltas`/`selection_mask` de 012.

## Critérios de aceitação

- [x] Script implementado em `experiments/run_small_subset.py` e versionado.
- [x] Executa sem erro: `python experiments/run_small_subset.py`.
- [ ] Δacurácia sobre o subconjunto **positiva e acima do preditor majoritário** e da rede inicial.
- [x] Métricas de diagnóstico (scores, confusão, atividade, surrogate/delta, transições) salvas em `results/exp-013-small-subset/`.
- [x] Se falhar: relatório do ponto de falha escrito nesta task (o que se quer investigar em seguida).
- [x] Commit `[013]` com mensagem descritiva (`6f0e49d`).

## Notas de implementação

- O objetivo é **diagnóstico + sinal**, não acurácia final: um ganho claro no subconjunto já valida o mecanismo.
- Fixar o subconjunto evita a ilusão de "aprendizagem" por sorte de split.
- Se o surrogate for zero em quase toda a rede, o problema é de **escala/raio**; se o delta se cancela, é de **sinal**; se o incremento é truncado, é de **quantização** — cada um aponta para uma correção distinta.
- Não introduzir margem/homeostase aqui; isso é 015.

## Dependências

- Bloqueado por: `[012]`
- Bloqueia: `[014]`

## Registro de execução (2026-09-17) — ramo FALHA (ponto de falha localizado)

- Script `experiments/run_small_subset.py` criado; reproducível
  (`--seed 42 --n 64`; hash de `metrics.json` sem timestamp idêntico
  entre execuções). Orçamento fixo: 3 epochs × 64 = 192 passos.
- Resultado canônico (n=64, seed=42): acc **0.1250 → 0.1562**
  (Δ=+0.0312) vs majoritário **0.1562** (classe 8). Δ positivo mas final
  **empata** o majoritário — ramo SUCESSO não atingido; falha reportada
  conforme previsto na task.
- Grade de robustez (6 combos scale/scale_factor/levels × seeds 42/7/123,
  n=32/64/128): nenhum supera o majoritário; melhor caso empata.
- **Primeiro ponto de falha** (trilha a cada 8 passos):
  1. passo 8 — colapso de diversidade: todas as 64 preds → classe 8
     (a rede aprende só o prior majoritário);
  2. passo 32 — morte do surrogate de **saída**: `q_fake(z_out)==0` em
     todo o subconjunto (|z_out| médio 2.4 → 20.6 >> raio 8) ⇒
     `delta_out = sign(error)*0 = 0` ⇒ nada se propaga (deltas h2/h1 zero);
  3. epochs 2–3 congeladas: 0 selecionados, 0 transições (transições por
     epoch: [22263, 0, 0]; bloqueios: [9938, 0, 0]).
  Na taxonomia da task: **surrogate zero ⇒ problema de escala/raio**
  (não de sinal nem de quantização — antes do colapso, surrogate 93–99%
  não-nulo e deltas densos em todas as camadas).
- Variante seed=7/n=32 confirma o padrão: colapso no passo 8
  (→ classe 9), surrogate de saída morto no passo 16, 2 epochs congeladas.
- Artefatos em `results/exp-013-small-subset/`: `subset.json` (índices),
  `metrics.json` (inclui `diag_trail` com 25 diagnósticos), `step_log.json`
  (192 passos), `scores_before/after.npy`, `weights.npz`, `REPORT.md`.
- O que investigar em seguida ([015], sem novos mecanismos aqui):
  como manter |z| dentro do cone do surrogate (margem/homeostase) e
  preservar diversidade de preds; updates de ±1/amostra (~500 pesos por
  amostra, ~30% da rede no epoch 1) são agressivos demais sem estabilização.
