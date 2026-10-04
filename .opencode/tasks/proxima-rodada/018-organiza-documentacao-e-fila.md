# Task 018: organiza documentação e reconcilia a fila

Estado: resolvida por subtasks (018-1 8/10 + 018-2 fechada 10/10; nota 7/10 e decisão corrigir originais preservadas)
Depende de: nenhuma
Origem: nenhuma

## Prompt
Reconciliar documentação e fila histórica sem implementar código novo nem reescrever evidências. Entregar: relatório de revisão em `docs/reports/revisao-f5d3a5a.md`, separação proposta científica (`docs/ideia.md`) x README fiel ao estado real, referência preservada ao relatório Astra, auditoria dos critérios de 007/013/014/015/016/017 (evidência x pendente x resultado negativo), destino formal de 008/009, atualização de `.tasks/INDEX.md` e criação de `results/INDEX.md` com separação entre status de execução e resultado científico. Não migrar código (responsabilidade de 021) nem atribuir novos testes à revisão (análise estática).

## Contexto
- `astra-relatório-commit-6f47914.md` — defeitos e correções que motivam a reconciliação.
- `.tasks/007-b1-initial.md`, `.tasks/008-b2-b3-variants.md`, `.tasks/009-comparison.md` — fila histórica com 008/009 pendentes.
- `.tasks/012-repara-b1-contrato-matematico.md`, `.tasks/013-prova-aprendizado-subconjunto.md`, `.tasks/014-ensaio-mnist-auditable.md`, `.tasks/015-ablacoes-margin-homeostasis-leak.md`, `.tasks/016-benchmark-eficiencia.md`, `.tasks/017-playground-viz.md` — critérios a auditar.
- `README.md` — mistura arquivos reais e planejados a corrigir.
- `results/exp-012-b1-reparo/`, `results/exp-013-small-subset/`, `results/exp-014-mnist-full/`, `results/exp-015-ablacoes/`, `results/exp-016-benchmark/` — evidências a referenciar por CSV/JSON/relatório.
- `AGENTS.md` — convenção de fila e commits a respeitar.

## Corretude
- [ ] C1 — Relatório `docs/reports/revisao-f5d3a5a.md` existe, cita fontes por arquivo, e está linkado pelo README.
- [ ] C2 — Árvore e comandos do README conferem com arquivos reais; planos estão identificados como tal.
- [ ] C3 — Tasks 008/009 têm destino explícito (substituição/replanejamento com referência a 020/021/025/027), sem falsa declaração de execução; textos históricos preservados.
- [ ] C4 — Critérios de 007/013/014/015/016/017 sem evidência permanecem desmarcados ou recebem correção auditável; resultado negativo de 013 segue explícito.
- [ ] C5 — `.tasks/INDEX.md` e `results/INDEX.md` permitem localizar task, experimento, configuração, evidências, limitações e status de interpretação; links locais verificados.

## Execução
- Modelo: opencode/muse-spark-1.3-contributor-free
- Base: f5d3a5ada972343fc9ea5c96a9b5b2beeae62b89
- Entrega: 46ade96604d0879cecd02971c810847652622218 (docs/reports/revisao-f5d3a5a.md, docs/ideia.md, README.md, .tasks/INDEX.md, results/INDEX.md)
- Arquivos da entrega: docs/reports/revisao-f5d3a5a.md, docs/ideia.md, README.md (fiel ao estado real), .tasks/INDEX.md, results/INDEX.md
- Evidências: análise estática (sem reexecução); 008/009 preservados com `[ ]` e dirs vazios verificados; 007 sem `results/exp-007-*/`; 013 negativo explícito mantido; 014/015/016 lidos via metrics.json/curves/provenances; 017 via tracer-bit-identity.log + smoke.log

## Notas

### Dificuldade esperada
3/10 — Auditoria documental e escrita, sem código nem ambiente; trabalho é cruzamento de fontes e links.

### Dificuldade sentida
4/10 — Cruzamento de 6 tasks × métricas versionadas sem reexecutar nada; o trabalho foi decidir o que conta como evidência (execução × resultado) sem reescrever histórico.

### Revisão
- Orquestrador: task-master (Muse Spark, opencode/muse-spark-1.3-contributor-free)
- Alvo: base f5d3a5a, entrega 46ade96, registro 70f5a29 (árvore conferida; entrega isolada em 5 arquivos documentais, sem código)
- Avaliações: muse-spark-1.3 (task-reviewer-muse): 9/10, depois admite parcial em C2/C4/C5; nemotron-3-ultra-free (task-reviewer-nemotron): 10/10; gpt-5.6-terra (task-reviewer-terra): 7/10. Esclarecimentos retomados nas mesmas sessões; sem falha de revisor, sem substituição.
- Nota final: 7/10 — reconciliação fiel e rastreável com C1/C3 comprovados, mas generalização falsa do congelamento universal em 015/B+ deixa C2/C4/C5 parciais e exige correção delimitada; não é média das três notas.
- Parecer: C1 comprovado (muse+nemotron+terra: revisao-f5d3a5a.md existe, §2 cita fontes, README linka). C3 comprovado (unânime: 008/009 preservados com `[ ]`, diff vazio nas históricas, destinos 008→020/021/025 e 009→026/027). Divergência essencial em 015: terra apontou transições não nulas em epochs 2–3 para margin/combinações vs. universal `[~19900,0,0]`; muse confirmou em esclarecimento que só leu base/leak (`base [19896,0,0]`) e generalizou indevidamente, verificando depois no SHA da entrega `margin_1 [168146,134101,131680]`, `margin_2 [163126,125900,125950]`, `margin_5 [186297,147984,147994]`, `margin_2+homeo [193387,74462,165024]` — logo são falsas como universais `revisao-f5d3a5a.md:64`, `results/INDEX.md:20,29`, `.tasks/INDEX.md:24`, `README.md:9-10`; núcleo marginal mantido (margin_2 ~0.10 vs base ~0.09). Nemotron 10 superestima por não verificar as 13 configs. Ressalvas menores (muse): `exp-007` dito “não existe” quando é dir vazio não-rastreado como 008/009; link `.opencode/plans/...` untracked, só em disco no SHA da entrega.
- Decisão: corrigir
- Reversão: não se aplica — correção preserva entrega 46ade96
- Próximo passo: família resolvida (018-1 8/10 + 018-2 fechada 10/10); libera 019

Plano: [plan.md](../../plans/proxima-rodada/plan.md)
