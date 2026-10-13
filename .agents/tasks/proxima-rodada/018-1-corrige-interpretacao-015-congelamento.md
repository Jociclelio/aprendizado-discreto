# Task 018-1: corrige interpretação 015 congelamento

Estado: resolvida por subtasks (018-2 fechada 10/10; nota 8/10 e decisão corrigir originais preservadas)
Depende de: 018
Origem: 018 (correção; preserva entrega 46ade96)

## Prompt
Corrigir apenas a generalização falsa do congelamento universal em 015/B+ sem reescrever evidências nem histórico. Preservar a entrega 018 (46ade96). Editar somente `docs/reports/revisao-f5d3a5a.md`, `results/INDEX.md`, `.tasks/INDEX.md` e `README.md` para distinguir: base, 6× leak e homeo_0.25 congelam `[*,0,0]` após epoch 1; variantes com margem/combinações continuam transitando (ex.: margin_1/2/5 e margin_2+homeo) mas seguem ~9–10% sem sucesso científico. Não migrar código, não reexecutar experimentos, não alterar `.tasks/007–017` históricos.

## Contexto
- `.agents/tasks/proxima-rodada/018-organiza-documentacao-e-fila.md` — task mãe, decisão corrigir, nota 7/10, alvo base f5d3a5a/entrega 46ade96/registro 70f5a29.
- `results/exp-015-ablacoes/metrics.json` — oráculo: `base [19896,0,0]`, leak `~[19915,0,0]`, `homeo_0.25 [57829,0,0]` vs `margin_1 [168146,134101,131680]`, `margin_2 [163126,125900,125950]`, `margin_5 [186297,147984,147994]`, `margin_2+homeo [193387,74462,165024]` (listas = [epoch1,epoch2,epoch3]).
- `docs/reports/revisao-f5d3a5a.md:64`, `results/INDEX.md:20,29`, `.tasks/INDEX.md:24`, `README.md:9-10` — frases universais falsas a corrigir no SHA 46ade96.
- `AGENTS.md` — convenção de fila e commits.

## Corretude
- [x] C1 — `revisao-f5d3a5a.md:64` não afirma congelamento “em todas as configs”; distingue quem congela de margem/combinações que transitam, mantendo conclusão marginal sem inferir sucesso.
- [x] C2 — `results/INDEX.md:20,29` e `.tasks/INDEX.md:24` não propagam congelamento como padrão 010/011/013/014/015/016; precisão verificável por `git show <entrega>:results/exp-015-ablacoes/metrics.json`.
- [x] C3 — `README.md:9-10` não generaliza B1/B+ como congelamento após 1ª epoch; árvore/comandos auditados em 018 intactos.
- [x] C4 — Nenhum `.tasks/007–017` reescrito; `git diff <base-018-1>..<entrega-018-1> -- .tasks/` vazio salvo `.tasks/INDEX.md`; links locais verificados.

## Execução
- Modelo: opencode/muse-spark-1.3-contributor-free
- Base: 0ef98fab2f3840380bc1b11d206664533cc52f4e
- Entrega: ecb435091eae6988db10d9a6bcc4664479b89357
- Arquivos da entrega: docs/reports/revisao-f5d3a5a.md, results/INDEX.md, .tasks/INDEX.md, README.md
- Evidências: análise estática; oráculo `results/exp-015-ablacoes/metrics.json` (base [19896,0,0], leak ~[19915,0,0] ×6, homeo_0.25 [57829,0,0] vs margin_1 [168146,134101,131680], margin_2 [163126,125900,125950], margin_5 [186297,147984,147994], margin_2+homeo [193387,74462,165024]); `git diff -- .tasks/` só INDEX.md; sem código novo, sem reexecução

## Notas

### Dificuldade esperada
2/10 — Edição documental pontual em 4 arquivos com oráculo JSON já identificado; sem código nem ambiente.

### Dificuldade sentida
2/10 — Edição documental em 4 arquivos com oráculo JSON já identificado; verificação por grep/python do metrics.json e diff restrito a `.tasks/INDEX.md`.

### Revisão
- Orquestrador: task-master (Muse Spark, opencode/muse-spark-1.3-contributor-free)
- Alvo: base 0ef98fab2f3840380bc1b11d206664533cc52f4e, entrega ecb435091eae6988db10d9a6bcc4664479b89357, registro 567b96f381a6e45a26d401f9219e841674786b7c (árvore conferida; entrega isolada em 4 arquivos documentais, sem código; HEAD = registro, só untracked alheios)
- Avaliações: task-reviewer-muse (muse-spark-1.3): 8/10, C1 ok, C2 ok, C3 parcial, C4 ok; task-reviewer-nemotron (nemotron-3-ultra): 10/10, C1–C4 ok; task-reviewer-terra (gpt-5.6-terra): 7/10, C1 ok, C2 ok, C3 parcial, C4 ok. Sem falha de revisor, sem substituição.
- Nota final: 8/10 — correção exata ao oráculo nos 4 pontos literais com escopo respeitado e 46ade96 preservado, mas README:112 mantém generalização universal que contradiz a correção e impede fechar; não é média das três notas (nemotron 10 superestima por não ver o residual).
- Parecer: C1 comprovado (unânime: revisao-f5d3a5a.md:64 distingue base/6× leak/homeo_0.25 que congelam de margem/combinações que transitam, ex. margin_1 [168146,134101,131680], sem inferir sucesso). C2 comprovado (unânime: results/INDEX.md:20,29 e .tasks/INDEX.md:24 qualificados como só base/leak/homeo_0.25 vs margem que transita, verificáveis por git show ecb4350:results/exp-015-ablacoes/metrics.json). C4 comprovado (unânime: diff base..entrega só os 4 documentais; .tasks/007–017 intactos salvo INDEX.md; links locais válidos). Divergência em C3: muse+terra apontam residual falso em README:112 (“B1/B+ seguem ~9–10% com congelamento após a 1ª epoch (013/014/015/016…)”, herdado de 46ade96 e fora do diff) que generaliza 015 como congelamento universal contra o oráculo (margin_1/2/5 e margin_2+homeo transitam nas 3 epochs) e contra o próprio README:9-10 corrigido; C3 literal (:9-10 + árvore intacta) comprovado, C3 nível-arquivo parcial. Nemotron 10 não verifica esse residual.
- Decisão: corrigir
- Reversão: não se aplica — correção preserva entrega ecb4350
- Próximo passo: 018-2-corrige-readme-112-congelamento fechada (10/10, sem pendências); libera 019

Plano: [plan.md](../../plans/proxima-rodada/plan.md)
