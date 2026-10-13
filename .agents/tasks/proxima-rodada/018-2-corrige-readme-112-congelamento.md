# Task 018-2: corrige residual README:112 congelamento

Estado: fechada
Depende de: 018-1
Origem: 018-1 (correção; preserva entregas 46ade96 e ecb4350)

## Prompt
Corrigir apenas a generalização falsa residual em `README.md:112` (“B1/B+ seguem ~9–10% com congelamento após a 1ª epoch (013/014/015/016…)”), herdada de 46ade96 e fora do diff 0ef98fa..ecb4350, que contradiz o `README.md:9-10` já corrigido em 018-1. Reescrever só essa frase no mesmo padrão de `:9-10`: congelam após a 1ª epoch só base, as 6 leak e `homeo_0.25`; variantes 015 com margem/combinações transitam nas 3 epochs mas seguem ~9–10%, sem sucesso científico. Preservar as entregas 018 (46ade96) e 018-1 (ecb4350). Não migrar código, não reexecutar experimentos, não alterar `.tasks/007–017` históricos nem `docs/reports/revisao-f5d3a5a.md`, `results/INDEX.md`, `.tasks/INDEX.md`.

## Contexto
- `.agents/tasks/proxima-rodada/018-1-corrige-interpretacao-015-congelamento.md` — task mãe, decisão corrigir, nota 8/10, alvo base 0ef98fa/entrega ecb4350/registro 567b96f + revisão 68aab55.
- `README.md:9-10` (já corrigido em ecb4350) — padrão de redação a replicar em `:112`.
- `README.md:112` — frase residual falsa a corrigir.
- `results/exp-015-ablacoes/metrics.json` — oráculo: `base [19896,0,0]`, leak `~[19915,0,0]` (×6), `homeo_0.25 [57829,0,0]` vs `margin_1 [168146,134101,131680]`, `margin_2 [163126,125900,125950]`, `margin_5 [186297,147984,147994]`, `margin_2+homeo [193387,74462,165024]` (listas = [epoch1,epoch2,epoch3]).
- `AGENTS.md` — convenção de fila e commits.

## Corretude
- [x] C1 — `README.md:112` não afirma congelamento universal de 015/B+; distingue quem congela (base/6× leak/`homeo_0.25`) de margem/combinações que transitam, com os vetores do oráculo, mantendo ~9–10% sem inferir sucesso e consistente com `README.md:9-10`.
- [x] C2 — Diff da entrega restrito a `README.md:112` (uma frase); `docs/reports/revisao-f5d3a5a.md`, `results/INDEX.md`, `.tasks/INDEX.md`, `src/`, `experiments/`, `results/exp-*/` intactos; `git diff <base-018-2>..<entrega-018-2> --stat` comprova.
- [x] C3 — Nenhum `.tasks/007–017` reescrito; links locais verificados.

## Execução
- Modelo: —
- Base: ecb435091eae6988db10d9a6bcc4664479b89357
- Entrega: c01ec02
- Arquivos da entrega: README.md
- Evidências: análise estática; oráculo `results/exp-015-ablacoes/metrics.json` (base [19896,0,0], leak ~[19915,0,0] ×6, homeo_0.25 [57829,0,0] vs margin_1 [168146,134101,131680], margin_2 [163126,125900,125950], margin_5 [186297,147984,147994], margin_2+homeo [193387,74462,165024]); `git diff ecb4350..c01ec02 --stat` = 1 file changed, 1 insertion(+), 1 deletion(-); sem código novo, sem reexecução

## Notas

### Dificuldade esperada
1/10 — Edição de uma frase em um arquivo com oráculo e padrão de redação já fixados em 018-1; sem código nem ambiente.

### Dificuldade sentida
1/10 — Edição de uma frase em um arquivo com oráculo e padrão de redação já fixados em 018-1; sem código nem ambiente.

### Revisão
- Orquestrador: task-master (Muse Spark, opencode/muse-spark-1.3-contributor-free)
- Alvo: base ecb435091eae6988db10d9a6bcc4664479b89357, entrega c01ec02e4f44b5081893ce136458e60071d421cf, registro 6da0ae6b89407fc02939cd76ee25e34c228040bc (árvore conferida; entrega isolada em 1 arquivo documental, sem código; HEAD = registro, só untracked alheios)
- Avaliações: task-reviewer-muse (muse-spark-1.3): 10/10, C1 ok, C2 ok, C3 ok; task-reviewer-nemotron (nemotron-3-ultra): 10/10, C1–C3 ok; task-reviewer-terra (gpt-5.6-terra): 10/10, C1–C3 ok. Sem falha de revisor, sem substituição.
- Nota final: 10/10 — residual universal de README:112 eliminado com redação exata ao oráculo (base/leak/homeo_0.25 congelam, margem/combinações transitam, ~9–10% sem sucesso), entrega isolada em 1 commit de 1 arquivo e escopo respeitado (46ade96 e ecb4350 preservados); não é média automática, é convergência unânime sem ressalvas.
- Parecer: C1 comprovado (unânime: c01ec02:README.md:112 distingue base [19896,0,0]/leak ~[19915,0,0]/homeo_0.25 [57829,0,0] que congelam de margem/combinações que transitam, ex. margin_1 [168146,134101,131680], mantendo ~9–10% sem inferir sucesso e consistente com README.md:9-10). C2 comprovado (unânime: git show c01ec02 --stat = 1 file changed, 1 insertion(+), 1 deletion(-), só README.md; revisao-f5d3a5a.md, results/INDEX.md, .tasks/INDEX.md, src/, experiments/, results/exp-*/ intactos; intervalo ecb4350..c01ec02 contém os documentais intermediários 567b96f/68aab55/961ccc6, fora da entrega isolada). C3 comprovado (unânime: nenhum .tasks/007–017 tocado; links locais válidos). Sem divergências e sem Corretude pendente.
- Decisão: fechar
- Reversão: não se aplica — correção preserva entregas 46ade96 e ecb4350
- Próximo passo: 019-estrutura-testes-e-ambiente (liberada; família 018 resolvida por subtasks)

Plano: [plan.md](../../plans/proxima-rodada/plan.md)
