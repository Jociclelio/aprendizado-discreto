# Task 018-1: corrige interpretação 015 congelamento

Estado: pendente
Depende de: 018
Origem: 018 (correção; preserva entrega 46ade96)

## Prompt
Corrigir apenas a generalização falsa do congelamento universal em 015/B+ sem reescrever evidências nem histórico. Preservar a entrega 018 (46ade96). Editar somente `docs/reports/revisao-f5d3a5a.md`, `results/INDEX.md`, `.tasks/INDEX.md` e `README.md` para distinguir: base, 6× leak e homeo_0.25 congelam `[*,0,0]` após epoch 1; variantes com margem/combinações continuam transitando (ex.: margin_1/2/5 e margin_2+homeo) mas seguem ~9–10% sem sucesso científico. Não migrar código, não reexecutar experimentos, não alterar `.tasks/007–017` históricos.

## Contexto
- `.opencode/tasks/proxima-rodada/018-organiza-documentacao-e-fila.md` — task mãe, decisão corrigir, nota 7/10, alvo base f5d3a5a/entrega 46ade96/registro 70f5a29.
- `results/exp-015-ablacoes/metrics.json` — oráculo: `base [19896,0,0]`, leak `~[19915,0,0]`, `homeo_0.25 [57829,0,0]` vs `margin_1 [168146,134101,131680]`, `margin_2 [163126,125900,125950]`, `margin_5 [186297,147984,147994]`, `margin_2+homeo [193387,74462,165024]` (listas = [epoch1,epoch2,epoch3]).
- `docs/reports/revisao-f5d3a5a.md:64`, `results/INDEX.md:20,29`, `.tasks/INDEX.md:24`, `README.md:9-10` — frases universais falsas a corrigir no SHA 46ade96.
- `AGENTS.md` — convenção de fila e commits.

## Corretude
- [ ] C1 — `revisao-f5d3a5a.md:64` não afirma congelamento “em todas as configs”; distingue quem congela de margem/combinações que transitam, mantendo conclusão marginal sem inferir sucesso.
- [ ] C2 — `results/INDEX.md:20,29` e `.tasks/INDEX.md:24` não propagam congelamento como padrão 010/011/013/014/015/016; precisão verificável por `git show <entrega>:results/exp-015-ablacoes/metrics.json`.
- [ ] C3 — `README.md:9-10` não generaliza B1/B+ como congelamento após 1ª epoch; árvore/comandos auditados em 018 intactos.
- [ ] C4 — Nenhum `.tasks/007–017` reescrito; `git diff <base-018-1>..<entrega-018-1> -- .tasks/` vazio salvo `.tasks/INDEX.md`; links locais verificados.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
2/10 — Edição documental pontual em 4 arquivos com oráculo JSON já identificado; sem código nem ambiente.

### Dificuldade sentida
_/10 —

### Revisão
- Orquestrador: —
- Alvo: —
- Avaliações: —
- Nota final: _/10
- Parecer: —
- Decisão: pendente
- Reversão: não se aplica
- Próximo passo: —

Plano: [plan.md](../../plans/proxima-rodada/plan.md)
