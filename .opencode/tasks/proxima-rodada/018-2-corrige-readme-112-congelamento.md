# Task 018-2: corrige residual README:112 congelamento

Estado: pendente
Depende de: 018-1
Origem: 018-1 (correção; preserva entregas 46ade96 e ecb4350)

## Prompt
Corrigir apenas a generalização falsa residual em `README.md:112` (“B1/B+ seguem ~9–10% com congelamento após a 1ª epoch (013/014/015/016…)”), herdada de 46ade96 e fora do diff 0ef98fa..ecb4350, que contradiz o `README.md:9-10` já corrigido em 018-1. Reescrever só essa frase no mesmo padrão de `:9-10`: congelam após a 1ª epoch só base, as 6 leak e `homeo_0.25`; variantes 015 com margem/combinações transitam nas 3 epochs mas seguem ~9–10%, sem sucesso científico. Preservar as entregas 018 (46ade96) e 018-1 (ecb4350). Não migrar código, não reexecutar experimentos, não alterar `.tasks/007–017` históricos nem `docs/reports/revisao-f5d3a5a.md`, `results/INDEX.md`, `.tasks/INDEX.md`.

## Contexto
- `.opencode/tasks/proxima-rodada/018-1-corrige-interpretacao-015-congelamento.md` — task mãe, decisão corrigir, nota 8/10, alvo base 0ef98fa/entrega ecb4350/registro 567b96f + revisão 68aab55.
- `README.md:9-10` (já corrigido em ecb4350) — padrão de redação a replicar em `:112`.
- `README.md:112` — frase residual falsa a corrigir.
- `results/exp-015-ablacoes/metrics.json` — oráculo: `base [19896,0,0]`, leak `~[19915,0,0]` (×6), `homeo_0.25 [57829,0,0]` vs `margin_1 [168146,134101,131680]`, `margin_2 [163126,125900,125950]`, `margin_5 [186297,147984,147994]`, `margin_2+homeo [193387,74462,165024]` (listas = [epoch1,epoch2,epoch3]).
- `AGENTS.md` — convenção de fila e commits.

## Corretude
- [ ] C1 — `README.md:112` não afirma congelamento universal de 015/B+; distingue quem congela (base/6× leak/`homeo_0.25`) de margem/combinações que transitam, com os vetores do oráculo, mantendo ~9–10% sem inferir sucesso e consistente com `README.md:9-10`.
- [ ] C2 — Diff da entrega restrito a `README.md:112` (uma frase); `docs/reports/revisao-f5d3a5a.md`, `results/INDEX.md`, `.tasks/INDEX.md`, `src/`, `experiments/`, `results/exp-*/` intactos; `git diff <base-018-2>..<entrega-018-2> --stat` comprova.
- [ ] C3 — Nenhum `.tasks/007–017` reescrito; links locais verificados.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
1/10 — Edição de uma frase em um arquivo com oráculo e padrão de redação já fixados em 018-1; sem código nem ambiente.

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
