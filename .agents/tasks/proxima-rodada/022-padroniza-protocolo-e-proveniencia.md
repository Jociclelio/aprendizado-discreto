# Task 022: padroniza protocolo experimental e proveniência

Estado: pendente
Depende de: 021
Origem: nenhuma

## Prompt
Tornar cada execução identificável e reproduzível a partir de configuração, dados, splits, código e ambiente. Entregar `docs/protocolo-experimental.md`, configs versionadas em `configs/`, loader/validador de config (rejeita campos desconhecidos, registra config final pós-override CLI), splits e runner compartilhados, sementes separadas (split x init x ordem) com split fixo para comparações pareadas, persistência de índices reais, hashes, tamanhos efetivos, comando, commit, dirty-state, versões, dispositivo/threads, orçamento e tempos, diretórios `results/exp-NNN-nome/run-ID/` sem sobrescrita, histórico completo e resumo por seed. Padronizar unidades (acurácia `[0,1]`, segundos, bytes), preprocessing (`normalize=True` = pixels `[0,1]`), média de perda por amostra real e métricas anti-colapso. Restauração/retomada completa fica para 023.

## Contexto
- `experiments/run_mnist_full.py`, `experiments/run_small_subset.py`, `experiments/run_ablations.py`, `experiments/run_benchmark.py` — duplicação de splits/loops a unificar.
- `results/exp-014-mnist-full/` — ensaio com índice registrado como hash e `test_size` ignorado a corrigir como precedente.
- `src/baseline.py` — preprocessing da baseline a identificar explicitamente.
- `.agents/tasks/proxima-rodada/021-unifica-passo-de-treinamento.md` — núcleo e runner que esta task integra.

## Corretude
- [ ] C1 — Configs canônicas e protocolo versionados, com comandos reais de reprodução documentados.
- [ ] C2 — Execução curta reproduz splits e métricas determinísticas no ambiente registrado; alterar tamanhos configurados altera tamanhos efetivos; splits não vazios, sem sobreposição treino/validação, com `test_size` aplicado.
- [ ] C3 — Testes validam splits, parâmetros inválidos e lote final incompleto via `python -m pytest`.
- [ ] C4 — Manifesto identifica conteúdo executado mesmo com mudanças locais (hash inclui não-rastreados/modificados); nova execução não sobrescreve histórico; smoke com orçamento pequeno em `results/exp-022-protocolo/`.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
7/10 — Cruzar configs, splits, hashes, proveniência e runner compartilhado sem quebrar experimentos históricos exige validação cuidadosa.

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
