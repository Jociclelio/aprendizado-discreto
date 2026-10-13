# Plano: próxima rodada — reparo, protocolo e diagnóstico

Estado: aprovado

## Objetivo
Encerrar a divergência entre documentação, código e resultados após a revisão `f5d3a5a` e o relatório Astra (`6f47914`), estabelecendo contrato, núcleo único, protocolo reproduzível e diagnóstico auditável de aprendizado discreto antes de qualquer ampliação em MNIST.

## Abordagem
1. Reconciliar documentação e fila histórica sem reescrever evidências (018).
2. Fixar suíte de testes e ambiente reproduzível como rede de segurança (019).
3. Versionar o contrato algorítmico antes de refatorar (020), depois unificar o passo de treino (021).
4. Padronizar protocolo/proveniência (022), corrigir seleção/checkpoints (023).
5. Executar diagnóstico em conjunto pequeno com controles (024) como marco científico; condicionar ablações (025), benchmark (026) e consolidação (027) ao resultado.
6. Migrar planejamento de `.tasks/018–027` para `.agents/tasks/proxima-rodada/` no contrato `task-writer`; `.tasks/001–017` permanecem como histórico. (Fila movida de `.opencode/` para `.agents/` em [028] — convenção pi, estrutura espelhada preserva links relativos.)

## Escopo
Entregas: tasks 018–027 reescritas no template vigente, executadas uma por vez na ordem numérica.
Limites: nenhuma mudança de algoritmo fora do contrato sem nova task e decisão explícita de replanejamento; nenhuma expansão para MNIST maior sem critério prévio registrado em 024/025.

## Contexto
- `.tasks/018-organiza-documentacao-e-fila.md` — origem migrada para 018.
- `.tasks/019-estrutura-testes-e-ambiente.md` — origem migrada para 019.
- `.tasks/020-versiona-contrato-algoritmico.md` — origem migrada para 020.
- `.tasks/021-unifica-passo-de-treinamento.md` — origem migrada para 021.
- `.tasks/022-padroniza-protocolo-e-proveniencia.md` — origem migrada para 022.
- `.tasks/023-corrige-selecao-e-checkpoints.md` — origem migrada para 023.
- `.tasks/024-diagnostica-aprendizado-discreto.md` — origem migrada para 024, marco científico.
- `.tasks/025-reavalia-ablacoes-e-variantes.md` — origem migrada para 025.
- `.tasks/026-corrige-benchmark-de-eficiencia.md` — origem migrada para 026.
- `.tasks/027-consolida-resultados-e-conclusoes.md` — origem migrada para 027.
- `astra-relatório-commit-6f47914.md` — defeitos P0/P1 que motivaram 012 e a rodada.
- `src/ternary_b1.py`, `src/ablation.py`, `src/playground/tracer.py` — duplicação a unificar em 021.
- `results/exp-012-b1-reparo/`, `results/exp-013-small-subset/`, `results/exp-014-mnist-full/`, `results/exp-015-ablacoes/`, `results/exp-016-benchmark/` — evidências a reconciliar.

## Critérios de conclusão
- Tasks 018–027 executadas em ordem, cada uma com Corretude comprovada e 2 commits (entrega + registro).
- Diagnóstico 024 conclui um dos caminhos (aprendizado demonstrado ou resultado negativo com mecanismo de falha).
- Relatório 027 responde às perguntas da rodada com referências a run-IDs e artefatos, incluindo negativos e adiamentos.

## Questões abertas
Nenhuma para o planejamento. Questões científicas (regra de atualização, generalização, eficiência) serão decididas por 024/025/027 com evidências.

## Tasks
| Ordem | Task | Depende de | Estado |
|---|---|---|---|
| 1 | [018-organiza-documentacao-e-fila.md](../../tasks/proxima-rodada/018-organiza-documentacao-e-fila.md) | nenhuma | resolvida por subtasks (018-1 8/10 + 018-2 fechada 10/10; libera 019) |
| 1.1 | [018-1-corrige-interpretacao-015-congelamento.md](../../tasks/proxima-rodada/018-1-corrige-interpretacao-015-congelamento.md) | 018 | resolvida por subtasks (revisada 8/10; correção 018-2 fechada 10/10; libera 019) |
| 1.2 | [018-2-corrige-readme-112-congelamento.md](../../tasks/proxima-rodada/018-2-corrige-readme-112-congelamento.md) | 018-1 | fechada (revisada 10/10, sem pendências) |
| 2 | [019-estrutura-testes-e-ambiente.md](../../tasks/proxima-rodada/019-estrutura-testes-e-ambiente.md) | 018 família resolvida (018 + 018-1 + 018-2) | aguardando subtasks (revisada 7/10, corrigir) |
| 2.1 | [019-1-corrige-testes-e-lock.md](../../tasks/proxima-rodada/019-1-corrige-testes-e-lock.md) | 019 | pendente |
| 3 | [020-versiona-contrato-algoritmico.md](../../tasks/proxima-rodada/020-versiona-contrato-algoritmico.md) | 019-1 | pendente |
| 4 | [021-unifica-passo-de-treinamento.md](../../tasks/proxima-rodada/021-unifica-passo-de-treinamento.md) | 020 | pendente |
| 5 | [022-padroniza-protocolo-e-proveniencia.md](../../tasks/proxima-rodada/022-padroniza-protocolo-e-proveniencia.md) | 021 | pendente |
| 6 | [023-corrige-selecao-e-checkpoints.md](../../tasks/proxima-rodada/023-corrige-selecao-e-checkpoints.md) | 022 | pendente |
| 7 | [024-diagnostica-aprendizado-discreto.md](../../tasks/proxima-rodada/024-diagnostica-aprendizado-discreto.md) | 023 | pendente |
| 8 | [025-reavalia-ablacoes-e-variantes.md](../../tasks/proxima-rodada/025-reavalia-ablacoes-e-variantes.md) | 024 | pendente |
| 9 | [026-corrige-benchmark-de-eficiencia.md](../../tasks/proxima-rodada/026-corrige-benchmark-de-eficiencia.md) | 025 | pendente |
| 10 | [027-consolida-resultados-e-conclusoes.md](../../tasks/proxima-rodada/027-consolida-resultados-e-conclusoes.md) | 026 (usa evidências de 018–025) | pendente |
