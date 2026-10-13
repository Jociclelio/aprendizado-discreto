# Task 024: diagnostica aprendizado discreto em conjunto pequeno

Estado: pendente
Depende de: 023
Origem: nenhuma

## Prompt
Demonstrar aprendizado discriminativo auditável ou localizar o mecanismo que impede o ajuste, antes de qualquer ampliação. Fixar antes de executar: subconjunto MNIST 32–128 amostras com índices/classes/orçamento, ganho mínimo em pontos percentuais e regra de agregação entre ≥3 sementes. Executar e identificar separadamente: rede inicial + majoritário (com predições/confusão/distribuição), legado 012 via núcleo/protocolo atuais, controle readout treinável sobre ocultas fixas (otimizador declarado, ocultas congeladas), controle ternário com latentes + STE descrito (separado do método principal sem latentes), e rede menor/rasa como diagnóstico (sem misturar com topologia nominal `784→128→64→10`). Rastrear por camada atividade por neurônio, scores, surrogate/delta, incrementos pré/pós-quantização, contadores, seleção e saturação; identificar o primeiro ponto de colapso por seed. Mudança algorítmica fora do contrato exige decisão e replanejamento. Resultado negativo bem instrumentado atende à entrega diagnóstica.

## Contexto
- `results/exp-013-small-subset/` — empate com majoritário e morte do surrogate a explicar.
- `results/exp-014-mnist-full/` — colapso para classe única em 3 sementes.
- `results/exp-015-ablacoes/` — margem mantém transições sem classificação útil.
- `docs/contrato-matematico.md` — contrato de 020 que rege orçamento/seleções (dependência).
- `configs/diagnostics/` — diretório a criar com configs da rodada.
- `.agents/tasks/proxima-rodada/023-corrige-selecao-e-checkpoints.md` — seleção/retomada a usar.

## Corretude
- [ ] C1 — Configs, índices, orçamento, regra de sucesso e ≥3 sementes registrados antes da rodada; saídas em `results/exp-024-diagnostico/run-ID/` conforme 022/023.
- [ ] C2 — Controles e experimento principal executados e identificados separadamente, sem contaminação entre método sem latentes e controle com latentes.
- [ ] C3 — Artefatos permitem verificar transições, diversidade e onde o sinal desaparece, separando surrogate zero x cancelamento de delta x incremento inteiro zero, com cobertura de sinal e evidência descartada pelo arredondamento.
- [ ] C4 — Relatório conclui um caminho pelos critérios prévios (supera inicial + majoritário pelo ganho definido com discriminação; `>10%` isolado não basta) ou resultado negativo com mecanismo de falha e limites; decisão para 025 registrada (expansão permitida ou só diagnóstico); núcleo verificado e comandos de reprodução registrados.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
8/10 — Marco científico da rodada; exige instrumentação por camada/seed, controles limpos e conclusão binária auditável sob risco de colapso persistente.

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
