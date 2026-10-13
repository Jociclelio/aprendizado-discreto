# Task 023: corrige seleção por validação e checkpoints

Estado: pendente
Depende de: 022
Origem: nenhuma

## Prompt
Garantir que o modelo testado seja o checkpoint selecionado e que a retomada preserve todo o estado. Entregar seleção uniforme por validação para baseline e ternárias, restore do selecionado antes do teste final, regra de desempate entre épocas e tratamento de execução sem treino válido, formato versionado de checkpoint de inferência (arquitetura, pesos, bias, preprocessing, contrato, config de forward) e de retomada (parâmetros, contadores/otimizador, RNGs, cursor/permutação inclusive mid-batch, agenda leak/homeostase, histórico), vínculo checkpoint-config-métricas pelo mesmo identificador, e integração com o runner 022 com validação de shapes/versões. Retirar alegação de melhor B+ sem seleção reproduzível; manter receitas legadas nomeadas com origem. Resultados históricos 015/016 recebem nota de limitação, sem substituição silenciosa.

## Contexto
- `experiments/run_benchmark.py` — precedente que avalia última época em vez da melhor validação.
- `results/exp-015-ablacoes/`, `results/exp-016-benchmark/` — resultados a anotar com limitação.
- `src/ternary_refined.py` — configuração refinada a reconciliar com a selecionada.
- `.agents/tasks/proxima-rodada/022-padroniza-protocolo-e-proveniencia.md` — runner e manifesto a integrar.
- `.agents/tasks/proxima-rodada/020-versiona-contrato-algoritmico.md` — contrato que os checkpoints devem referenciar.

## Corretude
- [ ] C1 — Caso controlado em que a melhor época não é a última prova restauração correta; métrica de teste aponta para checkpoint e config realmente avaliados.
- [ ] C2 — Scores/predições antes e depois de salvar/carregar inferência são iguais; config incompatível ou arquivo incompleto gera erro informativo.
- [ ] C3 — Treino contínuo x interrompido/retomado produzem estados e métricas iguais no backend determinístico de referência, incluindo configuração com leak; contadores correspondem à etapa do checkpoint, sem associar contadores finais a pesos de época anterior.
- [ ] C4 — Suíte passa via `python -m pytest`; evidências em `results/exp-023-checkpoints/` sem sobrescrever histórico.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
6/10 — Exige formatos versionados distintos (inferência x retomada) e prova de equivalência contínua/retomada incluindo RNGs e cursor.

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
