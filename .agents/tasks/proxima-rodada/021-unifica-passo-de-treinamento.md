# Task 021: unifica passo de treinamento e organiza módulos

Estado: pendente
Depende de: 020
Origem: nenhuma

## Prompt
Implementar núcleo único de treino/tracing compartilhado por CLI, ablações e playground, organizado por responsabilidade. Entregar `src/core/`, `src/learning/`, `src/evaluation/`, `src/infrastructure/`, `src/visualization/` com mapa de migração, `train_step(net, counters, x, target, config, trace=False)` com políticas explícitas de 020 (perda, seleção, agenda, reset, leak), separando sinal, acumulação, seleção e aplicação, com tracing opcional sobre intermediários reais sem segundo backward/update. Migrar B1, ablações, wrapper refinado, stepper/runner/comparador do playground e separação de `utils.py` (dados, plotting, métricas, persistência) para o núcleo, reutilizando a baseline sem duplicar arquitetura. Manter adaptadores de compatibilidade para reproduzir históricos. Capturar referência determinística antes de substituir o legado. Não selecionar nova B+.

## Contexto
- `src/ternary_b1.py`, `src/ablation.py`, `src/playground/tracer.py` — tripla duplicação de backward/acumulação/update a unificar.
- `src/utils.py` — agregados de dados, plotting, métricas e persistência a separar.
- `src/baseline.py` — arquitetura a reutilizar sem duplicar.
- `src/counter_rules.py`, `src/reset_policies.py` — políticas já definidas em 020.
- `docs/contrato-matematico.md` — arquivo que existirá após 020 e rege as políticas (dependência).
- `experiments/run_ablations.py`, `experiments/run_mnist_full.py` — consumidores do núcleo.

## Corretude
- [ ] C1 — Existe uma única implementação de backward/acumulação/update nos caminhos ativos; B1, ablações, wrapper refinado e playground a consomem.
- [ ] C2 — Configuração legada reproduz a referência 012 em sequência determinística (pesos, bias, contadores, métricas iguais).
- [ ] C3 — Trace ligado/desligado produz mesmos pesos, bias, contadores e métricas; cópia de arrays só quando o trace exige snapshots.
- [ ] C4 — Políticas diferentes alteram somente etapas previstas pelo contrato; suíte 019, smoke CLI e smoke do playground passam com comandos registrados.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
8/10 — Refatoração estrutural ampla com risco de divergência silenciosa entre caminhos; exige referência determinística e migração sem quebra.

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
