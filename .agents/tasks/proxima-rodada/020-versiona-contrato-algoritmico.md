# Task 020: versiona contrato algorítmico e políticas de seleção

Estado: pendente
Depende de: 019
Origem: nenhuma

## Prompt
Escrever a versão inequívoca do algoritmo antes da refatoração 021. Entregar `docs/contrato-matematico.md` versionado (entradas, ativações, scores, perda, deltas, arredondamento, acumulação, seleção, aplicação, reset, leak, saturação) com identificadores explícitos para limiar por neurônio x por camada e Top-K por camada x global (incluindo agenda), orçamento de pesos/bias, elegibilidade de contador zero, `K=0`, validação de parâmetros, desempate determinístico, e distinção reset seletivo x error-feedback clássico. Registrar diferenças ideia original x `CONTRATO_012` e criar decisões em `docs/decisions/`. Atualizar descrições conflitantes em README/docstrings e adicionar testes de borda às primitivas existentes quando necessários ao contrato. Não escolher nova variante B+ nem medir aprendizado.

## Contexto
- `src/ternary_b1.py` — B1 com limiar médio de camada a especificar como tal.
- `src/counter_rules.py`, `src/reset_policies.py`, `src/fake_gradient.py` — primitivas com semântica a travar.
- `src/playground/tracer.py` — Top-K global a distinguir do Top-K por camada da proposta.
- `results/exp-012-b1-reparo/` — referência `CONTRATO_012` a preservar como configuração legada.
- `.agents/tasks/proxima-rodada/019-estrutura-testes-e-ambiente.md` — suíte que deve continuar passando.

## Corretude
- [ ] C1 — Contrato permite reconstruir cada política sem depender de nomes ambíguos (B1/B2/B3 citados só com identificador de política); versão e relação com 012 documentadas.
- [ ] C2 — Agenda, orçamento, bias, saturação e reset definidos separadamente; decisões distinguem correção de implementação x mudança de hipótese.
- [ ] C3 — Semântica por amostra no backward especificada (helper com média de surrogate no batch não conta como regra local vetorizada); `round` tratado como possível descarte de evidência subunitária salvo escala explícita de ponto fixo.
- [ ] C4 — Testes de borda pertinentes passam via `python -m pytest`; comandos registrados na task.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
6/10 — Exige desambiguar semânticas sutis (limiares, Top-K, backward por amostra, arredondamento) e travar em texto versionado sem quebrar legado.

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
