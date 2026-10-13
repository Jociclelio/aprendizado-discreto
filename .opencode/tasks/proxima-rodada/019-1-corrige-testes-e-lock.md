# Task 019-1: corrige tautologias, entradas [0,1] e pins

Estado: pendente
Depende de: 019
Origem: 019 (correção)

## Prompt
Corrigir os problemas identificados na revisão da task 019:

1. **Asserções tautológicas** (C2): Três testes em `tests/integration/test_b1_integration.py` possuem asserções que não verificam nada:
   - `test_independencia` — termina em `assert True` sem comparar `d1` vs `d1b`
   - `test_indices_int_sem_IndexError` — try/except com `assert True`
   - `test_apply_reset_on_manager_all_layers_normalize` — try/except com `assert True`/`allint`
   
   Corrigir para ter oráculos comportamentais reais: comparar resultados, verificar propriedades, usar asserções que falhariam se a regressão voltasse.

2. **Entradas fora do contrato** (C3): Os testes de equivalência CLI/tracer em `tests/integration/test_cli_tracer_equivalence.py` usam `np.random.randn(784)` (normal ilimitada). O contrato da task 019 exige "sequência sintética em `[0,1]`". Substituir por `np.random.rand(784)` ou `np.clip(np.random.randn(784), 0, 1)`.

3. **Dependências não travadas** (C4): No `requirements-lock.txt`, as dependências `scikit-learn>=1.3` e `pytest-cov>=5.0` estão com faixa aberta. Travar para versões específicas (ex.: `scikit-learn==1.3.0`, `pytest-cov==5.0.0` ou versões instaladas no ambiente validado).

## Contexto
- `.opencode/tasks/proxima-rodada/019-estrutura-testes-e-ambiente.md` — task original com os critérios C1-C4
- `tests/integration/test_b1_integration.py` — contém as 3 asserções tautológicas
- `tests/integration/test_cli_tracer_equivalence.py` — usa `randn` em vez de `[0,1]`
- `requirements-lock.txt` — dependências com faixa aberta

## Corretude
- [x] C1 — `python -m pytest` continua passando (118 testes) após as correções
- [x] C2 — As 3 asserções tautológicas foram substituídas por oráculos comportamentais reais
- [x] C3 — Testes de equivalência usam sequência sintética em `[0,1]`
- [x] C4 — Todas as dependências no `requirements-lock.txt` estão pinadas com `==`

## Execução
- Modelo: pytest 9.1.1 + numpy 2.5.3 + torch 2.13.0 + scikit-learn 1.9.1 + pytest-cov 7.1.0
- Base: 3143ee1 (HEAD pré-entrega), ambiente validado Python 3.14.7
- Entrega: eaab968
- Arquivos da entrega: `tests/integration/test_b1_integration.py`, `tests/integration/test_cli_tracer_equivalence.py`, `tests/unit/test_reset_policies.py`, `requirements-lock.txt`
- Evidências: 118 testes passam em 3.35s; `grep "assert True" tests/` vazio; `grep -c ">=" requirements-lock.txt` = 0 (8 pins `==`); oráculos verificados numericamente antes de fixar (test_independencia: d1=[[4,-4]], d1b=[[0,0]] com pesos da camada 2 zerados; normalize: 10×(1/10)=1 em todas as camadas com dtype int64 preservado); scikit-learn e pytest-cov instalados no ambiente validado antes do pin para registrar as versões reais

## Notas

### Dificuldade esperada
3/10 — Correções pontuais e bem delimitadas: substituir asserções, trocar fonte de entradas e travar versões.

### Dificuldade sentida
2/10 — Correções pontuais; o único cuidado foi verificar numericamente os oráculos (d1b anulado, escala 10→1) antes de fixar as asserções, e instalar scikit-learn/pytest-cov no ambiente validado para pinar versões reais em vez de chutes.

### Revisão
- Orquestrador: —
- Alvo: —
- Avaliações: —
- Nota final: _/10
- Parecer: —
- Decisão: pendente
- Reversão: não se aplica
- Próximo passo: 020

- Estado: entrega eaab968 registrada, aguardando revisão

Plano: [plan.md](../../plans/proxima-rodada/plan.md)
