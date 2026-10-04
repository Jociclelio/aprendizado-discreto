# Task 019: estrutura testes e ambiente reproduzível

Estado: pendente
Depende de: 018
Origem: nenhuma

## Prompt
Criar suíte automatizada e ambiente reproduzível sem refatorar `src/` estruturalmente (isso é 021). Entregar: `pyproject.toml` com Python mínimo validado e config pytest, manifesto de versões resolvidas com instruções de instalação (núcleo x dev x playground), `tests/unit/` + `tests/integration/` com pytest, migração dos casos relevantes de 012 para testes determinísticos, cobertura de direção de atualização, forward/avaliação, deltas por camada, máscara/reset, bias, leak simétrico, normalização, limites de Top-K e equivalência CLI/tracer. Testes padrão não baixam MNIST, não exigem GPU/GUI, não escrevem em `results/` e não executam o aceite histórico sobre o diretório histórico.

## Contexto
- `results/exp-012-b1-reparo/run_012_checks.py` — casos de aceite a migrar sem sobrescrever o histórico.
- `src/ternary_b1.py` — núcleo com contrato 012 a proteger por regressão.
- `src/utils.py` — agregados a testar sem efeitos de escrita.
- `requirements.txt` — manifesto atual sem versões travadas.
- `.opencode/tasks/proxima-rodada/018-organiza-documentacao-e-fila.md` — índices e estado reconciliado.

## Corretude
- [ ] C1 — `python -m pytest` a partir da raiz passa no ambiente registrado, sem acesso a dados externos e sem criar artefatos em `results/` (saídas em diretório temporário do runner).
- [ ] C2 — Regressões dos defeitos críticos do Astra estão representadas por testes com oráculo comportamental (redes pequenas, expectativas conhecidas), não apenas reprodução de fórmulas.
- [ ] C3 — Equivalência CLI/tracer verificada em mais de um passo com sequência sintética em `[0,1]`, raio adaptativo e contadores acumulados.
- [ ] C4 — Dependências e versões suficientes para recriar o ambiente estão versionadas; instalação documentada foi executada e registrada.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
5/10 — Criar harness pytest e traduzir checks ad-hoc em testes determinísticos exige desenho de oráculos, mas sem refatoração do núcleo.

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
