# AGENTS.md — Convenções para Agentes de IA

## Princípio geral

Este repositório é gerido por agentes de IA (pi, Claude, etc.). Todas as alterações devem seguir estas convenções para manter a trilha de decisão auditable e reproduzível.

## Regras

### 1. Tasks são a unidade de trabalho

- Toda tarefa de implementação ou experimentação deve ter seu próprio arquivo em `.tasks/`.
- Tasks são executadas **uma por vez**, na ordem numérica.
- Nenhuma task deve ser marcada como "done" sem:
  - Código implementado e testável
  - Referência cruzada ao arquivo de task
  - Commit git com mensagem descritiva

### 2. Nomenclatura de tasks

```
.tasks/NNN-nome-curto.md
```

- `NNN`: número sequencial de 3 dígitos (001, 002, ...)
- `nome-curto`: kebab-case, descreve o objetivo

### 3. Estrutura de um arquivo de task

Ver `.tasks/task.md` para o template padrão.

### 4. Commits

- Mensagens no formato: `[NNN] descrição curta`
- Exemplo: `[001] Implementa MLP baseline com PyTorch para MNIST`
- Um commit por task concluída (ou sub-tasks agrupáveis logicamente)

### 5. Código

- Arquivos em `src/` com nomes descritivos
- Funções com docstrings
- Preferência por clareza sobre brevidade
- Testes unitários mínimos para validar cada módulo

### 6. Resultados

- Gráficos e métricas em `results/`
- Logs de experimento em `results/logs/`
- Cada experimento tem seu subdiretório: `results/exp-NNN-nome/`

### 7. Branches (se necessário)

- `main` — sempre funcional
- `exp/NNN-nome` — experimentos isolados

## Fluxo de trabalho típico

```
1. Ler .tasks/task.md (padrão)
2. Criar .tasks/NNN-nome.md (nova task)
3. Implementar
4. Testar
5. Commit: [NNN] descrição
6. Atualizar README.md se estrutura mudou
```
