# Pool de revisores — ordem de prioridade

Fonte de verdade do pool de revisores do task-master. Para trocar os
revisores padrão, edite **este arquivo** (reordene linhas, adicione ou
remova membros) — não é preciso editar skills ou ferramentas.

- A linha 1 é a maior prioridade.
- Os três primeiros com modelos distintos formam a rodada padrão.
- O restante é a ordem de fallback quando um revisor falha ou está indisponível.
- `.agents/task-reviewer-pool.md` no projeto sobrepõe este arquivo global.
- Modelos precisam existir no catálogo do pi (`pi --list-models`).

| #   | Papel         | Modelo                     |
|-----|---------------|----------------------------|
| 1   | task-reviewer | sd2/GLM-5.3-Flash          |
| 2   | task-reviewer | sorgan/qwen3.8-flash-next  |
| 3   | task-reviewer | nvidia/z-ai/glm-5.2        |
| 4   | task-reviewer | sorgan/qwen3-27b           |
