# Pool de revisores — ordem de prioridade

Fonte de verdade do pool de revisores do task-master. Para trocar os
revisores padrão, edite **este arquivo** (reordene linhas, adicione ou
remova membros) — não é preciso editar skills ou ferramentas.

- A linha 1 é a maior prioridade.
- Os três primeiros com modelos distintos formam a rodada padrão.
- O restante é a ordem de fallback quando um revisor falha ou está indisponível.
- Modelos verificados por spawn real em 2026-10-13
  (`pi --mode json --print --no-session -ne` com ping de resposta).
- Excluídos por decisão do usuário: modelos do provider llama-cpp
  (`llama-server=http://127.0.0.1:8081`).

| #   | Papel         | Modelo                       |
|-----|---------------|------------------------------|
| 1   | task-reviewer | sd2/GLM-5.3-Flash            |
| 2   | task-reviewer | google/gemini-3.8-flash      |
| 3   | task-reviewer | google/gemini-3.7-flash      |
| 4   | task-reviewer | google/gemini-3.6-flash      |
| 5   | task-reviewer | google/gemini-3.5-flash-lite |
| 6   | task-reviewer | google/gemma-4-31b-it        |

## Indisponíveis (reavaliar antes de re-incluir)

| Modelo | Motivo |
|---|---|
| nvidia/* (todos os modelos) | 403 Authorization failed — chave NVIDIA inválida/expirada (`~/.pi/agent/auth.json`); reinserir após renovar |
| opencode-zen-free/* (muse-spark-1.3, big-pickle, longcat-2.5, etc.) | FreeTierError keyless neste egress IP ("free tier can only be used from within OpenCode"); requer Zen key (`/login opencode-zen-free` no pi ou `OPENCODE_API_KEY`) |
| sorgan/qwen3-27b, sorgan/qwen3.8-flash-next | servidor local :8082 parado (connection error); reinserir após `~/sorgan-llm.sh` |
| nvidia/z-ai/glm-5.2 | fim de vida (410 Gone) desde 21/08/2026 |
