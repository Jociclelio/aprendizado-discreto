# [028] Port do workflow task-cycle (opencode → pi/pidev)

### Objetivo

Implementar para o pi (pidev) o mesmo workflow global de tasks com subagentes
existente no opencode: plano → tasks → execução → revisão por quórum de 3
modelos, com ciclo automático opt-in, usando a convenção de projetos `.agents/`
para intercambiar documentos entre ferramentas.

### Contexto

No opencode o fluxo vive em `~/.config/opencode/` (agents, skills, plugin
`task-cycle`). O pi não tem "agents" nativos; os equivalentes são extensões
(ferramentas), skills (papéis) e subprocessos `pi --mode json` com modelo
explícito. A instalação é global no pidev, com setup por repositório via
`.agents/`.

### Escopo

- [x] Extensão pi com ferramentas `task_cycle`, `task_execute`, `task_review`
- [x] Comando `/taskcycle-init` para setup de repositório (`.agents/`)
- [x] Handoff pós-idle via `agent_settled` + `sendUserMessage`
- [x] Checkpoint + lock em `.git/pi-task-cycle-state.json(.lock)`
- [x] Skills: `planer`, `task-writer`, `task-executor`, `task-master`
- [x] Contratos de papel (roles) para executor e revisor
- [x] Pool de revisores com vários modelos (sd2, sorgan, nvidia)
- [x] Testes unitários da máquina de estados (`node --test`)
- [x] `install.sh` (registro global em `~/.pi/agent/settings.json`)
- [x] WORKFLOW.md + README.md adaptados

### Especificações técnicas

- Pacote em `~/Documentos/pidev/pi-task-cycle/` (repo git próprio)
- Filhos: `pi --mode json --print --no-session -ne --model <prov/id>
  --tools <allowlist> --append-system-prompt <role.md>`
- Identidade real verificada por `provider`/`model` do `message_end`
- Marcador `<!-- task-cycle: {...} -->` idêntico ao opencode (intercâmbio)
- Decisões: fechar/corrigir/reiniciar/repartir/concluido/bloqueada
- Limites: `max_cycles=20`, `max_family_attempts=3`

### Critérios de aceitação

- [x] `node --test tests/` passa no pacote
- [x] Extensão carrega no pi sem erro de sintaxe/registro
- [x] Documentação de uso equivalente à do opencode
- [x] Commit `[028]` no repositório

### Notas de implementação

- pi não cria sessão raiz nova por task: o ciclo continua na mesma sessão do
  master (handoff por mensagem de usuário após idle); compaction cuida do
  contexto. Divergência documentada no WORKFLOW.md.
- Sem revisores explícitos no ciclo, o master lê o pool por arquivo
  (`.agents/task-reviewer-pool.md` no projeto; global no pacote).
