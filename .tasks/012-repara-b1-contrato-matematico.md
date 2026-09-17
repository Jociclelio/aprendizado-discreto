# [012] Repara o B1 — contrato matemático e correções P0/P1

## Objetivo

Deixar o treinador B1 (`src/ternary_b1.py`) matematicamente consistente com a ideia proposta, corrigindo os achados P0 (A–D) e P1 (E–H) do `astra-relatório-commit-6f47914.md`, de modo que os próximos experimentos medam a técnica e não os bugs de implementação.

## Contexto

A análise do repositório (commit `6f47914`) identificou defeitos que impedem qualquer interpretação dos ~10% registrados: a avaliação usa uma rede diferente da treinada, a atualização de saída pode empurrar o peso na direção contrária, as ocultas reutilizam o delta da camada errada, seleção e reset de B1 não correspondem, o bias não aprende, a normalização dos contadores inverte a escala e os leaks têm viés de arredondamento oposto. Enquanto esses itens persistirem, adicionar margem/homeostase (015) só mascara o problema.

Esta task é a **Etapa 1** da sequência recomendada no relatório (seção 9).

## Escopo

### Contrato matemático (definir e documentar antes de codar)

- [x] Especificar, num bloco único (docstring de módulo ou `.tasks/012-...md`), o que cada símbolo é:
  - `z_l = W_l @ x_{l-1} + b_l` (pré-ativação), `a_l = step(z_l)`;
  - o **score** final usado para `argmax` (é `z_out` da camada de saída, antes do degrau);
  - o **sinal** que o contador `C` acumula: definir se é **gradiente** (`W ← clip(W − sign(C))`) ou **direção de descida** (`W ← clip(W + sign(C))`); escolher UM e aplicar a todas as camadas sem multiplicação extra por `sign(−error)` da amostra corrente;
  - a **semântica de saturação**: o que acontece quando o peso já está em ±1 e a evidência pede o mesmo sentido (a tentativa é bloqueada; definir se o contador é consumido ou mantido).
- [x] Decidir a **política de arredondamento do leak** (simetria de sinal, comportamento perto de zero, pontos fixos) e documentar a escolha.

### Correções P0 (obrigatórias)

- [x] **(A)** Unificar o forward: uma única implementação que exponha `z_out` (scores) e `a_out` (binária); `evaluate()` deve consumir os scores, **sem reimplementar** as camadas e aplicando o degrau em **todas** as ocultas.
- [x] **(B)** Corrigir a direção da atualização: remover a segunda combinação de sinais (`sign(C) * sign(−error)`); usar a convenção escolhida no contrato.
- [x] **(C)** Guardar `deltas[layer_idx]` explicitamente no backward; cada camada usa **o seu próprio** delta para o sinal de atualização.
- [x] **(D)** Calcular **uma máscara única** de seleção em B1; aplicá-la à atualização **e** ao reset (error-feedback). Reconciliar `train_sample_online` com a primitiva `apply_update_neuron()`.

### Correções P1 (obrigatórias)

- [x] **(E)** Implementar o aprendizado do bias: passar `delta_bias` para `CounterManager.accumulate()` e atualizar `layer.bias` dentro da faixa `[-N, N-1]`.
- [x] **(F)** Corrigir `reset_normalize()` para usar o fator inverso (`target_scale / mean_abs`) com regra de quantização inteira; corrigir `apply_reset_on_manager_all_layers(policy="normalize")` para usar índices de dtype inteiro.
- [x] **(G)** Alinhar os dois leaks à política de arredondamento decidida (corrigir o comentário de "simetria" do bitshift e testar simetria/pontos fixos).
- [x] **(H)** Aplicar quantização coerente à propagação das ocultas (evitar truncar produtos < 1 a zero antes de acumular); medir o sinal antes/depois da conversão.

### Métricas e robustez (P2 selecionados)

- [x] Separar nas métricas: **selecionados**, **tentativas**, **bloqueios por clipping** e **transições efetivas** (peso que de fato trocou de valor).
- [x] Alinhar `DEFAULT_HYPERPARAMS` com a assinatura de `train_b1()` (raio adaptativo on/off).
- [x] Corrigir `ternary_binary_loss()` (docstring: L1 médio, não BCE) e/ou adicionar métrica anti-colapso (taxa de disparos por camada).
- [x] Validar `K=0` e limites de `get_top_k_weights()` / `top_k_indices()`.

## Especificações técnicas

- **Arquitetura**: `784 → 128 → 64 → 10`, inalterada.
- **Módulos afetados**: `src/ternary_b1.py`, `src/ternary_net.py`, `src/counter_rules.py`, `src/reset_policies.py`, `src/fake_gradient.py`.
- **Interface esperada**:
  - `TernaryNet.forward()` (ou `forward_with_scores`) retorna `(scores, activations)`; `evaluate()` usa `scores`.
  - `train_sample_online(...)` guarda `deltas` por camada e uma `selection_mask` por camada, reaproveitadas na atualização e no reset.
- **Regra de atualização (convenção a fixar)**: `C` acumula gradiente ⇒ `W ← clip(W − sign(C), -1, 1)`.
- **Tipos**: pesos `int8`, bias `int32`, contadores `int64`.
- **Dependências entre módulos**: `ternary_b1` importa de `ternary_net`, `counter_rules`, `reset_policies`, `fake_gradient`, `utils`.

## Critérios de aceitação

- [x] Contrato matemático escrito e consistente com o código.
- [x] Testes determinísticos em **redes pequenas** (casos controlados) comprovam:
  - direção esperada da atualização (reprodução do relatório agora invertida);
  - `evaluate()` == `argmax(scores)` de um forward correto;
  - delta de cada camada aplicado à **própria** camada;
  - seleção e reset usam a **mesma** máscara;
  - bias muda dentro da faixa quando há evidência;
  - leak simétrico nos pontos testados.
- [x] Os dois casos mínimos do relatório (seção 10) passam com o resultado esperado:
  - avaliação pequena → 1.0 (antes 0.0);
  - classe correta inativa → peso de saída vai para `+1` (antes `−1`).
- [x] Testes existentes continuam passando: `python -m src.ternary_b1`, `python -m src.counter_rules`, `python -m src.reset_policies`.
- [x] Commit `[012]` com mensagem descritiva.
- [x] Notas da reparação registradas nesta task e em `results/exp-012-b1-reparo/NOTAS.md`.

## Notas de implementação

- **Não exigir** queda de loss em cada passo de rede profunda quantizada; os testes de aceite usam casos controlados com direção conhecida.
- As reproduções do relatório (seção 10) são o ponto de partida dos testes — convertê-las em testes determinísticos.
- Ao corrigir o sinal (B), rever o teste de B1 em `counter_rules.py` que fixa a convenção antiga.
- Manter a separação cálculo (forward/backward) x aplicação (atualização) para facilitar a reutilização por B2/B3 e por 015.
- Este commit deve ser **revisável**: um diff focado em correção, sem introduzir novos mecanismos de aprendizado (margem/homeostase ficam em 015).

## Dependências

- Bloqueado por: `[011]`
- Bloqueia: `[013]`

## Registro de execução (2026-09-17)

- Contrato CONTRATO_012 fixado na docstring de `src/ternary_b1.py`.
- 22/22 checks em `results/exp-012-b1-reparo/run_012_checks.py` → `metrics.json`.
- Casos mínimos: avaliação → 1.0; peso de saída → +1.
- Existentes OK: `ternary_b1`, `counter_rules`, `reset_policies`, `fake_gradient`, `ternary_net`.
- Detalhes em `results/exp-012-b1-reparo/NOTAS.md`.
