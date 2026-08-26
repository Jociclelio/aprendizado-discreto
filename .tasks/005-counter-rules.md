# [005] Implementa regras de atualização por contador (B1, B2, B3)

## Objetivo

Implementar `src/counter_rules.py` com as três regras de seleção de conexões para atualização:
- **B1**: Por neurônio (disparo individual)
- **B2**: Por camada (Top-K na camada)
- **B3**: Por rede (Top-K por camada simultaneamente)

## Contexto

A unidade fundamental de aprendizado é a **conexão individual** (seu contador). O neurônio e a camada só fornecem o contexto que decide quando e quais conexões atualizar. Esta task implementa os três mecanismos de seleção.

## Escopo

- [ ] Criar `src/counter_rules.py`
- [ ] Classe `CounterManager` — gerencia contadores por camada
  - `C_ij` inteiro por conexão (peso) + por bias
  - Métodos: `accumulate(layer_idx, delta, input)`, `get_top_k(layer_idx, k)`, `get_neuron_counters(layer_idx, neuron_idx)`
- [ ] Função `apply_update_neuron(layer, counters, neuron_idx, delta_sign)` — B1
  - Quando neurônio dispara, atualizar todas as suas conexões de entrada baseadas nos contadores individuais
  - Regra: se `|C_ij| > threshold`, atualizar `W_ij` na direção de `sign(C_ij)`
- [ ] Função `apply_update_layer_topk(layer, counters, k)` — B2
  - Selecionar K conexões com maior `|C_ij|` na camada inteira
  - Atualizar apenas essas K conexões
- [ ] Função `apply_update_network_topk(layers, counters_list, k_per_layer)` — B3
  - Aplicar Top-K em cada camada simultaneamente
  - Total: `N_layers × K` atualizações por evento
- [ ] Função `update_weight(W, sign)` — aplica uma unidade de mudança: `W += sign` (com clip em {−1, 0, +1})

## Especificações técnicas

- **Contadores**: numpy int64 por conexão (para evitar overflow)
- **Top-K**: numpy argpartition para eficiência (O(n) vs O(n log n))
- **Threshold B1**: hiperparâmetro configurável; se não configurado, usar `max(1, mean(|C|))`
- **K**: hiperparâmetro configurável; para B2/B3, sugerir K ≈ 10% do total de pesos na camada
- **Update**: cada conexão muda de −1 → 0 → +1 (ou inverso) por incremento unitário

## Critérios de aceitação

- [ ] Código implementado em `src/counter_rules.py`
- [ ] `CounterManager.accumulate()` incrementa corretamente os contadores
- [ ] `get_top_k()` retorna os índices das K maiores conexões
- [ ] `apply_update_neuron()` atualiza conexões do neurônio especificado
- [ ] `apply_update_layer_topk()` atualiza apenas K conexões da camada
- [ ] `apply_update_network_topk()` atualiza K conexões em cada camada
- [ ] Testes unitários para cada regra
- [ ] Commit [005] com mensagem descritiva

## Notas de implementação

- **B1 (por neurônio)**: a condição de disparo do neurônio é definida pelo forward pass (degrau > 0). Quando dispara, todas as suas conexões de entrada são candidatas.
- **B2 (por camada Top-K)**: a camada inteira é avaliada; apenas as K conexões com maior evidência (|C|) são atualizadas.
- **B3 (por rede Top-K)**: o mesmo que B2, mas aplicado a todas as camadas simultaneamente.
- **Threshold vs K**: B1 usa threshold, B2/B3 usam K. Ambos são hiperparâmetros a ajustar.

## Dependências

- Bloqueado por: `[004]`, `[003]`
- Bloqueia: `[007]`, `[008]`
