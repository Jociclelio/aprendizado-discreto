# [006] Implementa políticas de zeragem de contadores

## Objetivo

Implementar `src/reset_policies.py` com as três políticas de reset de contadores:
- **Error-feedback**: zerar somente os contadores das conexões atualizadas
- **Full reset**: zerar todos os contadores da camada/rede
- **Normalize**: normalizar os contadores restantes (escalar para manter magnitude média)

## Contexto

A hipótese principal é que preservar o resíduo (error-feedback) supera zerar tudo, sustentado por resultados de compressão de gradiente (Deep Gradient Compression; Error Feedback for SignSGD). Esta task implementa as três políticas para comparação empírica.

## Escopo

- [ ] Criar `src/reset_policies.py`
- [ ] Função `reset_error_feedback(counters, updated_indices)` — B
  - Zera contadores apenas nas posições de `updated_indices`
  - Contadores não-atualizados mantêm seu valor (resíduo preservado)
- [ ] Função `reset_full(counters)` — B
  - Zera todos os contadores da camada
- [ ] Função `reset_normalize(counters, updated_indices)` — B
  - Zera contadores atualizados
  - Escala contadores restantes por fator `mean(|C_non_updated|) / target_scale`
  - `target_scale` = hiperparâmetro (sugerir: 1.0 ou magnitude inicial)
- [ ] Classe `ResetPolicy` — factory/enum para selecionar política
  - `apply(policy_name, counters, updated_indices)`
  - `policy_name ∈ {"error_feedback", "full_reset", "normalize"}`

## Especificações técnicas

- **Error-feedback**: `counters[updated_indices] = 0`, demais inalterados
- **Full reset**: `counters[:] = 0`
- **Normalize**:
  - `non_updated = counters.copy(); non_updated[updated_indices] = 0`
  - `scale_factor = np.mean(np.abs(non_updated[non_updated != 0])) / target_scale` (se houver não-zero)
  - `counters[~updated_indices] *= scale_factor`
  - Se todos zerados, não fazer nada
- **Tipos**: numpy int64 para contadores

## Critérios de aceitação

- [ ] Código implementado em `src/reset_policies.py`
- [ ] `reset_error_feedback()` zera apenas índices fornecidos
- [ ] `reset_full()` zera todos os contadores
- [ ] `reset_normalize()` escala corretamente os não-atualizados
- [ ] `ResetPolicy.apply()` roteia corretamente para cada política
- [ ] Testes unitários
- [ ] Commit [006] com mensagem descritiva

## Notas de implementação

- **Error-feedback** é a política favorita (hipótese principal) — priorizar sua correção
- **Normalize** deve preservar a informação relativa entre contadores não-atualizados
- Todas as funções devem ser puras (não modificar o array original) ou ter comportamento documentado
- Considerar se normalize deve ser aplicado antes ou depois do zero dos atualizados

## Dependências

- Bloqueado por: `[005]`
- Bloqueia: `[007]`, `[008]`
