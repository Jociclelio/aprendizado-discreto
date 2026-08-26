# [008] Implementa variantes B2 e B3 com todas as políticas de reset

## Objetivo

Implementar e treinar as configurações restantes da Rede B:
- **B2** (por camada Top-K) × 3 políticas de reset
- **B3** (por rede Top-K) × 3 políticas de reset

São 6 configurações adicionais para completar a varredura qualitativa.

## Contexto

Com B1 validado, agora exploramos as regras de seleção Top-K. Cada variante é testada com as 3 políticas de reset (error-feedback, full reset, normalize), totalizando 6 experimentos. O objetivo é identificar quais combinações aprendem e quais não.

## Escopo

- [ ] Criar `src/ternary_b2.py` — B2 (Top-K por camada)
- [ ] Criar `src/ternary_b3.py` — B3 (Top-K por rede)
- [ ] Para cada variante, rodar 3 experimentos (uma por política de reset):
  - B2 + error-feedback → `results/exp-008-b2-ef/`
  - B2 + full reset → `results/exp-008-b2-fr/`
  - B2 + normalize → `results/exp-008-b2-norm/`
  - B3 + error-feedback → `results/exp-008-b3-ef/`
  - B3 + full reset → `results/exp-008-b3-fr/`
  - B3 + normalize → `results/exp-008-b3-norm/`
- [ ] Script de execução: `experiments/run_b2_b3.sh` ou `experiments/run_b2_b3.py`
- [ ] Mínimo: 20 epochs por configuração

## Especificações técnicas

- **Hiperparâmetros B2**:
  - `K`: começar com 10% do total de pesos na camada (ex.: K=16 para camada 784→128)
  - Condição de atualização: a cada epoch ou a cada N amostras?
  - Sugerir: atualizar a cada epoch (mais simples para comparação)
- **Hiperparâmetros B3**:
  - `K_per_layer`: mesmo K para todas as camadas
  - Total de atualizações: `K × N_camadas`
  - Condição de atualização: a cada epoch
- **Hiperparâmetros compartilhados**:
  - `scale` (fake gradient): 1.0
  - `radius` (fake gradient): 1.0
  - `levels` (quantização): 5
  - `epochs`: 20 mínimo
- **Perda e avaliação**: idênticas ao B1 (para comparação justa)

## Critérios de aceitação

- [ ] Código implementado em `src/ternary_b2.py` e `src/ternary_b3.py`
- [ ] 6 experimentos executam sem erro
- [ ] Pelo menos 2 configurações mostram sinal de aprendizado (acurácia > 10%)
- [ ] Curvas de acurácia salvas para cada configuração
- [ ] Commit [008] com mensagem descritiva
- [ ] Resultados salvos em `results/exp-008-*/`

## Notas de implementação

- **Condição de atualização B2/B3**: a cada epoch é mais simples, mas pode ser lento (poucas atualizações). Alternativa: a cada mini-batch ou a cada N amostras.
- **K progressivo**: considerar aumentar K ao longo do treino (mais exploração no início)
- **Monitorar**: fração de pesos atualizados, estabilidade do δ em profundidade, magnitude dos contadores
- **Comparar com B1**: usar os mesmos hiperparâmetros de fake gradient para isolar o efeito da regra de seleção

## Dependências

- Bloqueado por: `[007]`
- Bloqueia: `[009]`
