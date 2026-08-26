# [009] Comparação qualitativa final — Rede A vs. todas as configurações da Rede B

## Objetivo

Gerar gráficos comparativos lado a lado: curva de acurácia da baseline (Rede A) vs. todas as configurações da Rede B que mostraram aprendizado. Análise qualitativa dos comportamentos observados.

## Contexto

Esta é a etapa final da primeira rodada de experimentos. O objetivo não é benchmark quantitativo definitivo, mas entender qualitativamente como a técnica proposta se comporta em relação ao padrão vigente.

## Escopo

- [ ] Criar `src/compare.py` — script de geração de gráficos comparativos
- [ ] Gráfico 1: Acurácia de treino vs. teste — Rede A vs. todas as configurações B
  - Eixo X: epochs, Eixo Y: acurácia
  - Linha sólida = baseline, linhas tracejadas = configurações B
- [ ] Gráfico 2: Fração de pesos que mudam por epoch — comparar B1/B2/B3
- [ ] Gráfico 3: Magnitude do δ por camada (estabilidade da propagação)
  - Para cada configuração B, plotar magnitude média de δ em cada camada ao longo do treino
- [ ] Resumo qualitativo em `results/exp-009-comparison/README.md`
  - Quais configurações aprendem?
  - Quais políticas de reset funcionam melhor?
  - B1 vs B2 vs B3: qual converge mais rápido?
  - Estabilidade do δ em profundidade?
  - Observações sobre custo de memória vs acurácia

## Especificações técnicas

- **Carregar resultados**: ler CSVs de `results/exp-NNN-*/`
- **Plotting**: matplotlib, layout grid (baseline à esquerda, configurações B à direita)
- **Resumo qualitativo**: markdown, listar observações objetivas
- **Métricas a extrair**:
  - Acurácia final de teste
  - Epoch de convergência (quando acurácia estabiliza)
  - Fração média de pesos atualizados por epoch
  - Magnitude máxima/mínima de δ por camada
  - Bits por peso efetivos (ternário ≈ 1.58 bit vs 32 bit float)

## Critérios de aceitação

- [ ] Código implementado em `src/compare.py`
- [ ] Gráficos gerados e salvos em `results/exp-009-comparison/`
- [ ] Resumo qualitativo escrito
- [ ] Commit [009] com mensagem descritiva
- [ ] Resultados salvos em `results/exp-009-comparison/`

## Notas de implementação

- **Se poucas configurações aprenderam**: o gráfico comparativo pode ser pequeno. Isso é uma informação válida — documentar.
- **Se nenhuma configuração B aprendeu**: flagar como resultado negativo importante e documentar hipóteses de porquê.
- **Custo de memória**: calcular e reportar para cada configuração B:
  - Pesos: int8 = 1 byte cada
  - Contadores: int64 = 8 bytes cada (temporário, pode ser descartado)
  - Total: ~107KB pesos + ~856KB contadores = ~963KB vs ~428KB baseline
  - Sem contadores (deploy): ~107KB vs ~428KB → ~4x economia

## Dependências

- Bloqueado por: `[007]`, `[008]`
- Bloqueia: nenhuma (última task desta rodada)
