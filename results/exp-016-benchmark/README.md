# exp-016 — Benchmark A vs B1 vs B+

## Resultados

| Rede | Seed | Val Acc | Test Acc | Tempo (s) | Bits/Param | Mem Params (KB) | Mem Treino (KB) |
|---|---|---|---|---|---|---|---|
| A (baseline) | 42 | 0.9218 | 0.9252 | 5.6 | 32.00 | 427.3 | 1281.9 |
| B1 (corrigido) | 42 | 0.0932 | 0.0974 | 52.0 | 1.58 | 107.4 | 962.0 |
| B+ (refinada) | 42 | 0.1022 | 0.0892 | 46.3 | 1.58 | 107.4 | 962.0 |
| A (baseline) | 123 | 0.9220 | 0.9323 | 3.4 | 32.00 | 427.3 | 1281.9 |
| B1 (corrigido) | 123 | 0.0924 | 0.0892 | 52.8 | 1.58 | 107.4 | 962.0 |
| B+ (refinada) | 123 | 0.1096 | 0.1009 | 44.5 | 1.58 | 107.4 | 962.0 |
| A (baseline) | 999 | 0.9284 | 0.9358 | 3.3 | 32.00 | 427.3 | 1281.9 |
| B1 (corrigido) | 999 | 0.1000 | 0.1010 | 51.9 | 1.58 | 107.4 | 962.0 |
| B+ (refinada) | 999 | 0.1038 | 0.0892 | 43.9 | 1.58 | 107.4 | 962.0 |

## Largura de contador

| Bits | Val Acc | Test Acc |
|---|---|---|
| 8bits | 0.0932 | 0.0974 |
| 16bits | 0.0932 | 0.0974 |

## Conclusoes

- Reducao de memoria de parametros: 3.98x
- Overhead de memoria de treinamento: 2.25x
- 'Zero MACs float' e propriedade de uma implementacao especializada, nao do codigo atual.
- A comparacao de memoria de treinamento e incompleta sem considerar momentos do Adam e temporarios.
- Se a Rede B nao alcancar a baseline, a contribucao e o trade-off (bits x transicoes x memoria), nao a acuracia.

## Ressalvas

- O benchmark NAO deve concluir 'a Rede B e superior' sem evidencia.
- Tempo de treino com loops Python por amostra nao e representativo de implementacao otimizada.
- Hardware/empacotamento: avaliar somente depois de existir um modelo treinado confiavel.
