# [001] Implementa MLP baseline com PyTorch para MNIST

## Objetivo

Treinar um MLP `784 → 128 → 64 → 10` com ReLU e softmax em MNIST, alcançando acurácia de teste ≥ 97%, como referência fixa para a Rede B.

## Contexto

A Rede A é a vara de medir de toda a comparação qualitativa. Sem uma baseline sólida, não há como avaliar se a técnica ternária com contadores aprende algo útil. Esta task estabelece o ponto de partida.

## Escopo

- [ ] Criar `src/baseline.py` — MLP float32 com PyTorch
- [ ] Loader de MNIST (usar `torchvision.datasets.MNIST`)
- [ ] Treino com Adam ou SGD, mini-batches
- [ ] Avaliação no conjunto de teste
- [ ] Salvar curva de acurácia em `results/exp-001-baseline/`
- [ ] Acurácia de teste ≥ 97%

## Especificações técnicas

- **Arquitetura**: `nn.Linear(784, 128)` → ReLU → `nn.Linear(128, 64)` → ReLU → `nn.Linear(64, 10)` → softmax
- **Pesos**: inicialização He (padrão do PyTorch para ReLU)
- **Otimizador**: Adam (lr=1e-3) ou SGD (lr=0.01, momentum=0.9)
- **Perda**: CrossEntropyLoss
- **Batch size**: 128
- **Epochs**: 20 (ou até convergência clara)
- **Split**: 60k treino / 10k teste (padrão MNIST)

## Critérios de aceitação

- [ ] Código implementado em `src/baseline.py`
- [ ] Executa sem erro: `python src/baseline.py`
- [ ] Acurácia de teste ≥ 97%
- [ ] Curva de acurácia salva em `results/exp-001-baseline/accuracy_curve.csv`
- [ ] Commit [001] com mensagem descritiva
- [ ] Resultados salvos em `results/exp-001-baseline/`

## Notas de implementação

- Manter o código simples e legível — não precisa de abstrações complexas aqui
- Salvar checkpoints se possível (para comparar depois com a Rede B)
- Plotar curva de treino no final (acurácia por epoch)
- Usar `torchvision` para download automático do MNIST

## Dependências

- Bloqueado por: nenhuma
- Bloqueia: `[002]`
