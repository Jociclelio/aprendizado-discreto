# [002] Cria utilitários compartilhados e loader de MNIST

## Objetivo

Implementar `src/utils.py` com funções compartilhadas: loader de MNIST, plotting de curvas de acurácia, salvamento de resultados, e qualquer helper reutilizável pelas Redes A e B.

## Contexto

Tanto a baseline quanto a Rede B precisam de MNIST, plotting, e salvamento de resultados. Centralizar isso evita duplicação e garante consistência nos gráficos comparativos.

## Escopo

- [ ] Criar `src/utils.py`
- [ ] Função `load_mnist()` — retorna train_data, train_labels, test_data, test_labels como numpy arrays (normalizados [0,1])
- [ ] Função `plot_accuracy_curve(epochs, train_acc, test_acc, title, save_path)` — gera gráfico lado a lado treino/teste
- [ ] Função `save_metrics(metrics_dict, save_path)` — salva JSON com métricas do experimento
- [ ] Função `count_ternary_weights(model)` — helper para contar pesos em {-1,0,+1} (útil para a Rede B)
- [ ] Testar com `python -c "from src.utils import load_mnist; ..."`

## Especificações técnicas

- **Input normalizado**: pixels de MNIST normalizados para [0, 1] (float32)
- **Formato de saída**: numpy arrays, shape `(N, 784)` para imagens, `(N,)` para labels
- **Plotting**: matplotlib, salvar como PNG, incluir em `results/exp-NNN-nome/`
- **Métricas**: salvar como JSON com timestamp, acurácia treino/teste final, etc.

## Critérios de aceitação

- [ ] Código implementado em `src/utils.py`
- [ ] `load_mnist()` retorna arrays corretos (train: 60k, test: 10k)
- [ ] `plot_accuracy_curve()` gera PNG funcional
- [ ] `save_metrics()` salva JSON válido
- [ ] Executa sem erro
- [ ] Commit [002] com mensagem descritiva

## Notas de implementação

- `load_mnist()` deve normalizar para [0,1] e flatten (784,) — não precisa de tensor PyTorch aqui
- `count_ternary_weights()` será usada pela Rede B para métricas de sparsity
- Manter funções puras (sem estado global)

## Dependências

- Bloqueado por: nenhuma
- Bloqueia: `[003]`, `[004]`
