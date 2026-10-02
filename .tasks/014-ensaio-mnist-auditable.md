# [014] Ensaio MNIST auditável (proveniência e avaliação limpa)

## Objetivo

Executar o B1 corrigido no MNIST completo com **proveniência total** e **avaliação limpa** (validação para seleção, teste só no fim, múltiplas sementes, unidades padronizadas), produzindo um ensaio que possa ser recriado e auditado por qualquer agente.

## Contexto

Os experimentos anteriores (007, 010, 011) usavam comandos ad hoc, escolhiam checkpoint pelo conjunto de teste e misturavam unidades (% x fração). O relatório (seção 9, Etapa 3) exige que cada experimento seja recriável por script versionado e que o conjunto de teste seja reservado para avaliação final. Só com um ensaio auditável os números de MNIST deixam de ser "registrados historicamente" e passam a ser evidência.

Esta task é a **Etapa 3** da sequência recomendada (relatório, seção 9).

## Escopo

- [ ] Criar `experiments/run_mnist_full.py` (script versionado) que recria o experimento a partir de uma configuração serializável (JSON).
- [ ] **Proveniência** (gravar em `results/exp-014-mnist-full/provenance.json`):
  - commit (`git rev-parse HEAD`), estado local (`git status --porcelain`);
  - versões de `numpy`, `torch`, `torchvision`;
  - seed, índices/splits (treino/validação/teste), configuração completa, tempo de execução.
- [ ] **Split 3-vias**: treino / **validação** / teste. Usar validação para escolher configuração e/ou checkpoint; **reservar teste para a avaliação final única**.
- [ ] **Múltiplas sementes**: ao menos **3 sementes** na rodada; registrar **cada** execução (não só a melhor).
- [ ] **Unidades padronizadas**: definir uma unidade canônica (ex.: acurácia em fração [0,1]) e converter; documentar.
- [ ] **Controle de preprocessing**: registrar explicitamente que a baseline usa padronização MNIST (0.1307/0.3081) e a Rede B usa pixels `[0,1]`; discutir se isso isola o efeito da quantização (não isola, dado que também mudam ativação/perda).
- [ ] Curvas de treino/validação/teste em `results/exp-014-mnist-full/` (PNG + CSV).
- [ ] Matriz de confusão final (no teste, única vez) e distribuição de predições.

## Especificações técnicas

- **Arquitetura**: `784 → 128 → 64 → 10` (invariante).
- **Split**: 60k treino → (ex.: 50k treino / 10k validação) e 10k teste (ou proporção acordada, gravada).
- **Sementes**: ≥ 3 (ex.: 42, 123, 999).
- **Avaliação**: usar `forward_with_scores` + `argmax` (API de 012); **não** reavaliar o teste a cada epoch.
- **Configuração**: `scale`, `radius`/`scale_factor`, `levels`, `batch_size`, `epochs`, `decay_*`, política de reset — tudo no JSON de proveniência.
- **Dependências**: API unificada de 012 e o subconjunto/controle de 013 (mesma API de diagnóstico).

## Critérios de aceitação

- [ ] Script implementado em `experiments/run_mnist_full.py` e versionado.
- [ ] `provenance.json` completo e suficiente para recriar o experimento.
- [ ] Teste avaliado **uma única vez** por execução (no fim).
- [ ] ≥ 3 sementes, todas registradas.
- [ ] Curvas e matriz de confusão salvas em `results/exp-014-mnist-full/`.
- [ ] Commit `[014]` com mensagem descritiva.

## Notas de implementação

- Este ensaio mede o **B1 corrigido** (sem margem/homeostase); as ablações vêm em 015.
- Se a acurácia seguir perto do preditor majoritário, o próximo passo é reexaminar 013 (diagnóstico), **não** adicionar mecanismos.
- A comparação com a baseline (001) deve usar o **mesmo** split de teste e unidade; o checkpoint da baseline deve ser escolhido por validação (reavaliar o checkpoint do 001 se necessário).
- Manter o custo baixo: o objetivo é auditabilidade, não o melhor número.

## Dependências

- Bloqueado por: `[012]`, `[013]`
- Bloqueia: `[015]`
