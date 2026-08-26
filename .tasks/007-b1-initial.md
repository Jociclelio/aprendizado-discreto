# [007] Configuração inicial B1 + error-feedback — valida aprendizado em MNIST

## Objetivo

Implementar e treinar a primeira configuração da Rede B: **B1 (por neurônio) + reset error-feedback**. Validar que o mecanismo básico aprende algo real em MNIST (não apenas em XOR).

## Contexto

Esta é a validação mais crítica: se a técnica proposta não aprende nada em MNIST, toda a abordagem está comprometida. Não é necessário bom desempenho — é necessário *sinal de aprendizado*. Acurácia > random (10%) já seria um sinal positivo nesta fase.

## Escopo

- [ ] Criar `src/ternary_b1.py` — integra `ternary_net.py`, `counter_rules.py`, `reset_policies.py`, `fake_gradient.py`
- [ ] Loop de treino:
  - Forward pass com degrau
  - Cálculo de erro (soma de erro absoluto por neurônio de saída)
  - Fake gradient triangular quantizado
  - Propagação de δ camada a camada
  - Acumulação nos contadores
  - Seleção B1 (por neurônio disparado)
  - Atualização de pesos
  - Reset error-feedback
- [ ] Avaliação periódica (a cada epoch) no conjunto de teste
- [ ] Salvar métricas e curvas em `results/exp-007-b1-error-feedback/`
- [ ] Mínimo: 20 epochs

## Especificações técnicas

- **Hiperparâmetros iniciais**:
  - `scale` (fake gradient): 1.0
  - `radius` (fake gradient): 1.0
  - `levels` (quantização): 5 → `{-2, -1, 0, 1, 2}`
  - `K` (B1 threshold): usar threshold dinâmico = `max(1, mean(|C|))`
  - `batch_size`: 32 (amostra por amostra ou mini-batch)
  - `epochs`: 20 mínimo
- **Perda**: soma de erro absoluto por neurônio de saída (adaptado para degrau)
  - `loss = Σ_i |target_one_hot[i] - output[i]|`
  - Ou: contagem de classes erradas
- **Avaliação**: acurácia = % de amostras com pelo menos um neurônio de saída correto
  - Para MNIST com degrau, a métrica é: argmax do output binário vs argmax do target
  - Se múltiplos neurônios ativados: argmax do soma ponderada antes do degrau? Ou simplesmente maior soma?
  - Decisão: usar `argmax(output)` onde output é o valor antes do degrau (ou soma de pesos ativados)
- **Input**: amostra por amostra (online) ou mini-batch de 32

## Critérios de aceitação

- [ ] Código implementado em `src/ternary_b1.py`
- [ ] Executa sem erro por 20 epochs
- [ ] Acurácia de teste melhora ao longo do treino (sinal de aprendizado)
- [ ] Curva de acurácia salva em `results/exp-007-b1-error-feedback/`
- [ ] Commit [007] com mensagem descritiva
- [ ] Resultados salvos em `results/exp-007-b1-error-feedback/`

## Notas de implementação

- **Métrica de avaliação**: com saída degrau (binária por neurônio), a classificação é ambígua se múltiplos neurônios disparam. Sugerir:
  - Opção A: `argmax` do pre-activation (z antes do degrau) — mais estável
  - Opção B: contar neurônios ativados e escolher o mais ativado
  - Opção C: se nenhum/ne poucos ativam, escolher aleatoriamente
- **Online vs mini-batch**: começar com amostra por amostra (mais natural para contadores)
- **Monitorar**: fração de pesos que mudam por epoch, magnitude dos contadores, estabilidade do δ
- **Não otimizar hiperparâmetros aqui** — só validar que aprende. Ajustes vêm depois.

## Dependências

- Bloqueado por: `[001]`, `[002]`, `[003]`, `[004]`, `[005]`, `[006]`
- Bloqueia: `[008]`
