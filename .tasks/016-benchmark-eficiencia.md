# [016] Benchmark final e validação da proposta de eficiência

## Objetivo

Produzir um benchmark consolidado (Baseline A vs. B1 corrigido vs. B+ refinada) que **avalie se há ganho** — sem pressupô-lo — e validar a proposta de eficiência (memória, bits por parâmetro, frequência de transição, tempo), só então discutir hardware/empacotamento.

## Contexto

O plano anterior (013) presumia "ganho de desempenho obtido" e planejava registrar "zero MACs float" para a Rede B. O relatório (seções 7 e 8) corrige isso: (a) o benchmark deve **avaliar** se há ganho, não assumi-lo; (b) "zero MACs float" só é defensável para uma implementação **especializada**, não para o código NumPy atual; (c) a economia de memória dos parâmetros (≈4×) não é a mesma conta do estado de treinamento (parâmetros + contadores int64 ≈ 2,25×).

Esta task é a **Etapa 5** da sequência recomendada (relatório, seção 9).

## Escopo

- [ ] Benchmark consolidado em `experiments/run_benchmark.py` (ou `src/compare.py`) comparando, com **mesmo orçamento e split** (014):
  1. **Rede A** — baseline float32 (001), checkpoint escolhido por **validação**;
  2. **Rede B Original** — B1 corrigido (012);
  3. **Rede B Refinada (B+)** — melhor configuração de 015 (`src/ternary_refined.py`).
- [ ] Métricas por rede:
  - acurácia (validação e teste final);
  - época de estabilização;
  - **bits por parâmetro** armazenados (A: 32; B pesos: 1 byte int8, ~1,58 bit teórico empacotado; contadores: 8 bytes int64);
  - **frequência de transição** de pesos por epoch (transições efetivas, não tentativas);
  - memória de parâmetros x estado de treinamento (contagem da seção 7 do relatório);
  - tempo de treino (com a ressalva de loops Python/amostra).
- [ ] **Largura de contador**: ensaiar contadores de 8/16 bits com **política de saturação/overflow** explícita; registrar o efeito na acurácia.
- [ ] Gráficos multi-painel em `results/exp-016-benchmark/`:
  - acurácia (treino/validação/teste);
  - perda;
  - taxa de transição por epoch;
  - distribuição de disparos/ativações.
- [ ] **Tabela consolidada + relatório de conclusões** em `results/exp-016-benchmark/README.md`:
  - onde há ganho, onde há perda, trade-offs;
  - o que é defensável (memória de inferência) x o que **não** é (MACs zero, velocidade) sem implementação especializada.
- [ ] **Hardware/empacotamento**: avaliar **apenas depois** de existir um modelo treinado confiável; registrar como etapa futura, não como resultado.

## Especificações técnicas

- **Arquitetura**: `784 → 128 → 64 → 10` (invariante nas três redes).
- **Topologia**: 109.184 pesos + 202 bias = 109.386 parâmetros (base das contas de memória).
- **Memória (seção 7 do relatório)**: baseline 437.544 B; B pesos+bias 109.992 B; B +contadores 985.080 B.
- **Sementes**: ≥ 3 (014); registrar cada execução.
- **Unidades**: padronizadas (014).
- **Dependências**: 001, 012, 014, 015.

## Critérios de aceitação

- [ ] Script de benchmark executado com sucesso e versionado.
- [ ] Tabela consolidada e relatório de conclusões (com trade-offs) em `results/exp-016-benchmark/README.md`.
- [ ] Métricas de eficiência (bits, transições, memória, tempo) reportadas com as ressalvas do relatório.
- [ ] Contadores 8/16 bits ensaiados com política de saturação.
- [ ] Gráficos em `results/exp-016-benchmark/`.
- [ ] Commit `[016]` com mensagem descritiva.

## Notas de implementação

- O benchmark **não deve** concluir "a Rede B é superior" sem evidência; pode concluir "a Rede B perde X pontos por Y de bits economizados".
- "Zero MACs float" é propriedade de uma implementação a construir, **não** do código atual.
- A comparação de memória de treinamento é incompleta sem considerar momentos do Adam e temporários; registrar essa limitação.
- Se a Rede B não alcançar a baseline, a contribuição é o **trade-off** (bits × transições × memória), não a acurácia.

## Dependências

- Bloqueado por: `[015]`
- Bloqueia: nenhuma (fecha a rodada)
