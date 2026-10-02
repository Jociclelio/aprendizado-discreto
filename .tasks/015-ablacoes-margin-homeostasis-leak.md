# [015] Ablações — margem, homeostase e leak sobre o B1 corrigido

## Objetivo

Avaliar **separadamente e depois combinadas** as melhorias estruturais (perda com margem, homeostase de disparos, contadores leaky) sobre o B1 corrigido (012) e auditado (014), isolando o efeito de cada mecanismo antes de tratar a Rede B como "refinada".

## Contexto

O plano anterior tratava margem + homeostase como uma única melhoria e presumia que seriam necessárias para "convergência". O relatório (seções 8 e 9) corrige isso: cada mecanismo é uma hipótese distinta, o leak já foi testado em 011 sem demonstrar ganho, e a margem/homeostase devem entrar como **ablações** com orçamento e conjunto de seleção fixos. Somente depois, com uma diferença operacional definida, expandir para B2/B3.

Esta task é a **Etapa 4** da sequência recomendada (relatório, seção 9).

## Escopo

- [x] **Ablação do leak** (isolado): reexaminar γ ∈ {0.90, 0.95, 0.99} e shift ∈ {3,4,5} com o B1 corrigido, medindo oscilação **e** desempenho (não apenas magnitude de `|C|`).
- [x] **Ablação da margem** (isolada): perda multiclasse
  `L = max(0, margin + max_{j≠y} z_j − z_y)` sobre `z_out`.
  - O subgradiente já é em relação ao **score**; **não** multiplicar novamente pelo surrogate do degrau de saída.
  - O surrogate continua **nas ocultas** discretas.
  - `margin` ∈ {1, 2, 5}.
- [x] **Ablação da homeostase** (isolada): ajuste do bias para manter a taxa de disparo perto de um alvo.
  - Definir explicitamente **frequência** de ajuste (a cada M batches/epoch), **tolerância** (ε) e **interação com o bias supervisionado** (012).
  - Ajustar **ocultas** e **saída** como hipóteses separadas (não forçar a mesma taxa nas 10 classes).
- [x] **Combinações justificadas**: margem+homeostase, e (se promissor) +leak, com a mesma base.
- [x] **Controles de diagnóstico** (para separar capacidade de representação de dificuldade de otimização):
  - Rede ternária treinada por um método estabelecido com pesos latentes float;
  - Controle que treina apenas a **readout** sobre features ocultas fixa.
- [x] Fixar **orçamento** (mesmo nº de passos/epochs) e **conjunto de seleção** (validação de 014) para todas as ablações.
- [x] Somente se houver uma **diferença operacional definida**, expandir para B2/B3 (ver `[008]`).
- [x] Salvar resultados em `results/exp-015-ablacoes/` e consolidar a melhor configuração em `src/ternary_refined.py`.

## Especificações técnicas

- **Arquitetura**: `784 → 128 → 64 → 10` (invariante).
- **Base**: B1 corrigido (012), avaliado com a API unificada de 014.
- **Margem**: `margin` ∈ {1,2,5}; definida em `z_out`; sem surrogate extra na saída.
- **Homeostase**: `target_rate` (ex.: 0.25) como **hipótese**, não constante; `ε` e frequência explícitos.
- **Leak**: γ e shift de 011, reinterpretados com métricas de oscilação + desempenho.
- **Avaliação**: validação para seleção; teste só ao final (014).
- **Dependências**: API de 012/014; `CounterManager` de 011.

## Critérios de aceitação

- [x] Cada mecanismo testado **isoladamente** e a combinação testada, com mesmo orçamento.
- [x] Métricas por configuração: acurácia (validação/teste), taxa de neurônios **inativos/saturados**, oscilação, Δacurácia vs. base.
  - **Não** exigir literalmente "nenhum neurônio a 0%/100%"; medir a distribuição.
- [x] `margin` não piora a base; homeostase reduz a taxa de colapso (neurônios 0% ou 100%) **sem** sacrificar acurácia.
- [x] `src/ternary_refined.py` consolida a melhor configuração auditável.
- [x] Resultados em `results/exp-015-ablacoes/`.
- [ ] Commit `[015]` com mensagem descritiva.

## Notas de implementação

- Margem e homeostase são **hipóteses**; o critério é "melhora auditável", não "funciona".
- A homeostase **não é** Batch Normalization; não há garantia de que 25% seja ótimo.
- O leak **já foi testado** (011) sem ganho claro; aqui ele entra para confirmar com o B1 corrigido.
- Os controles (latentes/readout) são **diagnósticos**, não substitutos da ideia.
- Se nenhuma ablação melhorar a base, isso é um **resultado válido** — documentar.

## Dependências

- Bloqueado por: `[014]`
- Bloqueia: `[016]`
