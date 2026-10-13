# Rede Neural Ternária com Aprendizado por Contadores Discretos

> **Estado do repositório (revisão 018):** a proposta científica está em
> `docs/ideia.md` (hipóteses, não resultados). O que este repo **efetivamente**
> contém e mede está auditado em `docs/reports/revisao-f5d3a5a.md`, com
> índices em `.tasks/INDEX.md` (tasks × evidências) e `results/INDEX.md`
> (execução × interpretação). Análise externa preservada:
> `astra-relatório-commit-6f47914.md`.
> Leitura curta do estado atual: baseline A ~92–93% de teste; B1/B+ ~9–10%
> (base, leak e `homeo_0.25` congelam após a 1ª epoch — `base [19896,0,0]`,
> leak `~[19915,0,0]`, `homeo_0.25 [57829,0,0]` —; variantes com margem/
> combinações transitam nas 3 epochs, ex. `margin_1 [168146,134101,131680]`,
> mas seguem ~9–10%; resultado negativo documentado, não
> baseline de capacidade). Treinadores B2/B3 independentes **não existem**
> (008 pendente, destino formal em 018); `src/compare.py` **não existe**.

## Resumo conceitual e proposta de experimento em MNIST

---

## Parte 1 — Resumo da ideia (conceitos)

### Representação

- **Pesos**: ternários, `W ∈ {−1, 0, +1}`. Nenhum peso latente de alta precisão existe por trás — o valor ternário *é* o estado real do parâmetro, não uma sombra de um float.
- **Ativação**: função degrau centrada em zero.
- **Bias**: inteiro, `b ∈ [−N, N−1]`, onde `N` é o número de entradas da camada anterior — a faixa que cobre exatamente todos os limiares de decisão distintos possíveis dado que a soma ponderada de `N` entradas bipolares por pesos ternários satura em `[−N, N]`.

### Aprendizado sem gradiente verdadeiro

- **Fake gradient**: substituto triangular, centrado em zero, da derivada (indefinida) da função degrau. É a versão discreta/quantizada de uma técnica já estabelecida na literatura (*Straight-Through Estimator*; *surrogate gradients* em redes de picos). A altura do pico do triângulo é o hiperparâmetro equivalente à taxa de aprendizado.
- **Quantização do fake gradient**: o valor triangular contínuo é aproximado para um pequeno conjunto de inteiros (ex.: `{-2,-1,0,1,2}`, ou, no caso extremo, apenas o sinal `{-1,0,1}` — equivalente a SignSGD).
- **Propagação por camada (regra da cadeia discreta)**: o sinal de erro se propaga de trás para frente, camada por camada, usando quantidades sempre locais:
  - `δ_j^(l) = fake_grad(z_j^(l)) · Σ_k w_jk^(l+1) · δ_k^(l+1)`
  - Incremento do contador de uma conexão: `Δ_ij = δ_j^(l) · x_i^(l−1)`
  - Incremento do contador de um bias: `Δ_j = δ_j^(l)` (sem multiplicar pela entrada — o bias é tratado como peso de uma entrada fantasma sempre igual a 1)
- **Contadores integrativos por conexão**: cada peso e cada bias possui seu próprio contador inteiro `C_ij`, que acumula `Δ_ij` ao longo de várias amostras. O contador representa evidência histórica acumulada de que aquela conexão deveria mudar de estado.

### Regras de atualização (a comparar experimentalmente)

1. **Por neurônio**: quando um neurônio dispara (condição de disparo definida à parte), suas conexões de entrada são candidatas à atualização com base no valor absoluto de seus contadores individuais.
2. **Por camada (Top-K)**: quando a camada atinge sua condição de atualização, seleciona-se o subconjunto das `K` conexões com maior `|C_ij|` na camada inteira, e só essas são atualizadas.
3. **Por rede (Top-K por camada)**: a cada evento de atualização, seleciona-se `K` conexões por camada, totalizando `N × K` atualizações simultâneas na rede (`N` = número de camadas).

### Política de zeragem dos contadores (a comparar)

- Zerar somente os contadores das conexões atualizadas (**error-feedback**: o resíduo não usado continua acumulando) — hipótese principal, sustentada por resultados conhecidos de compressão de gradiente Top-K (Deep Gradient Compression; Error Feedback for SignSGD), que mostram que descartar o resíduo não-selecionado prejudica a convergência.
- Zerar todos os contadores da camada/rede após cada atualização.
- Normalizar os contadores restantes.

### Fluxo geral

```
Forward → erro → fake gradient (quantizado) → δ propagado camada a camada
→ acumulação nos contadores por conexão → regra de seleção (1, 2 ou 3)
→ atualização discreta do peso (W ∈ {−1,0,+1}) → política de zeragem/reset
```

A unidade fundamental de aprendizado é a **conexão individual** (seu contador); o neurônio e a camada só fornecem o *contexto* que decide quando e quais conexões atualizar.

---

## Parte 2 — Proposta de redes a implementar em MNIST

O objetivo desta primeira rodada é uma **comparação qualitativa**: entender se a técnica proposta aprende algo de real (não apenas "funciona em XOR"), como ela se comporta durante o treino, e onde ela diverge do padrão vigente — não ainda produzir um benchmark quantitativo definitivo.

### Rede A — Baseline (técnica vigente)

Referência padrão-ouro, sem nenhuma restrição de discretização, para servir de vara de medir.

| Aspecto | Especificação |
|---|---|
| Arquitetura | MLP simples: `784 → 128 → 64 → 10` |
| Pesos | float32, inicialização padrão (He/Xavier) |
| Ativação (camadas ocultas) | ReLU |
| Saída | softmax + cross-entropy |
| Otimizador | SGD ou Adam, taxa de aprendizado padrão |
| Treinamento | backpropagation clássico, mini-batches |
| Papel no experimento | baseline de referência — acurácia, curva de convergência e uso de memória "de mercado" |

### Rede B — Técnica proposta (pesos ternários + contadores)

Mesma topologia (`784 → 128 → 64 → 10`) para manter a comparação justa em capacidade nominal — só a forma de representar e atualizar os pesos muda.

| Aspecto | Especificação |
|---|---|
| Arquitetura | idêntica à Rede A: `784 → 128 → 64 → 10` |
| Pesos | ternários `{−1, 0, +1}`, sem peso latente |
| Bias | inteiro, `b ∈ [−N, N−1]` por camada |
| Ativação | degrau centrado em zero |
| Erro | mesma função de perda conceitual da Rede A (adaptada a saída discreta — ex. contagem de acerto/erro por classe, ou soma de erro absoluto por neurônio de saída) |
| Fake gradient | triangular centrado em zero, quantizado para inteiros pequenos |
| Propagação | δ por camada, regra da cadeia discreta (ver Parte 1) |
| Contadores | um `C_ij` inteiro por conexão (peso) e um por bias |
| Atualização | três variantes a testar isoladamente: **B1** (por neurônio), **B2** (por camada Top-K), **B3** (por rede Top-K) |
| Reset de contador | três políticas a testar isoladamente dentro de cada variante: zerar só atualizados / zerar tudo / normalizar |

Isso dá **3 (regra de atualização) × 3 (política de reset) = 9 configurações** da Rede B a explorar — não é necessário rodar todas com o mesmo rigor; a ideia é fazer uma varredura qualitativa e identificar quais combinações sequer aprendem, antes de investir em comparação fina.

### O que observar (comparação qualitativa Rede A vs. Rede B)

- **Curva de erro/acurácia ao longo do treino** — a Rede B converge de forma suave (como A) ou "em degraus" com platôs, como vimos no experimento de busca em coordenadas do XOR?
- **Fração de pesos que efetivamente mudam de estado** ao longo do treino (esperado: bem menor que 100% por passo, dado o Top-K e a natureza discreta).
- **Estabilidade da propagação do δ em profundidade** — verificar se o sinal chega às primeiras camadas sem colapsar a zero ou saturar, o equivalente discreto do problema de gradiente desaparecendo/explodindo.
- **Comparação entre B1/B2/B3** — qual regra de seleção aprende mais rápido, qual generaliza melhor no conjunto de teste.
- **Comparação entre políticas de reset dentro de cada variante** — confirmar (ou refutar) a hipótese de que preservar o resíduo (error-feedback) supera zerar tudo.
- **Custo de memória** — bits por peso (ternário ≈ 1.58 bit teórico vs. 32 bits float) e se a acurácia perdida (se houver) compensa essa economia.

### Ordem de implementação sugerida

> **Atualizado (relatório Astra, commit `6f47914`):** a análise identificou defeitos P0/P1 no B1 que impedem interpretar os ~10% registrados. A ordem foi reorganizada para **reparar antes de refinar** — ver a seção de Tasks abaixo (012→016).
> **Atualizado (revisão 018, commit `f5d3a5a` auditado):** 012–017 executadas; B1/B+ seguem ~9–10% — base, 6× leak e `homeo_0.25` congelam após a 1ª epoch (`base [19896,0,0]`, leak `~[19915,0,0]`, `homeo_0.25 [57829,0,0]`), enquanto variantes com margem/combinações transitam nas 3 epochs (ex. `margin_1 [168146,134101,131680]`) mas seguem ~9–10%, sem sucesso científico (013/014/015/016: resultados negativos/marginais, não superioridade). 018 reconcilia docs e fila sem código novo. Próximo: 019 (testes + ambiente) → 024 (diagnóstico como marco) → 025/026/027 condicionadas.

1. Rede A treinada e validada primeiro, como referência fixa de acurácia/curva de convergência.
2. Rede B com uma única configuração inicial (ex.: B1 + reset "zera só atualizados") para validar que o mecanismo básico aprende algo em MNIST (não só em XOR).
3. **Reparar o B1** (contrato matemático + P0/P1) e provar aprendizado num subconjunto pequeno.
4. Ensaio MNIST auditável (proveniência, validação, ≥3 sementes).
5. Ablações (margem, homeostase, leak) e benchmark final de eficiência.

---

## Estrutura do projeto

```
ideia/
├── README.md              ← este arquivo (estado real + limitações)
├── AGENTS.md              ← convenções para agentes de IA neste repositório
├── docs/
│   ├── ideia.md            ← proposta científica (hipóteses, não resultados)
│   └── reports/
│       └── revisao-f5d3a5a.md ← auditoria 018 (execução × evidência × resultado)
├── astra-relatório-commit-6f47914.md ← análise externa preservada (não reescrita)
├── .tasks/
│   ├── task.md            ← padrão de escrita de tasks
│   ├── INDEX.md           ← índice auditado 001–018 (texto das tasks intacto)
│   ├── 001-baseline-mlp.md
│   ├── 002-utils-and-data.md
│   ├── 003-fake-gradient.md
│   ├── 004-ternary-network.md
│   ├── 005-counter-rules.md   ← primitivas B1/B2/B3 (treinadores B2/B3 integrados: não existem)
│   ├── 006-reset-policies.md
│   ├── 007-b1-initial.md       ← histórico: declara 20 epochs, evidência ausente (ver revisao-f5d3a5a §3.1)
│   ├── 008-b2-b3-variants.md  ← PENDENTE, substituída por 020/021/025 (ver INDEX)
│   ├── 009-comparison.md       ← PENDENTE, substituída por 026/027 (ver INDEX)
│   ├── 010-adaptive-radius-and-data-normalization.md
│   ├── 011-leaky-counters-momentum.md
│   ├── 012-repara-b1-contrato-matematico.md
│   ├── 013-prova-aprendizado-subconjunto.md  ← resultado negativo explícito
│   ├── 014-ensaio-mnist-auditable.md
│   ├── 015-ablacoes-margin-homeostasis-leak.md
│   ├── 016-benchmark-eficiencia.md
│   └── 017-playground-viz.md
├── src/
│   ├── baseline.py        ← Rede A (MLP float32 + backprop)
│   ├── ternary_net.py     ← Rede B (pesos ternários + contadores)
│   ├── ternary_b1.py      ← B1: por neurônio + error-feedback (CONTRATO_012 na docstring)
│   ├── ternary_refined.py ← Rede B refinada (melhor config de 015; ganho marginal)
│   ├── counter_rules.py   ← Regras de atualização (primitivas B1, B2, B3) + CounterManager
│   ├── reset_policies.py  ← Políticas de zeragem de contadores
│   ├── fake_gradient.py   ← Função fake gradient triangular + quantização
│   ├── ablation.py        ← Runner das ablações (015)
│   ├── playground/      ← Interface web Streamlit (017: stepper amostra-a-amostra B1/B2/B3)
│   │   ├── app.py       ← UI (stepper, inspetor, runner, comparador)
│   │   ├── tracer.py    ← instrumentação bit-idêntica ao treino (só observa; B2/B3 só como máscara)
│   │   ├── state.py     ← session_state + cache MNIST
│   │   └── plots.py     ← heatmaps/histogramas (matplotlib)
│   └── utils.py           ← Utilitários (MNIST loader, plotting, etc.)
├── experiments/
│   ├── run_small_subset.py  ← 013 (subconjunto fixo + diagnóstico)
│   ├── run_mnist_full.py    ← 014 (proveniência + 3 sementes)
│   ├── run_ablations.py     ← 015 (13 configs)
│   └── run_benchmark.py     ← 016 (A vs B1 vs B+)
└── results/
    ├── INDEX.md           ← execução × interpretação por experimento (+ órfãos)
    └── exp-NNN-*/         ← gráficos, logs, métricas (008/009: dirs vazios = não executados)
```

## Tasks (ordem de execução)

| # | Task | Descrição | Bloqueia |
|---|------|-----------|----------|
| 001 | `001-baseline-mlp.md` | MLP baseline `784→128→64→10` em MNIST (≥97% acurácia) | 002 |
| 002 | `002-utils-and-data.md` | Utilitários: loader MNIST, plotting, salvamento | 003, 004 |
| 003 | `003-fake-gradient.md` | Fake gradient triangular + quantização | 004, 005, 006 |
| 004 | `004-ternary-network.md` | Núcleo Rede B: pesos ternários, bias, degrau | 005, 006, 007 |
| 005 | `005-counter-rules.md` | Regras de atualização: B1, B2, B3 | 007, 008 |
| 006 | `006-reset-policies.md` | Políticas de reset: error-feedback, full, normalize | 007, 008 |
| 007 | `007-b1-initial.md` | B1 + error-feedback — valida aprendizado em MNIST | 008 |
| 008 | `008-b2-b3-variants.md` | B2 e B3 × 3 políticas de reset (6 configs) | 009 |
| 009 | `009-comparison.md` | Comparação qualitativa final: A vs. todas as B originais | 010 |
| 010 | `010-adaptive-radius-and-data-normalization.md` | Normalização de dados e fake gradient adaptativo ao fan-in | 011 |
| 011 | `011-leaky-counters-momentum.md` | Contadores com decaimento temporal (leaky momentum) | 012 |
| 012 | `012-repara-b1-contrato-matematico.md` | **Repara B1**: contrato matemático + correções P0/P1 do relatório Astra | 013 |
| 013 | `013-prova-aprendizado-subconjunto.md` | Prova de aprendizado em subconjunto pequeno (diagnóstico) | 014 |
| 014 | `014-ensaio-mnist-auditable.md` | Ensaio MNIST auditável (proveniência, validação, ≥3 sementes) | 015 |
| 015 | `015-ablacoes-margin-homeostasis-leak.md` | Ablações: margem, homeostase e leak sobre o B1 corrigido | 016 |
| 016 | `016-benchmark-eficiencia.md` | Benchmark final + validação da proposta de eficiência | — |
| 017 | `017-playground-viz.md` | Playground Streamlit: stepper amostra-a-amostra B1/B2/B3 sobre MNIST | — |
| 018 | `.agents/tasks/proxima-rodada/018-organiza-documentacao-e-fila.md` | Reconciliação 018: `docs/reports/revisao-f5d3a5a.md`, `docs/ideia.md`, `.tasks/INDEX.md`, `results/INDEX.md`, README fiel; destino formal de 008/009 | 019 |

> 008/009 seguem **pendentes e substituídas** (textos preservados): 008 → 020/021/025,
> 009 → 026/027 (detalhes em `.tasks/INDEX.md` e `docs/reports/revisao-f5d3a5a.md` §5).
> A fila 019–027 (contrato versionado, passo único, protocolo, diagnóstico 024 como marco,
> reavaliações condicionadas, consolidação) está em `.agents/plans/proxima-rodada/plan.md`.

## Dependências

- Python 3.10+
- PyTorch (baseline e utilitários)
- NumPy
- Matplotlib (plotting)
- scikit-learn (métricas)
- Streamlit (só para o playground 017) — ver `requirements.txt`

## Playground (017)

Stepper interativo amostra a amostra do treino B1/B2/B3 sobre dígitos
reais do MNIST — mesma matemática do CLI (CONTRATO_012, bit-idêntica):

```bash
pip install -r requirements.txt
streamlit run src/playground/app.py
```

Abas: **Stepper** (7 estágios `z → erro → fake → q_fake → deltas → máscara →
update/reset`), **Inspetor de camada** (heatmaps amostrados, histograma de `z`
com cone do raio adaptativo, distribuição +1/0/-1), **Runner curto**
(cap 2000 amostras; full MNIST fica no CLI) e **Comparador B1/B2/B3**
(mesmo seed + mesmo subset, curvas sobrepostas).

## Licença

Este repositório é dedicado à pesquisa e experimentação acadêmica.

