# Relatório Astra — análise do repositório e da ideia

**Referência:** `6f47914153813defdd56d096ffe72f9190eb3ae4`

**Data da análise:** 16 de setembro de 2026

**Público:** autor do projeto e agentes que continuarão o trabalho.

## 1. Parecer executivo

**A ideia merece investigação, mas o repositório ainda não fornece um teste confiável de sua capacidade de aprender MNIST.** Há uma proposta coerente — pesos realmente ternários, evidência acumulada por conexão e atualizações discretas — e um baseline funcional. Entretanto, o treinador B1 contém erros que afetam diretamente a direção do aprendizado e a avaliação. Os resultados negativos atuais não permitem separar limitações da proposta de defeitos da implementação.

Minha recomendação principal é **corrigir e validar o mecanismo básico antes de adicionar margem, homeostase ou executar a grade de variantes**. Caso contrário, novas heurísticas poderão mascarar erros anteriores, e será difícil saber o que realmente funcionou.

Resumo das conclusões:

| Questão | Conclusão |
|---|---|
| A proposta é conceitualmente plausível? | Sim, como treinamento com surrogate gradients e estado discreto de otimização. |
| Os pesos têm cópias latentes float? | Não encontrei cópias latentes usadas pelo treinador; os pesos são `int8`. |
| O treinamento é totalmente inteiro? | Não. Há ativações e cálculos intermediários em ponto flutuante. |
| A Rede B demonstrou aprendizado? | Os artefatos examinados não demonstram aprendizado convincente; as acurácias estão aproximadamente entre 9% e 11%. |
| Isso refuta a ideia? | Não: há erros de implementação suficientes para tornar essa conclusão prematura. |
| Há vantagem de memória? | Nos parâmetros de inferência armazenados, sim, cerca de 4× frente a float32. Com contadores `int64`, o estado persistente de treinamento é maior que os parâmetros float32 isolados. |
| Há vantagem de velocidade/energia comprovada? | Não há benchmark ou implementação especializada que a demonstre. |
| A combinação é inédita? | Não estabelecido. Seria necessária revisão bibliográfica específica. |

## 2. Escopo e rastreabilidade

Foram examinados `README.md`, as convenções do repositório, os módulos em `src/`, o histórico recente, as tasks relevantes e os resultados tabulares em `results/`. Também foram executados testes existentes e pequenas reproduções determinísticas, sem novo treinamento completo de MNIST.

No início da análise, os arquivos versionados não tinham modificações locais. Existiam estes itens não rastreados:

- `.tasks/012-margin-loss-and-homeostasis.md`;
- `.tasks/013-refined-comparison-and-benchmarks.md`;
- `results/accuracy_curve_default.png`;
- `results/test-002/`.

As tasks 012 e 013 foram lidas como **planejamento local**, não como conteúdo do commit analisado. Os dois últimos itens não foram usados como evidência de desempenho. O checkpoint do baseline existe no workspace, mas não aparece em `git ls-files`; a acurácia citada aqui vem do CSV versionado, não de uma reavaliação desse checkpoint.

Este documento é uma análise, não uma implementação de task nem uma declaração de conclusão das tasks existentes. O hash no nome identifica o estado analisado; não representa um novo commit do relatório. Nenhum código de treinamento foi alterado nesta revisão.

## 3. O que há de interessante na ideia

### 3.1 O núcleo da proposta

A distinção relevante é entre **o parâmetro usado pela rede** e **o estado que acumula evidência para alterá-lo**:

- `W ∈ {−1, 0, +1}` é o parâmetro real;
- cada conexão mantém um contador `C`;
- pequenas evidências podem se somar ou cancelar antes de uma transição;
- a seleção limita quais conexões mudam em cada evento.

Isso oferece uma forma de histerese: não é necessário reagir a cada sinal ruidoso com uma mudança imediata do peso. Também permite estudar aprendizado com orçamento limitado de escrita em memória e separar o custo de treinamento do custo de inferência.

Os contadores, porém, continuam sendo **estado latente de otimização**. Não serem pesos float ocultos é uma propriedade importante, mas não significa que o sistema tenha apenas três estados por conexão durante o aprendizado. Com `int64`, há muito mais informação no otimizador que no próprio peso.

### 3.2 Relação com métodos conhecidos

A propagação de erro continua sendo backpropagation com uma derivada substituta. A descrição mais precisa é “sem derivada verdadeira do degrau”, e não “sem gradientes” ou “sem backpropagation”. STE, surrogate gradients, atualização pelo sinal e acumulação de resíduos são famílias relevantes para comparar a proposta.

O ponto potencialmente distintivo é a combinação e seu regime operacional: pesos ternários sem cópia float, contadores de precisão limitada, política de disparo/seleção e tratamento do resíduo. Este relatório não fez uma revisão externa da literatura e não atribui originalidade científica à combinação.

### 3.3 Questões conceituais que precisam ser explícitas

1. **O contador acumula gradiente ou direção de descida?** Ambas as convenções são válidas, mas a atualização precisa usar o sinal correspondente de maneira uniforme.
2. **O que dispara B1?** Ativação do neurônio, magnitude do erro, cruzamento de limiar do contador ou uma combinação? Bloquear toda atualização de neurônios inativos pode dificultar recuperá-los.
3. **Qual é a unidade temporal?** No código, o peso muda por amostra e o leak pode ocorrer por batch. Alterar `batch_size` altera a memória temporal efetiva mesmo sem transformar o treino em mini-batch vetorizado.
4. **Qual é a representação das ativações?** O README menciona entradas bipolares na justificativa do bias, mas o código usa ocultas em `{0,1}` e pixels em `[0,1]`.
5. **Qual é o tratamento de saturação?** Selecionar `W=+1` para subir não produz mudança. É preciso decidir se esse evento consome o contador e registrá-lo separadamente de uma transição real.

### 3.4 Preservação parcial de resíduo não é automaticamente error-feedback clássico

Manter contadores não selecionados é uma hipótese sensata. Contudo, zerar o contador selecionado depois de aplicar apenas `±1` pode descartar excesso de evidência; se o peso satura, pode descartar evidência sem realizar transição alguma.

Em uma formulação clássica de error-feedback, o resíduo é a diferença entre a atualização pretendida e a efetivamente transmitida/aplicada, com unidades compatíveis. Portanto, resultados teóricos de compressão de gradientes não se transferem automaticamente para esta regra.

Sugestão: chamar a regra atual de **reset seletivo com preservação dos não selecionados** e comparar, posteriormente, com uma regra de subtração de limiar que preserve o excesso. Nenhuma delas deve ser considerada superior antes do experimento.

## 4. Estado efetivo do repositório

| Componente | Situação observada |
|---|---|
| Baseline | Implementado em `src/baseline.py`, com curva de 20 épocas versionada. |
| Rede ternária | Forward, pesos ternários, bias inteiro e funções de perda/métrica em `src/ternary_net.py`. |
| Surrogate e contadores | Implementados em `src/fake_gradient.py` e `src/counter_rules.py`. |
| Regras B1/B2/B3 | Primitivas existentes; o único treinador integrado é B1. |
| Políticas de reset | Implementadas, com problemas descritos abaixo. |
| Treinadores B2/B3 | `src/ternary_b2.py` e `src/ternary_b3.py` não existem. |
| Comparação/refinamento | `src/compare.py` e `src/ternary_refined.py` não existem. |
| Scripts de experimento | `experiments/` está vazio. |
| Dependências e testes | Não há manifesto de dependências nem suíte separada em `tests/`; há testes nos blocos `__main__`. |

**Pontos positivos:** módulos pequenos, separação razoável de responsabilidades, sementes explícitas, referências às tasks nos arquivos e artefatos versionados para parte dos experimentos. Isso facilita recuperar a intenção e localizar discrepâncias.

**Problema documental:** a árvore do README mistura arquivos existentes e planejados. A task 007 marca treino de 20 épocas e aprendizado como aceitos, mas relata somente 50 amostras por duas épocas com aproximadamente 9–10%; seu diretório de resultados não está presente. Um valor padrão `epochs=20` não comprova uma execução de 20 épocas.

As tasks 008/009 continuam pendentes, embora 010/011 tenham commits posteriores. A ordem efetiva diverge da sequência prevista em `AGENTS.md`. A task 011 também marca redução de oscilação como aceita, embora o relato mostre ausência de mudanças após a primeira época com e sem leak. Isso não estabelece benefício do leak.

## 5. Achados técnicos por prioridade

As referências de linha abaixo se referem ao commit analisado.

### P0 — corrigir antes de interpretar novos resultados

#### A. A avaliação usa uma rede diferente da treinada

**Local:** `src/ternary_b1.py:349–355`.

Na recomputação dos scores, aplica-se o degrau depois da primeira camada, mas não depois da segunda. A saída é calculada a partir de `z_hidden2`, quando deveria usar `step(z_hidden2)`.

Isso altera a função avaliada. O `net.forward()` executado antes já calcula o caminho correto, mas seu resultado é ignorado para a classificação.

**Reprodução confirmada:** uma rede pequena com segunda pré-ativação `[2,1]` e pesos de saída `[[1,-1],[0,1]]` produz scores corretos `[0,1]`; a avaliação defeituosa produz `[1,1]`. Para o alvo 1, `evaluate()` retorna 0% em vez de 100%.

**Correção recomendada:** uma única implementação de forward que exponha tanto scores finais quanto saídas binárias; avaliação deve consumir esses scores sem reimplementar as camadas.

#### B. O sinal da atualização de saída pode ser contrário ao desejado

**Local:** `src/ternary_b1.py:139–155, 195–208`.

O contador já recebe o sinal de `prediction − target`. Depois, a atualização multiplica `sign(C)` por `sign(−error)` da amostra corrente. Há uma segunda combinação de sinais que não corresponde a descida consistente sobre a evidência acumulada.

**Reprodução confirmada:** com peso de saída zero, entrada oculta ativa e classe correta ainda inativa, o contador da classe correta recebe `−2`, mas o peso passa de `0` para `−1`. A ação esperada, nesse caso controlado, seria aumentar o score da classe correta.

**Correção recomendada:** se `C` acumula gradiente, usar `W ← clip(W − sign(C))`; se acumula direção de descida, usar soma. A convenção deve valer para todas as camadas, sem multiplicação extra pelo sinal do erro corrente.

#### C. As camadas ocultas reutilizam o delta da camada errada

**Local:** `src/ternary_b1.py:157–174, 213–221`.

`delta` é sobrescrito durante o backward e termina com dimensão da primeira oculta. Esse mesmo vetor é usado para construir os sinais de atualização de ambas as ocultas. Na arquitetura padrão, os 64 neurônios da segunda oculta recebem os primeiros 64 sinais de um vetor de 128 neurônios de outra camada.

Isso não gera necessariamente erro de dimensão e pode passar despercebido. É erro semântico, agravado pela multiplicação adicional com `sign(C)`.

**Correção recomendada:** guardar `deltas[layer_idx]` explicitamente e separar cálculo do backward da aplicação das mudanças.

#### D. Seleção e reset de B1 não correspondem entre si

**Local:** `src/ternary_b1.py:181–203, 225–251`.

- A atualização percorre todos os neurônios, sem usar a máscara de disparo.
- O limiar usado na atualização é uma média da camada.
- O reset considera somente neurônios disparados e recalcula um limiar por neurônio.

Consequentemente, uma conexão pode ser atualizada sem ter o contador zerado; outra pode ter o contador zerado sem ter sido selecionada na atualização. O B1 integrado também diverge da primitiva `apply_update_neuron()`.

**Correção recomendada:** calcular e guardar uma máscara única de seleção; aplicá-la à atualização e à política de reset. Definir separadamente o que fazer com tentativas bloqueadas pelo clipping.

### P1 — completar o algoritmo e corrigir as variantes

#### E. Bias existe, mas não aprende

**Local:** `src/ternary_b1.py:153–174`; `src/counter_rules.py:95–98, 128–148`.

O treinador não passa deltas de bias para `accumulate()` e não atualiza `layer.bias`. Os contadores de bias existem, mas ficam em zero nesse caminho.

A ideia prevê explicitamente aprendizado supervisionado dos limiares. Homeostase futura não substitui automaticamente esse mecanismo: são duas fontes distintas de atualização, que devem ser avaliadas separadamente.

#### F. A normalização dos contadores aumenta a escala quando deveria reduzi-la

**Local:** `src/reset_policies.py:87–101`.

O código calcula `mean_abs / target_scale` e multiplica por esse fator. Para aproximar a média do alvo, deveria usar o fator inverso, com regra explícita de quantização inteira.

**Reprodução confirmada:** `[10,20,30,40]`, removendo a primeira entrada e pedindo escala 10, vira `[0,60,90,120]`. A média remanescente aumenta de 30 para 90.

Os testes existentes passam porque repetem a fórmula incorreta como expectativa. Deve-se testar a propriedade desejada — magnitude próxima do alvo, dentro do erro de arredondamento — em vez de copiar a implementação.

Além disso, `apply_reset_on_manager_all_layers(..., policy="normalize")`, em `src/reset_policies.py:234–237`, passa arrays vazios float como índices. Foi reproduzido `IndexError`; os índices devem ter dtype inteiro.

#### G. Os dois leaks têm vieses de arredondamento opostos

**Local:** `src/counter_rules.py:100–110`.

Para `C=[−2,−1,0,1,2]`:

| Regra | Resultado confirmado |
|---|---|
| `floor(0.9*C)` | `[−2,−1,0,0,1]` |
| `C − (C >> 4)` | `[−1,0,0,1,2]` |

O primeiro não elimina `−1` inativo; o segundo não elimina pequenos positivos. O comentário que chama o bitshift de simétrico é incorreto. A fórmula multiplicativa implementa o piso solicitado pela task, mas esse contrato não produz esquecimento simétrico.

**Recomendação:** decidir deliberadamente a política de arredondamento; testar simetria de sinal, comportamento perto de zero e pontos fixos. Arredondamento estocástico ou representação de ponto fixo são hipóteses futuras, não correções automaticamente superiores.

#### H. A quantização intermediária não corresponde à descrição geral

**Local:** `src/ternary_b1.py:142–174`; `src/fake_gradient.py:159–189`.

Só o surrogate da saída é explicitamente quantizado com `levels`. Nas ocultas, a propagação usa o triangular contínuo e depois converte os produtos para `int64`, truncando em direção a zero. Produtos menores que 1 desaparecem antes da acumulação: um contador inteiro não consegue integrar uma evidência que já foi descartada.

Isso pode ser especialmente relevante na primeira camada, com pixels fracionários. É necessário medir separadamente o sinal antes e depois da conversão.

O helper `propagate_delta_with_fake()` também faz média do surrogate no eixo do batch. Isso é equivalente ao valor local quando há uma única amostra, como no treinador atual, mas não é a regra da cadeia por amostra em um futuro backward vetorizado.

### P2 — precisão das métricas e robustez

- **Tentativas não são mudanças:** `updated` conta candidatos ou sinais não nulos, incluindo pesos saturados que permanecem iguais. Medir seleção, tentativa, bloqueio e transição efetiva separadamente.
- **Snapshots não medem todas as oscilações:** comparar pesos entre épocas não detecta mudanças que se desfizeram dentro da época. Uma taxa final zero não prova ausência de atividade intermediária.
- **B2/B3 podem ser equivalentes:** `apply_update_network_topk()` apenas chama B2 em cada camada. Com os mesmos contadores, K e agenda de eventos, as operações sobre camadas distintas são equivalentes. Definir agendas diferentes antes de tratá-las como métodos distintos.
- **K=0:** os caminhos com `[-k:]` em `get_top_k_weights()` e `CounterAccumulator.top_k_indices()` selecionam tudo quando `k=0`. Validar K e testar os limites.
- **Contrato inteiro:** `CounterAccumulator.normalize()` transforma seus contadores em float, diferentemente de `CounterManager`.
- **Defaults divergentes:** `DEFAULT_HYPERPARAMS` anuncia raio adaptativo ativado, mas a assinatura de `train_b1()` usa `False`; o dicionário não configura automaticamente a função.
- **Checkpoint incompleto:** `train_b1()` salva somente pesos. Para inferência com bias treinado, é necessário salvar bias; para retomada fiel, também contadores, configuração e estado dos geradores aleatórios.
- **Terminologia da perda:** `ternary_binary_loss()` é L1 médio sobre bits, não equivalente a BCE como diz a docstring. Uma saída toda zero tem L1 de 0,1 em dez classes; loss pequena isoladamente pode esconder colapso.

## 6. O que os resultados realmente sustentam

| Artefato | Resultado registrado | Interpretação permitida |
|---|---|---|
| `results/exp-001-baseline/accuracy_curve.csv` | 20 épocas; teste de 95,35% a 97,97%; treino final 99,6783%. | Baseline com curva consistente e referência útil. |
| `results/exp-010-adaptive-radius/metrics.json` | Acurácias rápidas: 10,0%, 8,9%, 9,9%. | A adaptação de raio não demonstra ganho de classificação nesses registros. |
| Mesmo JSON | Frações não nulas entre aproximadamente 61% e 100%, conforme camada/configuração. | Há registros de maior cobertura de sinal; faltam scripts/metadados para auditar exatamente o estágio medido. |
| `results/exp-011-leaky-counters/metrics.json` | 10,8%, 11,0%, 9,0%, 11,0%; valores constantes nas três épocas por configuração. | Não há evolução de acurácia documentada nesse ensaio. |
| Mesmo experimento, sintético | Acumulação constante limitada em algumas variantes leaky. | Evidência da dinâmica do acumulador sob entrada constante, não de aprendizado de representação. |

As acurácias de B devem ser tratadas como **historicamente registradas e potencialmente afetadas pelo avaliador incorreto**, não como medidas finais do modelo pretendido.

“Maior que 10%” é um critério fraco: MNIST não tem classes exatamente balanceadas, e predizer sempre a classe majoritária pode superar 10%. Comparar com o preditor majoritário do split, a rede inicial e múltiplas sementes; registrar distribuição de predições e matriz de confusão.

As afirmações de que “precisa de margem/homeostase para convergir” são hipóteses, não conclusões demonstradas. Também não basta observar surrogate não nulo: o erro pode ser zero, a soma propagada pode cancelar e a conversão inteira pode eliminar o incremento.

## 7. Memória, custo computacional e hardware

A topologia tem **109.184 pesos e 202 bias**, totalizando 109.386 parâmetros.

Contabilidade dos arrays persistentes, sem overhead dos objetos, caches, dados ou temporários:

| Estado | Bytes |
|---|---:|
| Baseline: parâmetros float32 | 437.544 |
| Rede B: pesos int8 + bias int32 | 109.992 |
| Rede B: contadores int64 de pesos e bias | 875.088 |
| Rede B: parâmetros + contadores | 985.080 |

Logo, a inferência tem aproximadamente **3,98× menos bytes de parâmetros**. Durante treinamento, parâmetros e contadores de B ocupam cerca de **2,25×** os parâmetros float32 isolados. Isso não é uma comparação completa de memória de treinamento: Adam acrescenta momentos e gradientes; ambos os métodos usam temporários e ativações.

`log2(3) ≈ 1,585` é uma referência informacional de codificação ternária, não o armazenamento realizado pelo array `int8`. Um formato fixo simples usaria 2 bits por peso; aproximar-se de 1,585 exige empacotamento em blocos ou outra codificação apropriada. Bias, alinhamento e decodificação também entram na conta.

A inferência ternária pode, em princípio, trocar multiplicações por seleção/soma/subtração. Contudo, o código atual usa `@` com arrays float nas ativações e cálculos NumPy. A task 013 planeja registrar zero MACs float para B; isso só é defensável como propriedade de uma implementação especializada, não como benchmark do código atual.

Atualizações esparsas também não eliminam o forward/backward denso. Loops Python por amostra e por peso podem dominar o tempo. Primeiro demonstrar aprendizado; depois medir tempo, pico de memória e custo de transições em condições comparáveis.

## 8. Avaliação das melhorias planejadas

### Raio adaptativo

É uma heurística razoável para evitar uma janela muito estreita. A justificativa estatística precisa dizer que, sob hipóteses de independência e variância finita, **a variância cresce com N e o desvio-padrão com √N**. `√fan-in` não garante que neurônios permaneçam na janela durante o treino: densidade dos pesos, taxa de atividade e correlações importam.

### Margem multiclasse

Alinha melhor a perda com classificação por `argmax(scores)` e merece experimento após as correções. Se a perda é definida diretamente em `z_output`, seu subgradiente já é em relação ao score; multiplicar novamente pelo surrogate do degrau de saída introduziria um bloqueio adicional sem necessidade nessa formulação. O surrogate continua necessário nas ocultas discretas.

### Homeostase

Pode ajudar a evitar atividade excessivamente rara ou constante. Não é equivalente a Batch Normalization e não há garantia de que uma taxa de 25% seja ótima. Deve ter frequência, tolerância e interação com o bias supervisionado explícitas. Aplicá-la às ocultas é uma hipótese diferente de forçar a mesma taxa em cada classe de saída.

### Critérios de sucesso

A task 013 deveria “avaliar se há ganho”, não pressupor ganho. A task 012 deveria medir frequência de neurônios inativos/saturados e desempenho, em vez de exigir literalmente que nenhum neurônio jamais tenha taxa 0% ou 100%.

## 9. Sequência recomendada para os próximos agentes

As etapas abaixo são uma proposta de trabalho, não tasks executadas. Registrar a decisão de replanejamento na fila existente e executar uma task por vez conforme `AGENTS.md`.

### Etapa 1 — fechar o contrato matemático e reparar B1

1. Especificar scores, ativações, sinal do contador, limiar e semântica de saturação.
2. Unificar o forward de treino e avaliação.
3. Corrigir a direção da atualização e a associação dos deltas às camadas.
4. Usar a mesma máscara para seleção e reset.
5. Implementar aprendizado dos bias previsto na proposta.
6. Corrigir normalização e definir arredondamento do leak.

**Aceitação:** testes determinísticos em redes pequenas que comprovem direção esperada, correspondência de scores, independência entre camadas, reset exato e comportamento de bias. Não exigir redução de loss em todo passo de uma rede profunda quantizada; testar casos controlados em que a direção esperada é conhecida.

### Etapa 2 — provar capacidade de ajuste em conjunto pequeno

- Fixar um subconjunto de 32–128 amostras, com sementes e índices salvos.
- Medir acurácia antes de treinar e evolução no próprio conjunto.
- Registrar scores, distribuição de classes previstas e taxa de atividade por camada.
- Medir surrogate, delta e incremento antes/depois da quantização.
- Separar candidatos selecionados, mudanças reais e saturação.

Se a rede não conseguir melhorar substancialmente nesse conjunto, investigar o primeiro ponto em que o sinal ou a diversidade desaparece. Isso é mais informativo que um treino longo com vários mecanismos novos.

### Etapa 3 — ensaio MNIST auditável

- Um script versionado deve recriar cada experimento sem comandos ad hoc ocultos.
- Salvar commit, estado local, versões, seed, índices/splits, configuração, tempo e curvas.
- Usar validação para escolher configuração/checkpoint e reservar teste para avaliação final. O baseline atual escolhe checkpoint pelo teste (`src/baseline.py:220–229`).
- Usar ao menos três sementes na rodada comparativa e registrar cada execução, não só a melhor.
- Padronizar unidades: baseline registra porcentagens; B registra frações.
- Controlar preprocessing: baseline usa padronização MNIST; B usa pixels `[0,1]`. Mesma topologia não isola o efeito da quantização quando também mudam ativação, perda e entrada.

### Etapa 4 — ablações antes da grade completa

Comparar B1 corrigido com: leak isolado, margem isolada, homeostase isolada e combinações justificadas. Fixar orçamento e conjunto de seleção. Só expandir para B2/B3 quando houver uma diferença operacional definida entre elas.

Depois, acrescentar controles que ajudem a separar capacidade de representação de dificuldade de otimização: por exemplo, uma rede ternária treinada por método estabelecido com pesos latentes e um controle que treine somente a leitura de features fixas. São experimentos de diagnóstico, não substitutos da ideia principal.

### Etapa 5 — validar a proposta de eficiência

Medir compromisso entre acurácia, largura de contador, frequência de transição, memória e tempo. Reduzir contadores para 8/16 bits exige política de saturação/overflow e ensaios próprios. Avaliar empacotamento ou hardware apenas depois de existir um modelo treinado confiável.

## 10. Verificações realizadas nesta revisão

Executados com sucesso:

```bash
python -m src.fake_gradient
python -m src.reset_policies
```

Ambos passaram. **Isso não invalida os achados:** o teste de normalização atualmente codifica o erro, e os testes de shapes não detectam a direção de aprendizado.

Também foram executadas reproduções por `python -c`, com NumPy `2.5.3`, confirmando os exemplos de avaliação, sinal de saída, leak e normalização. Abaixo estão dois casos mínimos reutilizáveis a partir da raiz do repositório:

```python
import numpy as np
from src.ternary_net import TernaryNet
from src.ternary_b1 import evaluate, train_sample_online
from src.counter_rules import CounterManager

# Avaliação deve preservar as duas ativações ocultas.
net = TernaryNet(input_dim=1, hidden1=1, hidden2=2, output_dim=2, seed=0)
net.layers[0].weights[:] = 1
net.layers[1].weights[:] = 1
net.layers[1].bias[:] = [1, 0]
net.layers[2].weights[:] = [[1, -1], [0, 1]]
x = np.array([[1]], dtype=np.float32)
net.forward(x)
expected_label = net.layers[-1]._cache["z_pre"].argmax(axis=1)
print(expected_label, evaluate(net, x, expected_label))
# Commit analisado: [1], 0.0. A avaliação correta seria 1.0.

# Classe correta inativa deve receber evidência para aumentar seu score.
net = TernaryNet(input_dim=1, hidden1=1, hidden2=1, output_dim=2, seed=0)
net.layers[0].weights[:] = 1
net.layers[1].weights[:] = 1
net.layers[2].weights[:] = 0
cm = CounterManager([(1, 1), (1, 1), (2, 1)])
train_sample_online(net, cm, np.array([1.], dtype=np.float32), 0, radius=100)
print(net.layers[2].weights[:, 0])
# Commit analisado: [-1, 0], direção errada para a classe correta.
```

Não foi executado treinamento completo, benchmark de hardware ou revisão bibliográfica externa. Os números de MNIST deste relatório são os artefatos existentes, não novos resultados produzidos pela análise.

## 11. Conclusão para o autor

O aspecto mais promissor da sua ideia é **fazer o aprendizado operar diretamente sobre pesos de poucos estados, usando evidência temporal explícita para decidir transições**. Isso é uma pergunta de pesquisa legítima e pode produzir resultados úteis mesmo sem alcançar a acurácia de um MLP float32.

O obstáculo imediato é que a implementação ainda não representa fielmente essa pergunta. Há erro de avaliação e casos em que o treinador empurra o peso na direção contrária à desejada. Assim, seria precipitado tanto abandonar a proposta por causa dos 10% quanto esperar que mais heurísticas resolvam o problema.

**O próximo marco deve ser simples e verificável: um B1 matematicamente consistente que aprenda um conjunto pequeno, com scores, contadores e transições auditáveis.** Depois disso, os experimentos de margem, homeostase, seleção e precisão dos contadores terão valor interpretável.
