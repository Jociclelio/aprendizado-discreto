# [003] Implementa fake gradient triangular e quantização

## Objetivo

Implementar `src/fake_gradient.py` com a função fake gradient triangular centrada em zero e a função de quantização para inteiros pequenos.

## Contexto

A Rede B não usa backpropagation clássico — usa um "fake gradient" como substituto da derivada da função degrau. Este módulo é a base de toda a propagação de erro na rede ternária.

## Escopo

- [x] Criar `src/fake_gradient.py`
- [x] Função `triangular_fake_grad(z, scale, radius)` — triangular com pico em z=0
  - Forma: `f(z) = max(0, 1 - |z| / scale)` onde `scale` é hiperparâmetro
  - Altura do pico = `scale` (hiperparâmetro equivalente à taxa de aprendizado)
- [x] Função `quantize_fake_grad(z, levels)` — quantiza para {-1,0,+1}, {-2..+2}, etc.
  - `levels=5` → `{-2, -1, 0, 1, 2}`
  - `levels=3` → `{-1, 0, 1}` (equivalente a SignSGD)
  - `levels=2` → `{-1, 1}` (sign puro, sem zero)
- [x] Função `propagate_delta(delta_next, weights, fake_grad_val)` — regra da cadeia discreta
  - `δ_j^(l) = fake_grad_val · Σ_k w_jk^(l+1) · δ_k^(l+1)`
- [x] Função `propagate_delta_with_fake()` — propagação completa com fake gradient local
- [x] Classe `CounterAccumulator` — acumulador de contadores inteiros por conexão
- [x] Testes unitários — 5 blocos, 20+ asserts, todos passaram

## Especificações técnicas

- **Triangular**: `f(z) = scale * max(0, 1 - |z| / radius)` onde `radius` define a largura da base
  - `scale` = hiperparâmetro principal (hipoteticamente ≈ lr)
  - `radius` = quanto longe de zero o fake gradient ainda é não-nulo
- **Quantização**: mapear valor contínuo para o inteiro mais próximo no conjunto `levels`
- **Propagação**: recebe δ da camada seguinte, multiplica por pesos da camada atual e fake gradient local
- **Tipos**: tudo em numpy para consistência com contadores inteiros

## Critérios de aceitação

- [x] Código implementado em `src/fake_gradient.py`
- [x] `triangular_fake_grad(0) = scale` (pico correto)
- [x] `triangular_fake_grad(z)` = 0 para `|z| >= radius`
- [x] `quantize_fake_grad` mapeia corretamente para os níveis solicitados
- [x] `propagate_delta` produz resultado dimensionalmente correto
- [x] Testes passam — `python src/fake_gradient.py` (20+ asserts)
- [x] Commit [003] com mensagem descritiva

## Notas de implementação

- O fake gradient é **local por neurônio** — cada neurônio calcula o seu próprio
- A propagação é camada a camada, de trás para frente
- Penser em como `scale` e `radius` interagem — isso é o equivalente discreto do lr
- A quantização deve ser determinística (round, não random)

## Dependências

- Bloqueado por: nenhuma
- Bloqueia: `[004]`, `[005]`, `[006]`
