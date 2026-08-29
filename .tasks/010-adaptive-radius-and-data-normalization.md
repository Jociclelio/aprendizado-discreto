# [010] Normalização estrita de entrada e Fake Gradient adaptativo ao Fan-In

## Objetivo

Corrigir a normalização de entrada dos dados de treino e implementar o raio do fake gradient dinâmico/adaptativo proporcional ao fan-in de cada camada ($\text{radius}_l \propto \sqrt{N_l}$), evitando a saturação precoce e morte de neurônios.

## Contexto

Na formulação original, o raio do fake gradient triangular é fixado em `radius = 1.0` para todas as camadas, e a soma ponderada de centenas de entradas ($\sum w_i x_i$) ultrapassa facilmente essa janela, resultando em gradientes nulos em quase 100% dos neurônios. Além disso, o acesso direto a `train_dataset.data` carrega valores em `[0, 255]`. Esta task resolve a sensibilidade de escala.

## Escopo

- [x] Corrigir `src/utils.py` para garantir que `load_mnist()` retorne valores estritamente em `[0, 1]` ou `[-1, 1]` ao ler tensores brutos
- [x] Atualizar `src/fake_gradient.py` com suporte a raio adaptativo baseado no fan-in da camada:
  $$\text{radius}_l = \text{scale\_factor} \times \sqrt{\text{input\_dim}_l}$$
- [x] Integrar o cálculo de raio por camada no forward/backward de `src/ternary_net.py` e `src/ternary_b1.py`
- [x] Executar teste com a nova escala e validar se a magnitude do $\delta$ se mantém estável através de todas as camadas (sem colapsar para 0)
- [x] Salvar métricas e curvas de validação em `results/exp-010-adaptive-radius/`

## Especificações técnicas

- **Normalização de Entrada**: $x \in [-1, +1]$ (bipolar) ou $[0, 1]$
- **Raio Adaptativo**:
  - Camada 1 ($784 \to 128$): $\text{radius}_0 = c \cdot \sqrt{784} \approx 28 \cdot c$
  - Camada 2 ($128 \to 64$): $\text{radius}_1 = c \cdot \sqrt{128} \approx 11.3 \cdot c$
  - Camada 3 ($64 \to 10$): $\text{radius}_2 = c \cdot \sqrt{64} = 8 \cdot c$
- **Hiperparâmetros**:
  - `scale_factor` ($c$): $0.5$ a $1.0$

## Critérios de aceitação

- [x] `src/utils.py` validado com assert de que `train_data.max() <= 1.0`
- [x] `src/fake_gradient.py` implementa `adaptive_triangular_fake_grad`
- [x] Sinal de $\delta$ não nulo em $>50\%$ dos neurônios em todas as camadas durante o backward
- [x] Resultados salvos em `results/exp-010-adaptive-radius/`
- [x] Commit `[010]` com mensagem descritiva

## Resultados

- `load_mnist(normalize=True)` agora retorna max 1.0 (antes 255.0) — `src/utils.py:89`
- `fake_gradient.py:189` adiciona `get_adaptive_radius`, `get_layer_radii`, `adaptive_triangular_fake_grad`
- Validação 1000 amostras: fixo r=1.0 → L0 11%, L1 6%, L2 12% não-nulo; adaptativo c=0.5 → 94%/61%/72%, c=1.0 → 99%/93%/98% (todos >50%)
- `ternary_b1.py:163` integra raio por camada via `adaptive_radius`/`scale_factor`; contadores L0 agora acumulam (antes max 0)
- Treino rápido 2000 amostras ainda ~10% acc — indica que só raio não basta, precisa 011 (leaky) + 012 (margin/homeostase) para convergência completa; métricas salvas em `results/exp-010-adaptive-radius/metrics.json` e `delta_fraction.png`

## Notas de implementação

- A variância da soma de $N$ variáveis aleatórias independentes de variância $\sigma^2$ escala com $\sqrt{N}$. Casar a base do triângulo com $\sqrt{N}$ garante que neurônios operem dentro do cone de aprendizado do fake gradient.

## Dependências

- Bloqueado por: `[009]`
- Bloqueia: `[011]`
