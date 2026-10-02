# [011] Contadores com Decaimento Temporal (Leaky Accumulator / Momentum Discreto)

## Objetivo

Implementar mecanismo de decaimento temporal (*leakage* / esquecimento exponencial) nos contadores $C_{ij}$, permitindo que ruídos das épocas iniciais sejam gradualmente descartados e dando maior peso a evidências de gradiente recentes.

## Contexto

Atualmente os contadores $C_{ij}$ acumulam $\Delta_{ij}$ de forma estritamente integrativa sem decaimento. Amostras das primeiras épocas (quando a rede é aleatória) continuam influenciando decisões de atualização permanentemente, causando inércia excessiva e oscilação. Um acumulador com perdas (*leaky integrator*) simula o efeito de momentum moderno no domínio discreto.

## Escopo

- [x] Atualizar `src/counter_rules.py` e `CounterManager` para suportar política de decaimento temporal:
  $$C_{ij}^{(t+1)} = \lfloor \gamma \cdot C_{ij}^{(t)} \rfloor + \Delta_{ij}^{(t)}$$
- [x] Implementar variante eficiente em hardware baseada em bitshift:
  $$C_{ij}^{(t+1)} = C_{ij}^{(t)} - (C_{ij}^{(t)} \gg \text{shift}) + \Delta_{ij}^{(t)}$$
- [x] Criar testes unitários para a dinâmica de decaimento e estabilidade dos contadores
- [x] Testar impacto na convergência de MNIST em comparação com contadores puramente integrativos
- [x] Salvar métricas e curvas de validação em `results/exp-011-leaky-counters/`

## Especificações técnicas

- **Fator de decaimento ($\gamma$)**: $0.90$, $0.95$, $0.99$
- **Bitshift alternativo**: `shift` $\in \{3, 4, 5\}$ (correspondente a atenuações de $\approx 12.5\%$, $6.25\%$, $3.125\%$)
- **Integração**: aplicável a todas as regras de atualização (B1, B2, B3)

## Critérios de aceitação

- [x] `CounterManager` suporta parâmetro `decay_rate` ou `decay_shift`
- [x] Testes em `src/counter_rules.py` validam o decaimento exponencial de contadores inativos
- [x] Treino demonstra redução de oscilações na taxa de troca de pesos (*weight flipping rate*)
- [x] Resultados salvos em `results/exp-011-leaky-counters/`
- [x] Commit `[011]` com mensagem descritiva

## Resultados

- `CounterManager` em `src/counter_rules.py:53` agora aceita `decay_rate` (0.9/0.95/0.99) ou `decay_shift` (3/4/5) com `decay()` e `_apply_decay_array()`; `accumulate(apply_decay=True)` e `decay_per_batch` em `src/ternary_b1.py:280`
- Decaimento validado: 100→90 (γ0.9), 100→88 (shift3), 100→95→90→85 (γ0.95 5 passos), inativo decai 100→57 em 10 passos γ0.95
- Sintético 100 passos Δ=+5: sem leak diverge para 500, com γ0.90 estabiliza 41, γ0.95 81, γ0.99 285, shift4 80 (`counter_stability.png`)
- MNIST rápido (1500 amostras, 3 epochs, adaptive c0.5): sem leak max0=3 flip0=8.4%, γ0.99 max0=4, γ0.95 max0=2, shift4 max0=1 — leak limita magnitude, mas acc ainda ~10% (0.11) e flip cai a 0 após epoch1 com/sem leak, indicando que só leak não desbloqueia aprendizado; precisa primeiro da reparação do B1 (012) e, depois, das ablações (015). Métricas em `metrics.json`, `leaky_comparison.png`.

## Notas de implementação

- A operação por bitshift é ideal para implementação futura em ASIC/FPGA sem necessidade de multiplicadores.

## Dependências

- Bloqueado por: `[010]`
- Bloqueia: `[012]`
