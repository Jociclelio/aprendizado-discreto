# [017] Playground web de visualização do treino (Streamlit)

## Objetivo

Criar uma interface web Streamlit que permita ver **amostra a amostra** o processo de aprendizagem da Rede B (B1/B2/B3) sobre dígitos reais do MNIST e verificar se cada estágio está sendo aplicado conforme o CONTRATO_012.

## Contexto

As tasks 012–016 repararam o B1, provaram (e delimitaram) o aprendizado e consolidaram a B+ refinada, mas a inspeção continua via logs JSON e PNGs estáticos em `results/`. Para depurar "o sinal morreu onde?" (surrogate zero, delta cancelado, quantização, máscara, bloqueio por saturação) é preciso um **stepper interativo**: escolher um dígito, avançar 1 amostra e ver `z → step → erro → fake-grad → deltas → dw → C → máscara → transição/reset`, com os mesmos números do CLI — sem reimplementar a matemática em paralelo.

Decisões fechadas com o usuário: granularidade **amostra a amostra**, stack **Streamlit**, variantes **B1+B2+B3 comparáveis**, **MNIST interativo essencial**.

## Escopo

- [x] **017.1 Tracer instrumentado** em `src/playground/tracer.py`:
  - [x] Envolve `TernaryNet.forward_with_scores`, `triangular_fake_grad`, `quantize_fake_grad`, `propagate_delta_with_fake_quantized`, `CounterManager` e `train_sample_online` (B1) **sem alterar `src/` existente**.
  - [x] Retorna `Trace` dict por amostra: `x, label, z_l, a_l, scores, pred, error, loss, fake_cont, q_fake, deltas[l], dw_float, dw_int, quant_stats, Cw/Cb antes/depois, thr_w/thr_b, mask, selecionados/tentativas/bloqueios/transições (pesos e bias)`.
  - [x] Adaptadores `trace_b2` / `trace_b3` que trocam **só** a função de máscara (Top-K por camada / por rede via `counter_rules.py`) mantendo o resto idêntico.
  - [x] Teste `python -m src.playground.tracer`: bit-identidade com `train_sample_online` (mesmos `W`, `C`, métricas para mesma seed/amostra).
- [x] **017.2 App Streamlit + inspetor** em `src/playground/{app,state,plots}.py`:
  - [x] Aba 1 — Stepper: sidebar (variante, `scale`, `radius/adaptativo c`, `levels {3,5,7}`, `seed`, índice MNIST/aleatório, Prev/Next/Step/Replay); painel nos 7 estágios do pipeline na ordem do CONTRATO_012.
  - [x] Aba 2 — Inspetor de camada L0/L1/L2: heatmaps **amostrados** (nunca matriz cheia; ex. crop 32×32 ou top-|C| determinístico), histograma de `z` com cone do raio adaptativo (`get_adaptive_radius`/`get_layer_radii`), distribuição `+1/0/-1` via `net.get_stats()`.
  - [x] `state.py`: `session_state` para net+cm (botão Reset com seed) + `@st.cache_data` para `load_mnist(normalize=True)`.
  - [x] `plots.py`: só matplotlib (Agg, padrão do repo).
- [x] **017.3 Runner + comparador + polish**:
  - [x] Aba 3 — Runner curto: N amostras (default 200–2000, **cap 2000** no interativo; full MNIST fica no CLI), live chart loss/acc/transições por batch (reusa `train_batch` + `evaluate`), log textual igual ao CLI.
  - [x] Aba 4 — Comparador B1/B2/B3: mesmo seed + mesmo subset, curvas sobrepostas (reusa `plot_comparison_curves` de `src/utils.py`), tabela acc final / transições / % bloqueios.
  - [x] `requirements`/docs: acrescentar `streamlit` (única dependência nova), seção no README, smoke test e screenshot/log em `results/exp-017-playground/`.

## Especificações técnicas

- **Arquitetura**: `784 → 128 → 64 → 10` (invariante; sem mudar matemática).
- **Contrato**: CONTRATO_012 (`src/ternary_b1.py` docstring): `C` acumula gradiente, update `W <- clip(W − sign(C))`, máscara única por camada aplicada a update E reset, saturação bloqueia peso mas consome contador, leak simétrico opcional.
- **Arquivos novos**: `src/playground/__init__.py`, `tracer.py`, `app.py`, `state.py`, `plots.py`. Nenhum arquivo `src/*.py` existente é modificado (só leitura/import).
- **Raios adaptativos**: defaults `c=1.0 → [28.0, ~11.3, 8.0]` para fan-ins `[784, 128, 64]`.
- **Performance**: L0 tem ~100k conexões — heatmaps sempre amostrados; MNIST via cache; treino interativo limitado; `batch_size` default 32.
- **Reprodutibilidade**: toda view exibe `seed`, hiperparâmetros e índice da amostra; `Reset` restaura net+cm da seed.

## Critérios de aceitação

- [x] `streamlit run src/playground/app.py` abre sem erro.
- [x] Step de 1 amostra exibe os 7 estágios com números **iguais** aos do CLI (teste de bit-identidade passa).
- [x] Troca B1 → B2 → B3 muda máscara/selecionados/transições de forma visível e correta.
- [x] Dígito MNIST clicável: imagem 28×28 + forward (`z`, firing, scores, pred) + backward completo.
- [x] Runner curto executa sem travar (≤ cap) e salva artefatos em `results/exp-017-playground/`.
- [x] Commit `[017]` com mensagem descritiva.

## Notas de implementação

- O tracer **observa e copia** arrays (`C` antes/depois, `dw`, `deltas`); o update real continua na função original. Se B2/B3 não tiverem `train_sample_online` equivalente, 017.1 inclui extrair um `train_sample` comum por variante — sem mudar a matemática de nenhuma.
- Separar sempre: selecionados vs tentativas vs bloqueios (saturação) vs transições efetivas; pesos vs bias; `dw` antes vs depois do `round` (diagnóstico de quantização da task 013).
- Armadilhas conhecidas (013): surrogate de saída zera quando `|z_out| >> raio` (rede congela, 0 transições) — o stepper deve tornar isso óbvio (histograma `z` + overlay do triângulo + `q_fake==0` destacado).
- Não introduzir margem/homeostase/leak novos aqui — só expor os hiperparâmetros já existentes (`scale`, `radius/c`, `levels`, `decay_*`, `margin` da B+ se trivial).

## Dependências

- Bloqueado por: `[012]` (contrato), `[015]` (config B+); usa `[002]` (loader/plots), `[003]` (fake-grad), `[005]` (regras).
- Bloqueia: nenhuma (ferramenta de inspeção, fora do caminho crítico 012→016).
