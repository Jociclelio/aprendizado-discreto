# Ideia — proposta científica (hipóteses, não resultados)

> Este documento descreve a **proposta de pesquisa**: o que se conjectura, por que seria interessante e o que a confirmaria ou refutaria.
> Números de experimentos, limitações de implementação e estado do código estão no `README.md`, em `docs/reports/revisao-f5d3a5a.md` e nos índices `.tasks/INDEX.md` / `results/INDEX.md` — não aqui.
> Referência de análise preservada: `astra-relatório-commit-6f47914.md`.

## 1. Pergunta de pesquisa

É possível treinar uma rede neural cujos **parâmetros usados no forward tenham só três estados** (`W ∈ {−1, 0, +1}`, bias inteiro), usando **evidência temporal explícita por conexão** para decidir transições discretas — sem nenhuma cópia latente float dos pesos?

## 2. Mecanismo proposto

- **Parâmetro real:** peso ternário `W ∈ {−1, 0, +1}`; não há sombra float — o ternário *é* o estado do parâmetro.
- **Estado do otimizador (não é peso):** contador inteiro `C_ij` por conexão (e por bias), que acumula evidência `Δ_ij` ao longo de amostras; a evidência pode somar ou cancelar antes de qualquer transição (histerese).
- **Sinal de erro:** backpropagation com derivada substituta (surrogate triangular do degrau — *straight-through / surrogate gradients*); a descrição honesta é "sem derivada verdadeira do degrau", não "sem gradientes".
- **Propagação discreta por camada:** `δ_j^(l) = fake_grad(z_j^(l)) · Σ_k w_jk^(l+1) · δ_k^(l+1)`; `Δ_ij = δ_j^(l) · x_i^(l−1)`; bias usa entrada fantasma 1.
- **Seleção (hipóteses concorrentes):** B1 por neurônio; B2 Top-K por camada; B3 Top-K por rede.
- **Reset (hipóteses concorrentes):** só atualizados (preserva resíduo dos demais); tudo; normalizar. A preservação parcial de resíduo **não** é automaticamente o error-feedback clássico (que preserva `pretendido − aplicado` em unidades compatíveis) — o nome de trabalho honesto é *reset seletivo com preservação dos não selecionados*.

## 3. Por que seria interessante (se funcionar)

- Aprendizado com **orçamento limitado de escrita** em memória (transições esparsas e discretas).
- **Custo de inferência** separado do custo de treinamento: parâmetros de ~1.58 bit/peso (teórico) vs 32 bit float.
- Histerese explícita: não reagir a cada sinal ruidoso com mudança imediata de peso.

## 4. O que está fora da alegação

- Os contadores são estado latente do otimizador (com `int64`, muito mais informação que o peso). "Sem peso float oculto" ≠ "só três estados por conexão durante o treino".
- O código atual **não** é totalmente inteiro (ativações e intermediários float) nem "zero MACs float" — isso seria propriedade de uma implementação especializada futura, não do protótipo NumPy.
- Economia de memória de **inferência** (~4×) ≠ economia de **treinamento** (parâmetros + contadores ≈ 2.25× os parâmetros float isolados, sem contar momentos do Adam).
- Originalidade da combinação **não estabelecida** (sem revisão bibliográfica externa nesta rodada).

## 5. Questões em aberto (a decidir por experimento, não por definição)

1. O contador acumula gradiente (`W ← clip(W − sign(C))`) ou direção de descida? — convenção fixada em CONTRATO_012; pode ser revista por nova task com decisão explícita.
2. O que dispara B1 (ativação, erro, limiar de `|C|`)? Bloquear atualização de neurônios inativos pode impedir recuperá-los.
3. Saturação: tentativa bloqueada pelo clipping consome o contador ou o mantém? (CONTRATO_012 define; medir separadamente selecionados/tentativas/bloqueios/transições.)
4. Política de arredondamento do leak (simetria, ponto fixo, comportamento perto de zero).
5. Margem multiclasse, homeostase de disparos e leak são **hipóteses de estabilização**, não partes do núcleo — cada uma precisa de ablação com orçamento e seleção fixos.
6. B2/B3 só são métodos distintos com agendas de eventos distintas (caso contrário são equivalentes por camada).

## 6. O que confirmaria ou refutaria a ideia

- **Confirmação mínima:** B1 matematicamente consistente aprende um subconjunto pequeno (32–128 amostras) acima do preditor majoritário, com trilha auditável (surrogate → delta → incremento → seleção → transição) — marco de 024.
- **Refutação parcial (já observada, a explicar):** rede aprende só o prior majoritário e congela quando `|z_out| >> raio` mata o surrogate de saída (mecanismo de 013) — 024 deve reproduzir com controles (pesos latentes float; readout sobre features fixas) para separar capacidade de representação de dificuldade de otimização.
- **Validação de eficiência (só depois de aprender):** compromisso acurácia × bits × transições × memória × tempo em protocolo reproduzível — 026; empacotamento/hardware só então.

## 7. Topologia de referência

`784 → 128 → 64 → 10` (109.184 pesos + 202 bias = 109.386 parâmetros). Invariante em toda a rodada para manter a comparação em capacidade nominal; mudar ativação/perda/entrada junto com a quantização impede isolar o efeito da quantização (limitação registrada).
