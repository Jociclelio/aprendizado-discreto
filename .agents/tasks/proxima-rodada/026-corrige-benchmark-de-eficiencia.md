# Task 026: corrige benchmark de eficiência e contadores

Estado: pendente
Depende de: 025
Origem: nenhuma

## Prompt
Produzir contabilidade verificável de armazenamento e custo, distinguindo implementação real, limite teórico e simulação. Substituir contabilidade fixa por medição de arrays/tensores e estados; separar bits reais por peso/bias, média por parâmetro, limite ternário e hipótese de empacotamento 2 bits; contabilizar gradientes, momentos Adam e auxiliares com pico medido por método declarado. Implementar modo de contadores limitados com saturação na acumulação antes de seleção/update/reset (faixa explícita simétrica ou complemento de dois assimétrico, soma em tipo largo, sem wraparound, preservando dtype em leak/normalização/seleção/checkpoint). Distinguir clipping `int64` de armazenamento `int8/int16` real; medir saturação, perda de evidência e efeito no aprendizado. Fixar dispositivo/threads, registrar batch/preprocessing/orçamento/sincronização. Se 025 sem modelo útil, limitar conclusões a custos + diagnóstico; se com modelo útil, comparar baseline x receitas aprovadas com checkpoints corretos em ≥3 sementes via 022/023.

## Contexto
- `results/exp-016-benchmark/` — 1,58 bit/parâmetro com `int8`, Adam como 3 cópias e clipping pós-passo a corrigir.
- `experiments/run_benchmark.py` — script a migrar para núcleo 021 e protocolo 022/023.
- `src/counter_rules.py` — acumulação onde a saturação deve ocorrer.
- `src/baseline.py` — baseline cuja diferença CUDA x CPU deve ser controlada.
- `configs/benchmarks/` — diretório a criar; saídas em `results/exp-026-eficiencia/run-ID/`.

## Corretude
- [ ] C1 — Bytes/bits reportados correspondem a tipos e estados efetivos; tabela separa parâmetros, estado persistente e pico medido, com limitações.
- [ ] C2 — Testes demonstram saturação antes da seleção e ausência de overflow silencioso, incluindo extremos de sinal e incrementos grandes, via `python -m pytest`.
- [ ] C3 — Ensaios distinguem simulação de armazenamento reduzido real; dispositivo, orçamento e método de medição constam em cada execução.
- [ ] C4 — Relatório condiciona qualquer alegação de trade-off útil à evidência de aprendizado de 024/025; não tratar contadores 8/16 bits sem perda em rede colapsada como prova de suficiência; suíte passa com comandos registrados.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
6/10 — Exige medição real de memória/tempo e semântica de saturação correta sem confundir simulação com implementação.

### Dificuldade sentida
_/10 —

### Revisão
- Orquestrador: —
- Alvo: —
- Avaliações: —
- Nota final: _/10
- Parecer: —
- Decisão: pendente
- Reversão: não se aplica
- Próximo passo: —

Plano: [plan.md](../../plans/proxima-rodada/plan.md)
