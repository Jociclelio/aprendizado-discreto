# Task 025: reavalia ablações e variantes com seleção limpa

Estado: pendente
Depende de: 024
Origem: nenhuma

## Prompt
Medir efeito de perda, seleção, reset, leak e homeostase com escopo condicionado à decisão de 024 e seleção só por validação. Registrar decisão de 024 e grade finita com orçamento justificado antes de executar. Se 024 sem aprendizado: limitar a hipóteses de falha e documentar adiamento do MNIST maior. Se 024 com aprendizado: comparar primeiro no conjunto pequeno e ampliar só mediante critério de continuidade prévio. Comparar mecanismos isolados antes de combinações; homeostase só com janela/frequência, taxas por neurônio, tolerância e separação ocultas/saída e bias supervisionado; variantes por identificador de política (não só B1/B2/B3) com orçamento/agenda controlados ou declarados. Usar split fixo, ≥3 sementes pareadas, teste reservado à avaliação final da receita congelada. Publicar receita só se critério prévio atendido; senão documentar ausência de configuração aprovada. Absorve investigação de variantes/reset de 008 sem herdar 20 épocas nem `>10%` como critério.

## Contexto
- `.agents/tasks/proxima-rodada/024-diagnostica-aprendizado-discreto.md` — decisão que condiciona o escopo.
- `results/exp-015-ablacoes/` — ablações com 1 seed, sem controles marcados nem atividade por neurônio.
- `src/ablation.py`, `src/ternary_refined.py` — implementações a migrar para o núcleo 021 e protocolo 022/023.
- `configs/ablations/` — diretório a criar; saídas em `results/exp-025-ablacoes/run-ID/`.
- `.tasks/008-b2-b3-variants.md` — escopo histórico de variantes a absorver com justificativa.

## Corretude
- [ ] C1 — Escopo executado corresponde ao caminho autorizado por 024; expansões adiadas têm justificativa e destino futuro.
- [ ] C2 — Comparações têm configs completas, orçamento e ≥3 sementes pareadas em split fixo; curvas, dispersão entre seeds e predições finais preservadas.
- [ ] C3 — Resultados isolados e combinados permitem atribuição limitada de efeitos, com métricas de atividade/colapso (0%/100% por neurônio, distribuição, sinal por camada, transições, bloqueios); transição ≠ oscilação salvo reversões medidas com definição temporal.
- [ ] C4 — Receita publicada corresponde exatamente à selecionada por validação, ou ausência aprovada está explícita; suíte passa; relatório e comandos de reprodução registrados.

## Execução
- Modelo: —
- Base: —
- Entrega: —
- Arquivos da entrega: —
- Evidências: —

## Notas

### Dificuldade esperada
7/10 — Varredura pareada multi-seed com disciplina condicional e atribuição de efeitos exige orçamento e protocolo rígidos.

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
