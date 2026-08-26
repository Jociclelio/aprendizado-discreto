# Template de Task

> Use este template como padrão para **todas** as tasks em `.tasks/NNN-nome.md`.

---

## [NNN] Título da Task

### Objetivo

Descreva em uma frase o que esta task entrega.

### Contexto

Por que esta task existe? Qual problema ela resolve ou qual milestone ela avança?

### Escopo

- [ ] Item 1
- [ ] Item 2
- [ ] Item 3

### Especificações técnicas

- Arquitetura: ...
- Hiperparâmetros: ...
- Interface esperada: ...
- Dependências entre módulos: ...

### Critérios de aceitação

- [ ] Código implementado em `src/...`
- [ ] Executa sem erro com `python src/...`
- [ ] Resultado mínimo alcançado (ex.: acurácia > X%, ou "aprende algo em MNIST")
- [ ] Commit [NNN] com mensagem descritiva
- [ ] Resultados salvos em `results/exp-NNN-.../`

### Notas de implementação

- Observações, referências, armadilhas conhecidas
- Decisões de design que precisam ser justificadas depois

### Dependências

- Bloqueado por: `[NNN]` (ou "nenhuma")
- Bloqueia: `[NNN]` (ou "nenhuma")

---

## Regras de preenchimento

1. **Sempre** preencher "Objetivo" e "Critérios de aceitação"
2. **Sempre** marcar critérios como [ ] antes de começar
3. Atualizar para [x] conforme progride
4. **Nunca** pular para a próxima task sem finalizar a anterior
5. Se uma task precisar ser dividida, criar subtasks numeradas (`001.1`, `001.2`)
6. Referenciar tasks por número: `[001]`, `[002]`, etc.
