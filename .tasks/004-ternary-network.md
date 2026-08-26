# [004] Implementa núcleo da Rede B — pesos ternários, bias inteiro, degrau

## Objetivo

Implementar `src/ternary_net.py` com a arquitetura base da Rede B: pesos ternários `{−1, 0, +1}`, bias inteiro, ativação degrau centrada em zero, e forward pass completo na topologia `784 → 128 → 64 → 10`.

## Contexto

Este é o núcleo da técnica proposta. A Rede B tem a mesma topologia da baseline, mas representa e opera com pesos discretos. O forward pass deve ser funcional antes de qualquer mecanismo de aprendizado ser adicionado.

## Escopo

- [x] Criar `src/ternary_net.py`
- [x] Classe `TernaryLayer(input_dim, output_dim)` — camada individual
  - Pesos: matriz `(output_dim, input_dim)` de int8 em `{−1, 0, +1}`
  - Bias: vetor `(output_dim,)` de int32 em `[−input_dim, input_dim−1]`
  - Forward: `z = W @ x + b`, saída = `step(z)` (degrau: 1 se z > 0, 0 se z ≤ 0)
  - Método `quantize_weights()` — garante W ∈ {−1, 0, +1}
- [x] Classe `TernaryNet()` — rede completa
  - Constrói 3 camadas: 784→128, 128→64, 64→10
  - Forward pass completo
  - Método `get_weights()` — retorna lista de matrizes de peso (para métricas)
  - Método `get_biases()` — retorna lista de vetores de bias
- [x] Forward pass testável com input MNIST (784,)

## Especificações técnicas

- **Pesos**: numpy int8, inicialização aleatória uniforme em {−1, 0, +1}
  - P(0) ≈ 1/3, P(+1) ≈ 1/3, P(−1) ≈ 1/3 (inicialização balanceada)
- **Bias**: numpy int32, inicializado em 0
  - Faixa: `b ∈ [−N, N−1]` onde N = input_dim da camada
- **Ativação degrau**: `step(z) = 1 se z > 0, else 0`
  - Sem bias na ativação — a decisão é puramente por z > 0
- **Camada de saída**: degrau também? Ou precisa de adaptação para multi-classe?
  - Para MNIST com 10 classes, degrau simples não funciona diretamente
  - Opção: saída como one-hot binário (10 neurônios, cada um decide "é classe i?")
  - Ou: argmax da soma ponderada (mas isso quebra a ativação degrau)
  - Decisão: usar degrau em todas as camadas, incluindo saída; perda = erro por classe

## Critérios de aceitação

- [x] Código implementado em `src/ternary_net.py`
- [x] `TernaryNet` forward pass com input (1, 784) retorna output (1, 10)
- [ ] Pesos permanecem em {−1, 0, +1} após qualquer operação
- [x] Bias permanece dentro da faixa permitida
- [x] Executa sem erro com input MNIST real
- [x] Commit [004] com mensagem descritiva

## Notas de implementação

- **Saída degrau para MNIST**: com 10 neurônios de saída e degrau, cada neurônio decide binariamente se a imagem pertence àquela classe. A perda pode ser soma de erro absoluto por neurônio (equivalente a BCE por classe, sem softmax).
- **Inicialização**: usar distribuição balanceada em {−1, 0, +1}. Talvez gerar floats uniformes e quantizar: `W = np.sign(np.random.randn(...) - 0.5)` (empurra ~50% para 0).
- **Contadores**: ainda não implementados aqui — vêm em tasks separadas. Esta task é apenas a estrutura base.
- **Memória**: pesos int8 = 1 byte cada. Para 784×128 + 128×64 + 64×10 = ~107k pesos → ~107KB vs ~428KB float32.

## Dependências

- Bloqueado por: `[002]`
- Bloqueia: `[005]`, `[006]`, `[007]`
