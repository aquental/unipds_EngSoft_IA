# exemplo-00 — Classificação de Clientes com Rede Neural

Demonstração de uma rede neural de classificação multiclasse usando **TensorFlow.js** no Node.js.
O modelo aprende a classificar pessoas em três categorias — `premium`, `medium` ou `basic` — com base em atributos como idade, cor favorita e localização.

## Tecnologias

- **Node.js** >= 18 e < 23 (ESM nativo com `"type": "module"`)
- **[@tensorflow/tfjs-node](https://www.npmjs.com/package/@tensorflow/tfjs-node)** ^4.22.0

## Como executar

```bash
npm install
npm start
```

O script `start` usa `--watch`, então o Node reinicia automaticamente quando `index.js` é salvo.

---

## Estrutura do `index.js`

O arquivo é dividido em quatro blocos lógicos: **funções**, **dados de treino**, **predição** e **saída formatada**.

---

### 1. `trainModel(inputXs, outputYs)` — Construção e Treino do Modelo

```js
async function trainModel(inputXs, outputYs) { ... }
```

#### Arquitetura da rede

```
Entrada (7 features)
       │
   Dense(80, ReLU)   ← camada oculta
       │
   Dense(3, Softmax) ← camada de saída
       │
 [premium, medium, basic]
```

| Camada | Neurônios | Ativação | Motivo |
|--------|-----------|----------|--------|
| Dense (oculta) | 80 | ReLU | Aprende padrões não-lineares; zera sinais negativos, evitando saturação |
| Dense (saída) | 3 | Softmax | Converte scores brutos em probabilidades que somam 1 |

#### Compilação

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| `optimizer` | `adam` | Ajuste adaptativo de taxa de aprendizado; boa escolha padrão |
| `loss` | `categoricalCrossentropy` | Função de perda padrão para classificação multiclasse com labels one-hot |
| `metrics` | `['accuracy']` | Monitora a acurácia durante o treino |

#### Treino (`model.fit`)

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| `epochs` | 100 | Número de passagens completas pelo dataset |
| `shuffle` | `true` | Embaralha os dados a cada época para reduzir viés de ordem |
| `verbose` | 0 | Suprime o log padrão do TensorFlow (saída limpa) |

O modelo treinado é retornado como `Promise<tf.Sequential>`.

---

### 2. `predict(model, pessoa)` — Inferência

```js
async function predict(model, pessoa) { ... }
```

Recebe o modelo treinado e um array 2D normalizado com os atributos de uma pessoa.
Retorna um array de objetos `{ prob, index }` com a probabilidade de cada classe.

**Fluxo interno:**
1. Converte o array JS em `tf.tensor2d`
2. Chama `model.predict()` — retorna probabilidades via Softmax
3. Converte o tensor para array JS com `.array()`
4. Mapeia cada probabilidade ao seu índice de classe

---

### 3. Dados de Treino

#### Formato das features (7 colunas)

| Posição | Feature | Descrição |
|---------|---------|-----------|
| 0 | `idade_normalizada` | Idade dividida pelo valor máximo do dataset |
| 1 | `azul` | 1 se cor favorita é azul, 0 caso contrário |
| 2 | `vermelho` | 1 se cor favorita é vermelho, 0 caso contrário |
| 3 | `verde` | 1 se cor favorita é verde, 0 caso contrário |
| 4 | `São Paulo` | 1 se localização é SP, 0 caso contrário |
| 5 | `Rio` | 1 se localização é Rio, 0 caso contrário |
| 6 | `Curitiba` | 1 se localização é Curitiba, 0 caso contrário |

> Cores e localizações usam **one-hot encoding**: apenas um elemento é 1, os demais são 0.
> Isso evita que o modelo interprete as categorias como valores ordinais.

#### Exemplos de treino

| Pessoa | idade_norm | azul | vermelho | verde | SP | Rio | Curitiba | Label |
|--------|-----------|------|----------|-------|----|-----|----------|-------|
| Erick  | 0.33 | 1 | 0 | 0 | 1 | 0 | 0 | premium |
| Ana    | 0.00 | 0 | 1 | 0 | 0 | 1 | 0 | medium  |
| Carlos | 1.00 | 0 | 0 | 1 | 0 | 0 | 1 | basic   |

#### Labels (one-hot)

| Classe | Vetor |
|--------|-------|
| premium | `[1, 0, 0]` |
| medium  | `[0, 1, 0]` |
| basic   | `[0, 0, 1]` |

---

### 4. Predição e Saída

O modelo é aplicado a uma nova pessoa: **Zé, 28 anos, cor verde, Curitiba**.

```js
const pessoaTensorNormalizado = [[0.2, 0, 0, 1, 0, 0, 1]];
```

As probabilidades são ordenadas da maior para a menor e exibidas em uma tabela ASCII alinhada dinamicamente:

```
Previsão para zé:
+---------+---------------+
| Classe  | Probabilidade |
+---------+---------------+
| basic   |        XX.XX% |
| medium  |        XX.XX% |
| premium |        XX.XX% |
+---------+---------------+
```

A largura das colunas se adapta ao conteúdo mais longo (cabeçalho ou dado), garantindo alinhamento correto independente dos valores retornados.

---

## Conceitos demonstrados

- Pré-processamento de dados categóricos com **one-hot encoding**
- Normalização de features contínuas
- Construção de rede neural sequencial com TensorFlow.js
- Função de ativação **ReLU** (camada oculta) e **Softmax** (saída)
- Otimizador **Adam** e perda **Categorical Cross-Entropy**
- Inferência com `model.predict()` e formatação da saída
