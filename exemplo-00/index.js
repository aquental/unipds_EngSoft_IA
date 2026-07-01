import tf from '@tensorflow/tfjs-node';

/**
 * Trains a neural network model to classify people into 'premium', 'medium', or 'basic'
 * based on normalized age, favorite color (one-hot), and location (one-hot).
 * 
 * @param {tf.Tensor2D} inputXs - Input features (tensor)
 * @param {tf.Tensor2D} outputYs - One-hot encoded labels
 * @returns {Promise<tf.Sequential>} Trained model
 */
async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // === First Hidden Layer ===
    // Input shape: 7 features
    //   - 1: normalized age
    //   - 3: one-hot encoded colors (azul, vermelho, verde)
    //   - 3: one-hot encoded locations (São Paulo, Rio, Curitiba)
    //
    // 80 neurons + ReLU activation:
    // ReLU helps the network learn non-linear patterns and acts as a "filter",
    // passing only positive signals forward.
    model.add(tf.layers.dense({
        inputShape: [7],
        units: 80,
        activation: 'relu'
    }));

    // === Output Layer ===
    // 3 neurons (one for each class: premium, medium, basic)
    // Softmax activation converts raw scores into probabilities that sum to 1.
    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    }));

    // Compile the model
    model.compile({
        // Adam optimizer: adaptive learning rate, generally good default for most problems
        optimizer: 'adam',

        // Categorical cross-entropy is the standard loss for multi-class classification
        // with one-hot encoded labels
        loss: 'categoricalCrossentropy',

        metrics: ['accuracy']
    });

    // Train the model
    await model.fit(inputXs, outputYs, {
        verbose: 0,           // Disable built-in logging (we can use callbacks if needed)
        epochs: 100,          // Number of times the model will see the entire dataset
        shuffle: true,        // Shuffle data each epoch to reduce bias
        callbacks: {
            // Example: log progress per epoch (uncomment if needed)
            // onEpochEnd: (epoch, log) => {
            //     console.log(`Epoch ${epoch}: loss = ${log.loss.toFixed(4)}, accuracy = ${log.acc.toFixed(4)}`);
            // }
        }
    });

    return model;
}

/**
 * Makes a prediction for a new person using the trained model.
 * 
 * @param {tf.Sequential} model - Trained TensorFlow.js model
 * @param {number[][]} pessoa - Normalized input array (2D tensor shape)
 * @returns {Promise<Array<{prob: number, index: number}>>} Array of {probability, classIndex}
 */
async function predict(model, pessoa) {
    // Convert JavaScript array to TensorFlow tensor
    const tfInput = tf.tensor2d(pessoa);

    // Get prediction (returns probabilities for each class)
    const pred = model.predict(tfInput);
    const predArray = await pred.array();

    // Convert to more readable format: array of {prob, index}
    return predArray[0].map((prob, index) => ({ prob, index }));
}

// ===================================================================
// ======================== TRAINING DATA ============================
// ===================================================================

// Normalized and one-hot encoded training data
// Order: [idade_normalizada, azul, vermelho, verde, SP, Rio, Curitiba]
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0.00, 0, 1, 0, 0, 1, 0], // Ana
    [1.00, 0, 0, 1, 0, 0, 1]  // Carlos
];

// Class names indexed to match the one-hot label order below
const labelsNomes = ['premium', 'medium', 'basic'];

// One-hot encoded labels: [premium, medium, basic]
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Create tensors
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// Train the model
const model = await trainModel(inputXs, outputYs);

// ===================================================================
// =========================== PREDICTION ============================
// ===================================================================

const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: "Curitiba" };

// Normalized input for prediction (must follow the same format as training data)
const pessoaTensorNormalizado = [[
    0.2,   // normalized age
    0,     // azul
    0,     // vermelho
    1,     // verde
    0,     // São Paulo
    0,     // Rio
    1      // Curitiba
]];

const predictions = await predict(model, pessoaTensorNormalizado);

// Sort by probability (highest first)
const sorted = predictions.sort((a, b) => b.prob - a.prob);

// Build a clean aligned table for the output
const rows = sorted.map(p => ({
    classe: labelsNomes[p.index],
    prob: `${(p.prob * 100).toFixed(2)}%`
}));

// Column widths adapt to the longest content (including headers)
const classeWidth = Math.max('Classe'.length, ...rows.map(r => r.classe.length));
const probWidth = Math.max('Probabilidade'.length, ...rows.map(r => r.prob.length));

const separator = `+-${'-'.repeat(classeWidth)}-+-${'-'.repeat(probWidth)}-+`;
const formatRow = (classe, prob) =>
    `| ${classe.padEnd(classeWidth)} | ${prob.padStart(probWidth)} |`;

console.log(`Previsão para ${pessoa.nome}:`);
console.log(separator);
console.log(formatRow('Classe', 'Probabilidade'));
console.log(separator);
for (const r of rows) {
    console.log(formatRow(r.classe, r.prob));
}
console.log(separator);
