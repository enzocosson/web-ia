// mnist_onnx.js
// Utilisation d'ONNX.js pour charger et utiliser le modèle MNIST ONNX dans le navigateur

// Inclure onnxruntime-web dans ton HTML :
// <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

async function loadModel() {
    // Charge le modèle ONNX (doit être servi par le serveur web)
    const session = await ort.InferenceSession.create('model.onnx');
    return session;
}

async function predictDigit(session, inputPixels) {
    // inputPixels : tableau JS de 28*28 valeurs normalisées (float32)
    // Prépare le tenseur ONNX (1, 1, 28, 28)
    const inputTensor = new ort.Tensor('float32', Float32Array.from(inputPixels), [1, 1, 28, 28]);
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    // Le nom de sortie est 'output' (voir export ONNX)
    const output = results.output.data;
    // Trouve l'indice du max (classe prédite)
    let maxIdx = 0;
    for (let i = 1; i < output.length; i++) {
        if (output[i] > output[maxIdx]) maxIdx = i;
    }
    return maxIdx;
}

// Fonction pour extraire et normaliser les pixels du canvas (centrage inclus)
function getNormalizedInputFromCanvas(canvas) {
    const small = document.createElement('canvas');
    small.width = 28; small.height = 28;
    const sctx = small.getContext('2d');
    sctx.drawImage(canvas, 0, 0, 28, 28);
    let imgData = sctx.getImageData(0, 0, 28, 28);
    // Centrage automatique du chiffre
    let minX = 28, minY = 28, maxX = 0, maxY = 0;
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const i = (y * 28 + x) * 4;
            if (imgData.data[i] < 200) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }
    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;
    if (boxW > 0 && boxH > 0) {
        const digit = sctx.getImageData(minX, minY, boxW, boxH);
        sctx.clearRect(0, 0, 28, 28);
        sctx.fillStyle = '#fff';
        sctx.fillRect(0, 0, 28, 28);
        const dx = Math.floor((28 - boxW) / 2);
        const dy = Math.floor((28 - boxH) / 2);
        sctx.putImageData(digit, dx, dy);
        imgData = sctx.getImageData(0, 0, 28, 28);
    }
    // Normalisation identique à l'entraînement
    let input = [];
    for (let i = 0; i < imgData.data.length; i += 4) {
        let v = 1 - imgData.data[i] / 255;
        v = (v - 0.1307) / 0.3081;
        input.push(v);
    }
    return input;
}

// Exemple d'intégration avec le canvas
async function onPredictClick() {
    const input = getNormalizedInputFromCanvas(document.getElementById('canvas'));
    const session = await loadModel();
    const prediction = await predictDigit(session, input);
    document.getElementById('result').textContent = 'Prédiction : ' + prediction;
}

// Exporte les fonctions si besoin
window.loadModel = loadModel;
window.predictDigit = predictDigit;
window.onPredictClick = onPredictClick;
