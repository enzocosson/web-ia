// web.js
// Initialise le modèle ONNX et gère la prédiction côté client

let mnistSession = null;

window.addEventListener('DOMContentLoaded', async () => {
    mnistSession = await loadModel();
    document.getElementById('predict').onclick = async () => {
        if (!mnistSession) {
            document.getElementById('result').textContent = 'Modèle non chargé';
            return;
        }
        const input = getNormalizedInputFromCanvas(document.getElementById('canvas'));
        const prediction = await predictDigit(mnistSession, input);
        document.getElementById('result').textContent = 'Prédiction : ' + prediction;
    };
});
