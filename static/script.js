document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('results');

    const updateResults = (data) => {
        if (data.status === "Error") {
            alert("Error: " + data.message);
            return;
        }
        document.getElementById('experiment_name').innerText = data.experiment_name || 'N/A';
        document.getElementById('run_id').innerText = data.run_id || 'N/A';
        document.getElementById('parameters').innerText = data.parameters ? JSON.stringify(data.parameters) : 'N/A';
        document.getElementById('accuracy').innerText = data.accuracy ? data.accuracy.toFixed(4) : 'N/A';
        document.getElementById('status').innerText = data.status || 'N/A';
    };

    // Load initial results if any
    const fetchInitialResults = async () => {
        try {
            const response = await fetch('/get_results');
            const data = await response.json();
            if (data.run_id) {
                updateResults(data);
            }
        } catch (error) {
            console.error("Failed to fetch initial results:", error);
        }
    };

    runBtn.addEventListener('click', async () => {
        // Show loading state
        runBtn.disabled = true;
        loadingDiv.classList.remove('hidden');
        resultDiv.classList.add('hidden');

        try {
            const response = await fetch('/run_train', {
                method: 'POST'
            });
            const data = await response.json();
            updateResults(data);
        } catch (error) {
            alert("Error triggering training: " + error.message);
        } finally {
            runBtn.disabled = false;
            loadingDiv.classList.add('hidden');
            resultDiv.classList.remove('hidden');
        }
    });

    // New Prediction Logic
    const predictBtn = document.getElementById('predict-btn');
    const predictionBox = document.getElementById('prediction-result');
    const predictedSpeciesSpan = document.getElementById('predicted-species');

    predictBtn.addEventListener('click', async () => {
        const payload = {
            sepal_length: document.getElementById('sepal_length').value,
            sepal_width: document.getElementById('sepal_width').value,
            petal_length: document.getElementById('petal_length').value,
            petal_width: document.getElementById('petal_width').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (data.error) {
                alert(data.error);
            } else {
                predictedSpeciesSpan.innerText = data.prediction;
                predictionBox.classList.remove('hidden');
            }
        } catch (error) {
            alert("Error predicting: " + error.message);
        }
    });

    fetchInitialResults();
});
