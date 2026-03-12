from flask import Flask, jsonify, render_template, request
import mlflow
from train import run_training, predict_iris

app = Flask(__name__)

# Experiment Name
EXPERIMENT_NAME = "IrisExperiment"

# Configure MLflow to use local 'mlruns' folder to avoid Windows path space issues
mlflow.set_tracking_uri("file:mlruns")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_train', methods=['POST'])
def run_train():
    try:
        # Trigger training and capture results
        result = run_training(EXPERIMENT_NAME)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

@app.route('/get_results', methods=['GET'])
def get_results():
    try:
        # Get MLflow client and search for runs in the specified experiment
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            return jsonify({"status": "No experiments found."})
        
        # Search for runs and sort by newest first
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"]
        )
        
        if not runs:
            return jsonify({"status": "No runs recorded yet."})
            
        # Format results
        latest_run = runs[0]
        result = {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": latest_run.info.run_id,
            "parameters": latest_run.data.params,
            "accuracy": latest_run.data.metrics.get("accuracy", "N/A"),
            "status": latest_run.info.status
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data.get('sepal_length')),
            float(data.get('sepal_width')),
            float(data.get('petal_length')),
            float(data.get('petal_width'))
        ]
        result = predict_iris(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
