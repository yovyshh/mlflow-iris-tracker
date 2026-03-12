import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def run_training(experiment_name="IrisExperiment"):
    # Fix for Windows paths with spaces
    import os
    tracking_uri = "file:mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set the experiment name
    mlflow.set_experiment(experiment_name)
    
    # Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameters
    params = {"C": 1.0, "max_iter": 200}
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Predictions and Metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model artifact
        mlflow.sklearn.log_model(model, "iris_model")
        
        # Get run info
        run_id = mlflow.active_run().info.run_id
        
    return {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "parameters": params,
        "accuracy": accuracy,
        "status": "Finished"
    }

def predict_iris(features):
    """
    features: list of [sepal_length, sepal_width, petal_length, petal_width]
    """
    mlflow.set_tracking_uri("file:mlruns")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("IrisExperiment")
    
    if not experiment:
        return {"error": "No experiment found. Please train the model first."}
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1
    )
    
    if not runs:
        return {"error": "No trained model found. Please click 'Run Training' first."}
        
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/iris_model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Class names mapping
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    
    # Predict
    prediction = model.predict([features])
    predicted_class = class_names[int(prediction[0])]
    
    return {
        "prediction": predicted_class,
        "run_id": run_id
    }

if __name__ == "__main__":
    result = run_training()
    print(f"Training completed: {result}")
