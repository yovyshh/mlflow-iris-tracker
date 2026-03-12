# MLflow Iris Experiment Tracker

A simple web-based experiment tracker for the Iris dataset using **MLflow**, **Flask**, and **Scikit-Learn**. This project demonstrates how to log model training runs, track metrics, and serve a model for predictions through a clean web interface.

## 🚀 Features

- **Automated Training**: Trigger model training (Logistic Regression) on the Iris dataset with a single click.
- **Experiment Tracking**: Logs parameters (e.g., `C`, `max_iter`), metrics (Accuracy), and model artifacts using MLflow.
- **Model Registry**: Automatically stores the latest trained model in the local `mlruns/` directory.
- **Real-time Predictions**: Interactive interface to input flower features (sepal/petal length and width) and get instant classifications.
- **Web Interface**: Simple Flask-based UI for managing experiments and predictions.

## 🛠️ Tech Stack

- **ML Framework**: Scikit-Learn
- **Experiment Tracking**: MLflow
- **Backend**: Flask (Python)
- **Data Handling**: Pandas, NumPy
- **Frontend**: HTML/CSS/JS

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yovyshh/mlflow-iris-tracker.git
   cd mlflow-iris-tracker
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

### 1. Start the Flask Web Server
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

### 2. Run Training
Click the **"Run Training"** button on the web interface. This will:
- Load the Iris dataset.
- Train a Logistic Regression model.
- Log the run to MLflow.
- Update the UI with the latest accuracy and Run ID.

### 3. Make Predictions
Enter the Sepal and Petal dimensions in the input fields and click **"Predict"** to see the classified Iris species (Setosa, Versicolor, or Virginica).

### 4. View MLflow UI
To see the detailed experiment logs and comparisons, run:
```bash
mlflow ui
```
Then visit `http://127.0.0.1:5000` in your browser.

## 📂 Project Structure

```text
.
├── app.py              # Flask application & API endpoints
├── train.py            # ML training logic & MLflow integration
├── requirements.txt    # Project dependencies
├── mlruns/             # MLflow local database (auto-generated)
├── static/             # CSS and JS files
└── templates/          # HTML templates
```

## 📝 License
MIT License. Feel free to use and modify!
