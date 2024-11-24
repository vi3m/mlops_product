import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes


os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"


# MLflow configurations
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Replace with your MLflow server URI
ARTIFACT_STORE_URI = "s3://mlflow"  # MinIO bucket configured for MLflow artifacts
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiment name
EXPERIMENT_NAME = "Diabetes Regression Experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

# Start MLflow experiment
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Log dataset details
    mlflow.log_param("dataset", "diabetes")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Train the model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Log model performance
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model logged to MLflow with MSE: {mse}")

    artifact_uri = mlflow.get_artifact_uri()
    print(f"Artifact URI: {artifact_uri}")

print(f"MLflow Experiment '{EXPERIMENT_NAME}' completed!")
