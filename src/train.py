import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes


os.environ['MLFLOW_S3_ENDPOINT_URL']="http://localhost:9000"
os.environ['AWS_ACCESS_KEY_ID']="minio"
os.environ['AWS_SECRET_ACCESS_KEY']="minio123"


# MLflow configurations
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Replace with your MLflow server URI
ARTIFACT_STORE_URI = "s3://mlflow"  # MinIO bucket configured for MLflow artifacts
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiment name
EXPERIMENT_NAME = "Diabetes Regression Experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

# Load the dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple regression model
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

    # Save and log the model
    # model_path = "models/diabetes_model"
    # mlflow.sklearn.save_model(model, model_path)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model logged to MLflow with MSE: {mse}")

    # Log the artifact
    artifact_uri = mlflow.get_artifact_uri()
    print(f"Artifact URI: {artifact_uri}")

# Verify metadata in PostgreSQL and artifacts in MinIO
print(f"MLflow Experiment '{EXPERIMENT_NAME}' completed!")



# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import load_diabetes
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# # Load dataset
# X, y = load_diabetes(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# mlflow.set_tracking_uri("http://localhost:5000")

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Log model with MLflow
# mlflow.sklearn.log_model(model, "model")
# mlflow.log_metric("r2_score", model.score(X_test, y_test))


####

# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# # Set up MLflow tracking server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Set up MinIO artifact store
# minio_artifact_uri = "s3://minio:minio123@minio:9000/mlflow"

# # Set up Postgres metadata store
# postgres_metadata_uri = "postgresql://mlflow:mlflow@localhost:5432/mlflow"

# # Load data
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# # Train model
# lr_model = LogisticRegression()
# lr_model.fit(X_train, y_train)

# # Log model to MLflow
# with mlflow.start_run():
#     mlflow.log_param("model_type", "LogisticRegression")
#     mlflow.log_metric("accuracy", lr_model.score(X_test, y_test))
#     mlflow.sklearn.log_model(lr_model, "model")

# # Store artifacts in MinIO
# mlflow.artifacts.store(minio_artifact_uri, "model", "model.pkl")

# # Store metadata in Postgres
# mlflow.metadata.store(postgres_metadata_uri, "model", "model_metadata")