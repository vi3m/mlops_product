import os
import mlflow
from ray import serve
import numpy as np
import ray

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

# Constants for MLflow
MLFLOW_TRACKING_URI = (
    "http://mlflow:5000"  # Replace with the correct MLflow tracking URI
)
EXPERIMENT_NAME = "Diabetes Regression Experiment"  # Replace with your experiment name


def get_latest_model_uri(experiment_name):
    """
    Fetch the latest model URI for the given experiment from the MLflow tracking server.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Get experiment details
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")

    experiment_id = experiment.experiment_id

    # Query runs to find the latest successful one
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(
            f"No successful runs found for experiment '{experiment_name}'."
        )

    # Extract the latest run's artifact URI
    latest_run_id = runs.iloc[0]["run_id"]
    return f"runs:/{latest_run_id}/model"


@serve.deployment
class MLModelDeployment:
    def __init__(self, experiment_name):
        """
        Initializes the deployment by loading the latest MLflow model for the experiment.
        """
        model_uri = get_latest_model_uri(experiment_name)
        print(f"Deploying model from URI: {model_uri}")
        self.model = mlflow.pyfunc.load_model(model_uri)

    async def __call__(self, request: dict):
        """
        Predict using the loaded model.
        :param request: A dictionary containing the "data" key with input data.
        :return: A dictionary with predictions or an error message.
        """
        data = await request.json()
        data = data.pop("data", {})
        if data is None:
            return {"error": "No data provided for prediction."}

        input_data = np.asarray(data)
        predictions = self.model.predict(input_data)

        return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    ray.init(include_dashboard=True, dashboard_host="0.0.0.0", dashboard_port=8265)
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})

    deployment = MLModelDeployment.bind(EXPERIMENT_NAME)
    serve.run(deployment)

    import time

    while True:
        time.sleep(3600)
