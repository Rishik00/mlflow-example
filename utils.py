import os
import mlflow
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

def start_mlflow_server():
    print("Starting MLflow server...")

    host = os.getenv("MLFLOW_HOST", "127.0.0.1")
    port = os.getenv("MLFLOW_PORT", "8080")

    mlflow_uri = f"http://{host}:{port}"
    mlflow.set_tracking_uri(uri=mlflow_uri)

    print(f"MLflow UI launching at {mlflow_uri}")
    subprocess.Popen(["mlflow", "ui", "--host", host, "--port", port])

    time.sleep(2)  # Give the server a moment to start
    print("Open your browser and visit:", mlflow_uri)

    mlflow.set_experiment("Iris Experiment")

def get_or_create_experiment(experiment_name):
      if experiment := mlflow.get_experiment_by_name(experiment_name):
          return experiment.experiment_id
      else:
          return mlflow.create_experiment(experiment_name)