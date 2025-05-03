import os
import math
import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

## Local imports
from model_tuning import objective, champion_callback
from data_gen import generate_synthetic_timeseries
from utils import start_mlflow_server, get_or_create_experiment

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR", "content")

if __name__ == "__main__":
    # override Optuna's default logging to ERROR only
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    run_name = 'first_attempt'
    
    start_mlflow_server()
    print("mlflow server at 8080")
    if os.path.exists(f'{BASE_DIR}/sample_data.csv'):
        df = pd.read_csv(f'{BASE_DIR}/sample_data.csv')

    else:
        generate_synthetic_timeseries(
            base_demand = 1000,
            n_rows = 100,
            competitor_price_effect = -50.0,
            file_name = 'sample_data.csv'
        )
        df = pd.read_csv(f'{BASE_DIR}/sample_data.csv')

    experiment_id = get_or_create_experiment("Apples Demand")
    mlflow.set_experiment(experiment_id=experiment_id)

    X = df.drop(columns=["date", "demand"])
    y = df["demand"]
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25, random_state=42)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)
    
    # Initialize the Optuna study
    func = lambda trial: objective(trial, dtrain, dvalid, valid_y)
    study = optuna.create_study(direction="minimize")

    # Execute the hyperparameter optimization trials.
    # Note the addition of the `champion_callback` inclusion to control our logging
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        study.optimize(func, n_trials=500, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mse", study.best_value)
        mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

        # Log tags
        mlflow.set_tags(
            tags={
                "project": "Apple Demand Project",
                "optimizer_engine": "optuna",
                "model_family": "xgboost",
                "feature_set_version": 1,
            }
        )

        # Log a fit model instance
        model = xgb.train(study.best_params, dtrain)
        artifact_path = "model"

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path=artifact_path,
            input_example=train_x.iloc[[0]],
            model_format="ubj",
            metadata={"model_data_version": 1},
        )

        # Get the logged model uri so that we can load it from the artifact store
        model_uri = mlflow.get_artifact_uri(artifact_path)
        print("Model logged at:", model_uri)



