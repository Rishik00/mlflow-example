# mlflow-example
quick mlflow example in script form taken from: https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs


## How to run it
1. Clone the repo, Set your .env which contains `MLFLOW_PORT` , `MLFLOW_HOST` and  `BASE_DIR` for data generation. 
2. Run this command: 
```bash
python main.py
```
3. If yoy want to just open the server either run
```bash
mlflow ui
```
or 
```bash
python utils.py
```
