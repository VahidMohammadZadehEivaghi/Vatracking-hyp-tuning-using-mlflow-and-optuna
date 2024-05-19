from objective import objective
import optuna
import mlflow

if __name__ == "__main__":

    tracking_uri = "https://127.0.0.1:8000"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("hyper-parameter-tuning")
    with mlflow.start_run(run_name="hyper-parameter-tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

