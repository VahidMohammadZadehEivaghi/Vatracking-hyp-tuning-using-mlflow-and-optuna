from objective import objective
import optuna
import mlflow
from train import train_and_evaluate_for_one_epoch
from model import MnistClassifier
from data_loader import load_data
import torch
from torchmetrics import Accuracy
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature
import numpy as np

if __name__ == "__main__":

    tracking_uri = "http://127.0.0.1:8080"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("hyper-parameter-tuning")
    with mlflow.start_run(run_name="hyper-parameter-tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        mlflow.log_params(best_params)
        epoch = 50
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 16
        train_loader, val_loader, test_loader = load_data(batch_size)
        model = MnistClassifier().to(device)
        metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), **best_params)

        for e in range(1, epoch + 1):
            print(f"------------------Epoch {e} / {epoch}-------------------")
            acc = train_and_evaluate_for_one_epoch(
                train_loader,
                val_loader,
                model,
                optimizer,
                metric_fn,
                loss_fn,
                device
            )
            print(f"acc:{acc}")

        input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 28, 28))])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pytorch.log_models(
            model,
            "mnist-classifier",
            signature=signature
        )
