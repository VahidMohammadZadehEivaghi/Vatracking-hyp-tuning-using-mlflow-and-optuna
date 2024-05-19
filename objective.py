import torch.optim as optim
import torch
from train import train_and_evaluate_for_one_epoch
from model import MnistClassifier
from torchinfo import summary
from data_loader import load_data
from torchmetrics import Accuracy
import mlflow


def objective(trial):

    if not hasattr(objective, "num_calls"):
        objective.num_calls = 1
    else:
        objective.num_calls += 1

    epoch = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    train_loader, val_loader, test_loader = load_data(batch_size)
    model = MnistClassifier().to(device)
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=f"run_{objective.num_calls}", nested=True):
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 0.0, 1.0)
        params = {
            "batch_size": batch_size,
            "epoch": epoch,
            "device": device,
            "metric": metric_fn.__class__.__name__,
            "learning_rate": lr,
            "momentum": momentum,
            "loss": loss_fn.__class__.__name__
        }
        mlflow.log_params(params)
        with open("model_summary.txt", "w") as fout:
            fout.write(str(summary(model)))

        mlflow.log_artifact("model_summary.txt")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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
        return acc
