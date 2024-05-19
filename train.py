import mlflow
import torch


def train_and_evaluate_for_one_epoch(
        train_loader,
        val_loader,
        model,
        optimizer,
        metric_fn,
        loss_fn,
        device
):
    model.train()
    model.to(device)
    metric_fn.to(device)

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)

        loss = loss_fn(y, y_pred)
        acc = metric_fn(y, y_pred)

        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch
            mlflow.log_metric("train_loss", loss, step=batch//10)
            mlflow.log_metric("train_accuracy", acc, step=batch//10)
            print(f"train_loss: {loss:3f}, train_accuracy: {acc:3f}")

    model.eval()
    acc = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            acc = metric_fn(y, y_pred)
        mlflow.log_metric("test_accuracy", acc)

    return acc


