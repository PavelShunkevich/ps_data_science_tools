import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_batch(
        model: nn.Module,
        inputs: torch.Tensor,
        target: torch.Tensor
) -> tuple[int, int]:
    """Evaluates a single batch of data and returns number of correct predictions and batch size."""
    outputs = model(inputs)
    _, predicted = torch.max(outputs, dim=1)

    total = target.size(0)
    correct = (predicted == target).sum().item()

    return correct, total


def evaluate_dataset(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> dict[str, float]:
    """Evaluates the model on a given dataset and returns a dictionary of metrics."""
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            target = target.to(device)

            batch_correct, batch_total = evaluate_batch(model, inputs, target)
            correct += batch_correct
            total += batch_total

    accuracy: float = correct / total
    return {"accuracy": accuracy}


def evaluate(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluates the model on train and validation datasets and returns a dictionary of metrics."""
    print(f"Evaluating on device {device}.")
    metrics = {}
    metrics["train"] = evaluate_dataset(model, train_loader, device)
    metrics["val"] = evaluate_dataset(model, val_loader, device)
    return metrics