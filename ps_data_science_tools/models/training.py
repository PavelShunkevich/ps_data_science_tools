import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime


def train_one_batch(
        model: nn.Module,
        inputs: torch.Tensor,
        target: torch.Tensor,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module
) -> float:
    """Trains a single batch of data."""
    outputs = model(inputs)
    loss = loss_fn(outputs, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch(
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
) -> float:
    """Trains one epoch and returns the average loss."""
    loss_train = 0.0
    for inputs, target in train_loader:
        inputs = inputs.to(device)
        target = target.to(device)
        loss_train += train_one_batch(model, inputs, target, optimizer, loss_fn)
    return loss_train / len(train_loader)


def training_loop(
        n_epochs: int,
        optimizer: optim.Optimizer,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        print_interval: int = 10
) -> None:
    """Executes the training loop for a given PyTorch model."""
    print(f"Training on device {device}.")
    for epoch in range(1, n_epochs + 1):
        avg_loss = train_epoch(model, optimizer, loss_fn, train_loader, device)

        if epoch == 1 or epoch % print_interval == 0:
            print(f"{datetime.datetime.now()} Epoch {epoch}, Training loss: {avg_loss:.4f}")