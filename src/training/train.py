from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


@dataclass
class TrainingResult:
    train_losses: list[float]
    validation_losses: list[float]


def resolve_device(device: str = "auto") -> torch.device:
    """
    Resolve the training device.
    """
    normalized_device = device.lower().strip()

    if normalized_device == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if normalized_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if normalized_device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: '{device}'. "
        "Available: auto, cuda, cpu."
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(features)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        batch_size = features.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on a validation/test dataset.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        predictions = model(features)
        loss = criterion(predictions, targets)

        batch_size = features.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    criterion: nn.Module,
    device: str = "auto",
) -> TrainingResult:
    """
    Full baseline training loop.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    device = resolve_device(device)
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    train_losses: list[float] = []
    validation_losses: list[float] = []

    for _ in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        validation_loss = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(validation_loss)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

    return TrainingResult(
        train_losses=train_losses,
        validation_losses=validation_losses,
    )