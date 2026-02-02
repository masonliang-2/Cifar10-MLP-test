"""
Baseline MLP on CIFAR-10 (PyTorch)

- Uses an 0.8 / 0.1 / 0.1 train/val/test split from the 50k training set.
- Trains a simple MLP on flattened 32x32x3 images.
- Tracks train loss/acc and val loss/acc per epoch, then evaluates once on test.
- Designed to be easy to modify for: architecture, init, optimizer, regularization.
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# -------------------------
# Config
# -------------------------

@dataclass
class Config:
    data_root: str = "./data"
    batch_size: int = 256
    epochs: int = 10
    adam_lr: float = 1e-3
    sgd_lr : float = 0.1
    weight_decay: float = 0.0  # baseline: no L2
    seed: int = 42
    num_workers: int = 2

    # Model
    input_dim: int = 32 * 32 * 3
    num_classes: int = 10
    hidden_dims: Tuple[int, ...] = (1024, 512, 256)  # baseline MLP widths
    dropout_p: float = 0.0  # baseline: no dropout
    
    # Initialization choices
    init_name: str = "kaiming"  # {"kaiming", "xavier"}

    # Optimizer choices (baseline uses Adam)
    optimizer_name: str = "adam"  # {"adam", "sgd"}
    momentum: float = 0.9  # only used for SGD

    # Normalization (recommended)
    use_cifar_norm: bool = True


# -------------------------
# Utilities
# -------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (can slow down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Data
# -------------------------

def make_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # CIFAR-10 normalization constants (common)
    # Mean/std per channel for CIFAR-10 training set
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    tfms = [transforms.ToTensor()]
    if config.use_cifar_norm:
        tfms.append(transforms.Normalize(cifar10_mean, cifar10_std))
    transform = transforms.Compose(tfms)

    full_train = datasets.CIFAR10(
        root=config.data_root, train=True, download=True, transform=transform
    )

    # 50k -> 40k train, 5k val, 5k test (from the original train set)
    n_total = len(full_train)  # 100% = 50000
    n_train = int(0.8 * n_total)  # 80% = 40000
    n_val = int(0.1 * n_total)    # 10% = 5000
    n_test = n_total - n_train - n_val  # 10% = 5000

    g = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds, test_ds = random_split(full_train, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# -------------------------
# Model
# -------------------------

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Tuple[int, ...],
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,32,32] -> flatten
        x = x.view(x.size(0), -1)
        return self.net(x)


def init_weights_kaiming(model: nn.Module) -> None:
    """
    Baseline init: Kaiming (He) for Linear layers + zero bias.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
def init_weights_xavier(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)


# -------------------------
# Train / Eval
# -------------------------

def make_optimizer(cfg: Config, model: nn.Module) -> optim.Optimizer:
    if cfg.optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.adam_lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(), lr=cfg.sgd_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer_name={cfg.optimizer_name}")


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, y) * bsz
        total_n += bsz

    return {
        "loss": total_loss / total_n,
        "acc": total_acc / total_n,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, y) * bsz
        total_n += bsz

    return {
        "loss": total_loss / total_n,
        "acc": total_acc / total_n,
    }


# -------------------------
# Main
# -------------------------

def main() -> None:
    cfg = Config()

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = make_dataloaders(cfg)

    model = MLP(
        input_dim=cfg.input_dim,
        num_classes=cfg.num_classes,
        hidden_dims=cfg.hidden_dims,
        dropout_p=cfg.dropout_p,
    ).to(device)

    # Baseline initialization
    if cfg.init_name.lower() == "kaiming":
        init_weights_kaiming(model)
    elif cfg.init_name.lower() == "xavier":
        init_weights_xavier(model)
    else:
        raise ValueError(f"Unknown init_name={cfg.init_name}")

    print(f"Device: {device}")
    print(f"Model params: {count_params(model):,}")
    print(f"Hidden dims: {cfg.hidden_dims}, dropout_p={cfg.dropout_p}")
    if cfg.optimizer_name.lower() == "adam":
        print(f"Optimizer: {cfg.optimizer_name}, lr={cfg.adam_lr}, weight_decay={cfg.weight_decay}")
    elif cfg.optimizer_name.lower() == "sgd":
        print(f"Optimizer: {cfg.optimizer_name}, lr={cfg.sgd_lr}, weight_decay={cfg.weight_decay}")
    else:
        print(f"Optimizer: {cfg.optimizer_name}, weight_decay={cfg.weight_decay}")
    print(f"Split: 0.8/0.1/0.1 from CIFAR10 train set (50k)")

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg, model)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            # Keep best model by validation accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
        )

    # Load best model (by val acc), then evaluate on test split
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f} | Test acc: {test_metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
