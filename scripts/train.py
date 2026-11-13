"""
Model training and optimization script.
Executes batch gradient descent with optional L2 regularization (weight decay).
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from .model_builder import LogisticRegressionModel
from .data_loader import load_and_preprocess_data


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    log_interval: int = 100
) -> Dict[str, Any]:
    """
    Trains the logistic regression classifier using SGD and Binary Cross-Entropy.

    Args:
        model: PyTorch nn.Module instance.
        X_train: Training feature tensor.
        y_train: Training target tensor.
        epochs: Number of complete passes over the dataset.
        lr: Optimization learning rate.
        weight_decay: L2 penalty parameter for weight regularization.
        log_interval: Frequency of printing training progress.

    Returns:
        Dictionary containing epoch loss history.
    """
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_history: List[float] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:04d}/{epochs:04d}] | Loss: {loss_val:.4f}")

    return {"loss_history": loss_history}


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test, features, _ = load_and_preprocess_data()
    print(f"Dataset ready. Train size: {X_train.shape[0]}, Features: {X_train.shape[1]}")

    model = LogisticRegressionModel(input_dim=X_train.shape[1])
    print("Starting training with L2 Regularization (weight_decay=0.01)...")
    train_model(model, X_train, y_train, epochs=1000, lr=0.01, weight_decay=0.01)

    # Save trained checkpoint
    torch.save(model.state_dict(), "models/logistic_regression_model.pth")
    print("Model checkpoint saved to models/logistic_regression_model.pth")
