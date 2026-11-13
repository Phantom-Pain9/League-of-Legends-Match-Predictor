"""
PyTorch model architecture definition.
Implements a single-layer logistic regression classifier with Sigmoid activation.
"""

import torch
import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    """
    Logistic Regression classifier implemented as a PyTorch Module.
    Maps input feature dimension to a single logit, followed by Sigmoid squashing.
    """
    def __init__(self, input_dim: int):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying linear transformation followed by sigmoid activation.
        """
        return torch.sigmoid(self.linear(x))


def build_model(input_dim: int) -> LogisticRegressionModel:
    """
    Factory helper to initialize and return the Logistic Regression model.
    """
    return LogisticRegressionModel(input_dim=input_dim)
