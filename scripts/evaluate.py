"""
Evaluation, metrics computation, and feature importance analysis.
Computes test accuracy, confusion matrix, ROC-AUC score, and feature weight impacts.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from .data_loader import load_and_preprocess_data
from .model_builder import LogisticRegressionModel


def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates model performance on held-out test data.

    Args:
        model: Trained PyTorch Module.
        X_test: Held-out test features tensor.
        y_test: Held-out test ground-truth labels tensor.
        threshold: Classification decision threshold.

    Returns:
        Dictionary of accuracy, confusion matrix, and ROC-AUC metrics.
    """
    model.eval()
    with torch.no_grad():
        raw_probabilities = model(X_test).numpy().flatten()
        binary_predictions = (raw_probabilities >= threshold).astype(int)
        y_true = y_test.numpy().flatten()

    acc = accuracy_score(y_true, binary_predictions)
    cm = confusion_matrix(y_true, binary_predictions)
    fpr, tpr, _ = roc_curve(y_true, raw_probabilities)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_true, binary_predictions, target_names=["Defeat (0)", "Victory (1)"])

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "classification_report": report,
        "probabilities": raw_probabilities,
        "predictions": binary_predictions,
    }


def extract_feature_importance(model: LogisticRegressionModel, feature_names: List[str]) -> pd.DataFrame:
    """
    Extracts linear layer weights as directional feature importance indicators.
    """
    weights = model.linear.weight.data.numpy().flatten()
    df = pd.DataFrame({
        "Feature": feature_names,
        "Weight": weights,
        "Abs_Weight": np.abs(weights)
    }).sort_values(by="Weight", ascending=False)
    return df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, _ = load_and_preprocess_data()
    model = LogisticRegressionModel(input_dim=X_train.shape[1])
    
    checkpoint_path = "models/logistic_regression_model.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded weights from {checkpoint_path}")

    results = evaluate_model(model, X_test, y_test)
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print("\nConfusion Matrix:\n", results['confusion_matrix'])
    print("\nClassification Report:\n", results['classification_report'])

    print("\nTop In-Game Metric Weights:")
    fi = extract_feature_importance(model, features)
    print(fi.to_string(index=False))
