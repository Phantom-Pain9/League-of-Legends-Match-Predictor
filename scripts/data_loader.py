"""
Data ingestion and preprocessing module.
Handles dataset loading, stratified splitting, feature standardization,
and PyTorch tensor conversion.
"""

from typing import Tuple, List
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv"


def load_and_preprocess_data(
    data_source: str = DEFAULT_DATA_URL,
    target_col: str = "win",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], StandardScaler]:
    """
    Loads League of Legends in-game statistics, splits into train/test sets,
    normalizes features via z-score scaling, and converts arrays to PyTorch tensors.

    Args:
        data_source: File path or HTTP URL to the CSV dataset.
        target_col: Name of the binary target label column.
        test_size: Proportion of the dataset allocated to testing.
        random_state: Seed for deterministic splitting.

    Returns:
        X_train: Standardized training feature tensor [N, D].
        X_test: Standardized testing feature tensor [M, D].
        y_train: Binary training labels tensor [N, 1].
        y_test: Binary testing labels tensor [M, 1].
        feature_names: List of feature column names.
        scaler: Fitted StandardScaler instance.
    """
    df = pd.read_csv(data_source)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")
        
    X_df = df.drop(columns=[target_col])
    y_series = df[target_col]
    feature_names = list(X_df.columns)

    # Train / test split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_df, y_series, test_size=test_size, random_state=random_state
    )

    # Feature standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Convert to PyTorch float32 tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw.values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test_raw.values, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test, feature_names, scaler
