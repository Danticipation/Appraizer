from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def make_preprocessor(categorical_cols: Sequence[str], numeric_cols: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_cols)),
            ("num", StandardScaler(), list(numeric_cols)),
        ],
        remainder="drop",
    )


@dataclass
class GBMModel:
    categorical_cols: Sequence[str]
    numeric_cols: Sequence[str]

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("prep", make_preprocessor(self.categorical_cols, self.numeric_cols)),
                ("gbm", HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05, max_iter=450)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


if torch is not None:
    class TabularRegressor(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


@dataclass
class DeepTabularModel:
    categorical_cols: Sequence[str]
    numeric_cols: Sequence[str]
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        self.preprocessor = make_preprocessor(self.categorical_cols, self.numeric_cols)
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        features = self.preprocessor.fit_transform(X)
        dense = features.toarray() if hasattr(features, "toarray") else features
        if torch is None:
            fallback = MLPRegressor(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                max_iter=300,
                random_state=42,
            )
            fallback.fit(dense, y.to_numpy())
            self.model = fallback
            return

        x_tensor = torch.tensor(dense, dtype=torch.float32)
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = TabularRegressor(in_dim=x_tensor.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.HuberLoss(delta=1500.0)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DeepTabularModel is not fitted")
        features = self.preprocessor.transform(X)
        dense = features.toarray() if hasattr(features, "toarray") else features
        if torch is None:
            return self.model.predict(dense)  # type: ignore[union-attr]
        x_tensor = torch.tensor(dense, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(x_tensor).numpy()
        return preds

