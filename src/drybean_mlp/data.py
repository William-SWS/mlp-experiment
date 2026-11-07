"""Carregamento, divisão e degradação controlada da base Dry Bean."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
DATASET_PATH = PROJECT_ROOT / "data" / "Dry_Bean_Dataset.xlsx"
TARGET_COLUMN = "Class"

# Agrupamento físico dos atributos, usado na Questão 5.
SIZE_FEATURES = (
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "EquivDiameter",
)
SHAPE_FEATURES = (
    "AspectRation",
    "Eccentricity",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
)


@dataclass(frozen=True)
class DataSplits:
    """Partições estratificadas e disjuntas; o teste só é lido a partir da Questão 4."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def _download_dataset(path: Path) -> None:
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    member = next(name for name in archive.namelist() if name.endswith(path.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive.read(member))


def load_dry_bean(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Devolve atributos e rótulos sem linhas duplicadas, baixando a base se preciso.

    Duplicatas são removidas antes de qualquer divisão: se ficassem, a mesma linha poderia
    cair no treino e no teste ao mesmo tempo.
    """
    if not path.exists():
        _download_dataset(path)
    frame = pd.read_excel(path).drop_duplicates().reset_index(drop=True)
    y = frame.pop(TARGET_COLUMN).astype(str)
    return frame.astype(float), y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> DataSplits:
    """Divide em treino, validação e teste mantendo a proporção das classes."""
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1 - test_size), stratify=y_rest, random_state=seed
    )
    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def subsample(
    X: pd.DataFrame, y: pd.Series, fraction: float, seed: int, stratified: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Sorteia uma fração do conjunto; `fraction=1.0` devolve o conjunto inteiro."""
    if fraction >= 1.0:
        return X, y
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=fraction, stratify=y if stratified else None, random_state=seed
    )
    return X_sample, y_sample


def corrupt_labels(y: pd.Series, rate: float, seed: int) -> pd.Series:
    """Troca uma fração dos rótulos por outra classe sorteada (nunca a original)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    labels = y.to_numpy(copy=True)
    flipped = rng.choice(len(labels), size=round(rate * len(labels)), replace=False)
    for position in flipped:
        others = classes[classes != labels[position]]
        labels[position] = rng.choice(others)
    return pd.Series(labels, index=y.index, name=y.name)
