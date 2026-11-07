"""Treinamento de uma MLP com monitoramento época a época."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EvalSets = Mapping[str, tuple[pd.DataFrame, pd.Series]]


@dataclass(frozen=True)
class MLPConfig:
    """Hiperparâmetros de um treinamento; a saída softmax e a entropia cruzada são fixas."""

    hidden_layers: tuple[int, ...] = (50, 50)
    activation: str = "relu"
    learning_rate: float = 0.01
    momentum: float = 0.9
    alpha: float = 1e-4
    batch_size: int = 200
    max_epochs: int = 300
    target_loss: float = 0.001


@dataclass(frozen=True)
class TrainingRun:
    """Resultado de `train_mlp`.

    `history` tem uma linha por época, com `<conjunto>_loss` e `<conjunto>_accuracy` para o
    treino e para cada conjunto monitorado. `train_seconds` não inclui o monitoramento.
    """

    model: Pipeline
    history: pd.DataFrame
    train_seconds: float
    diverged: bool
    reached_target: bool

    @property
    def epochs(self) -> int:
        return len(self.history)

    def epochs_to_loss(self, threshold: float) -> float:
        """Primeira época com perda de treino ≤ `threshold`; NaN se nunca atingida."""
        hits = self.history.loc[self.history["train_loss"] <= threshold, "epoch"]
        return float(hits.iloc[0]) if len(hits) else float("nan")


def _build_classifier(config: MLPConfig, seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=config.hidden_layers,
        activation=config.activation,
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=config.learning_rate,
        momentum=config.momentum,
        nesterovs_momentum=False,  # o enunciado estuda o termo momento clássico
        alpha=config.alpha,
        batch_size=config.batch_size,
        # Instância, não inteiro: a cada chamada o scikit-learn passa `random_state` por
        # check_random_state, e um inteiro repetiria o mesmo embaralhamento em toda época.
        random_state=np.random.RandomState(seed),
    )


def _epoch_metrics(
    classifier: MLPClassifier, X: np.ndarray, codes: np.ndarray, prefix: str
) -> dict[str, float]:
    probabilities = classifier.predict_proba(X)
    if not np.isfinite(probabilities).all():
        return {f"{prefix}_loss": float("nan"), f"{prefix}_accuracy": float("nan")}
    labels = np.arange(probabilities.shape[1])
    return {
        f"{prefix}_loss": log_loss(codes, probabilities, labels=labels),
        f"{prefix}_accuracy": float((probabilities.argmax(axis=1) == codes).mean()),
    }


def train_mlp(
    config: MLPConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    eval_sets: EvalSets | None = None,
) -> TrainingRun:
    """Treina até a perda de treino atingir `target_loss` ou até `max_epochs`.

    O z-score é ajustado apenas em `X_train`. Os `eval_sets` só são lidos para registrar
    métricas e nunca influenciam os pesos. Uma execução que produz pesos não finitos termina
    com `diverged=True` em vez de levantar exceção, para não derrubar uma grade inteira.

    O laço usa `partial_fit` porque `fit` com `warm_start=True` recria o otimizador a cada
    chamada, o que zeraria a velocidade do momento no início de toda época.
    """
    eval_sets = eval_sets or {}
    scaler = StandardScaler().fit(X_train)
    classifier = _build_classifier(config, seed)
    classes = np.unique(np.concatenate([y_train, *(y for _, y in eval_sets.values())]))

    X_train_scaled = scaler.transform(X_train)
    y_train_array = y_train.to_numpy()
    monitored = {
        "train": (X_train_scaled, np.searchsorted(classes, y_train_array)),
        **{
            name: (scaler.transform(X), np.searchsorted(classes, y.to_numpy()))
            for name, (X, y) in eval_sets.items()
        },
    }

    rows: list[dict[str, float]] = []
    train_seconds = 0.0
    diverged = reached_target = False
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                classifier.partial_fit(X_train_scaled, y_train_array, classes=classes)
        except ValueError:  # pesos não finitos, detectados pelo próprio scikit-learn
            diverged = True
            break
        train_seconds += time.perf_counter() - started

        row: dict[str, float] = {"epoch": epoch}
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for name, (X, codes) in monitored.items():
                row |= _epoch_metrics(classifier, X, codes, name)
        if not np.isfinite(row["train_loss"]):
            diverged = True
            break
        rows.append(row)
        if row["train_loss"] <= config.target_loss:
            reached_target = True
            break

    columns = ["epoch"] + [f"{name}_{m}" for name in monitored for m in ("loss", "accuracy")]
    return TrainingRun(
        model=Pipeline([("scaler", scaler), ("mlp", classifier)]),
        history=pd.DataFrame(rows, columns=columns),
        train_seconds=train_seconds,
        diverged=diverged,
        reached_target=reached_target,
    )


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Perda, acurácia e medidas macro: cada classe pesa igual, inclusive a minoritária."""
    probabilities = model.predict_proba(X)
    if not np.isfinite(probabilities).all():
        return dict.fromkeys(("loss", "accuracy", "f1", "precision", "recall"), float("nan"))
    predictions = model.predict(X)
    classes = model.named_steps["mlp"].classes_
    return {
        "loss": log_loss(y, probabilities, labels=classes),
        "accuracy": accuracy_score(y, predictions),
        "f1": f1_score(y, predictions, average="macro", zero_division=0),
        "precision": precision_score(y, predictions, average="macro", zero_division=0),
        "recall": recall_score(y, predictions, average="macro", zero_division=0),
    }
