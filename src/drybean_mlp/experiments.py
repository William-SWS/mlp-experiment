"""Execução paralela de treinamentos, agregação por seed e regras de seleção."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from drybean_mlp.data import PROJECT_ROOT, DataSplits, corrupt_labels, subsample
from drybean_mlp.training import MLPConfig, evaluate, train_mlp

AUXILIARY_LOSS = 0.20
"""Limiar usado para comparar velocidade quando a perda de 0,001 não é atingida."""

_memory = Memory(PROJECT_ROOT / "results" / ".cache", verbose=0)
_DESCRIPTORS = ("hidden_layers", "activation", "learning_rate", "momentum", "alpha", "n_features")
_NOT_AGGREGATED = {*_DESCRIPTORS, "seed", "train_fraction", "label_noise"}


@dataclass(frozen=True)
class Trial:
    """Um treinamento: configuração, seed e, opcionalmente, como degradar os dados de treino."""

    label: str
    config: MLPConfig
    seed: int
    features: tuple[str, ...] | None = None
    train_fraction: float = 1.0
    stratified: bool = True
    label_noise: float = 0.0
    noise_features: int = 0


@dataclass(frozen=True)
class TrialResults:
    """Saída de `run_trials`: uma linha por treinamento e uma linha por época.

    `predictions` guarda, para cada `(label, seed, conjunto)`, os rótulos previstos no fim do
    treino, o que evita retreinar para montar uma matriz de confusão.
    """

    summary: pd.DataFrame
    histories: pd.DataFrame
    predictions: dict[tuple[str, int, str], np.ndarray]


def _add_noise_columns(X: pd.DataFrame, count: int, rng: np.random.Generator) -> pd.DataFrame:
    noise = {f"noise_{i}": rng.normal(size=len(X)) for i in range(count)}
    return X.assign(**noise)


def _prepare_data(trial: Trial, splits: DataSplits, monitor: Sequence[str]):
    X_train, y_train = subsample(
        splits.X_train, splits.y_train, trial.train_fraction, trial.seed, trial.stratified
    )
    if trial.label_noise:
        y_train = corrupt_labels(y_train, trial.label_noise, trial.seed)

    eval_sets = {name: (getattr(splits, f"X_{name}"), getattr(splits, f"y_{name}"))
                 for name in monitor}
    if trial.features is not None:
        columns = list(trial.features)
        X_train = X_train[columns]
        eval_sets = {name: (X[columns], y) for name, (X, y) in eval_sets.items()}
    if trial.noise_features:
        rng = np.random.default_rng(trial.seed)
        X_train = _add_noise_columns(X_train, trial.noise_features, rng)
        eval_sets = {
            name: (_add_noise_columns(X, trial.noise_features, rng), y)
            for name, (X, y) in eval_sets.items()
        }
    return X_train, y_train, eval_sets


@_memory.cache
def _run_trial(trial: Trial, splits: DataSplits, monitor: tuple[str, ...]):
    X_train, y_train, eval_sets = _prepare_data(trial, splits, monitor)
    run = train_mlp(trial.config, X_train, y_train, trial.seed, eval_sets)

    config = trial.config
    row = {
        "label": trial.label,
        "seed": trial.seed,
        "hidden_layers": "×".join(map(str, config.hidden_layers)),
        "activation": config.activation,
        "learning_rate": config.learning_rate,
        "momentum": config.momentum,
        "alpha": config.alpha,
        "n_train": len(X_train),
        "n_features": X_train.shape[1],
        "train_fraction": trial.train_fraction,
        "label_noise": trial.label_noise,
        "epochs": run.epochs,
        "epochs_to_target": run.epochs if run.reached_target else np.nan,
        "epochs_to_aux": run.epochs_to_loss(AUXILIARY_LOSS),
        "reached_target": run.reached_target,
        "diverged": run.diverged,
        "train_seconds": run.train_seconds,
    }
    predictions = {}
    for name, (X, y) in {"train": (X_train, y_train), **eval_sets}.items():
        metrics = evaluate(run.model, X, y) if not run.diverged else {}
        for metric in ("loss", "accuracy", "f1", "precision", "recall"):
            row[f"{name}_{metric}"] = metrics.get(metric, np.nan)
        if name != "train" and not run.diverged:
            predictions[(trial.label, trial.seed, name)] = run.model.predict(X)
    for name in monitor:
        losses = run.history[f"{name}_loss"]
        row[f"{name}_loss_min"] = losses.min()
        row[f"{name}_loss_min_epoch"] = (
            run.history.loc[losses.idxmin(), "epoch"] if losses.notna().any() else np.nan
        )

    history = run.history.assign(label=trial.label, seed=trial.seed)
    return row, history, predictions


def _collect(outcomes: list[tuple]) -> TrialResults:
    rows, histories, predictions = zip(*outcomes, strict=True)
    merged: dict[tuple[str, int, str], np.ndarray] = {}
    for item in predictions:
        merged.update(item)
    return TrialResults(pd.DataFrame(rows), pd.concat(histories, ignore_index=True), merged)


def run_trials(
    trials: Iterable[Trial],
    splits: DataSplits,
    monitor: Sequence[str] = ("val",),
    n_jobs: int = -1,
) -> TrialResults:
    """Executa os treinamentos em paralelo, com cache em disco.

    `monitor` diz quais conjuntos são lidos, além do treino. O conjunto de teste só é usado
    quando a etapa o pede aqui de forma explícita.
    """
    monitor = tuple(monitor)
    outcomes = Parallel(n_jobs=n_jobs)(
        delayed(_run_trial)(trial, splits, monitor) for trial in trials
    )
    return _collect(outcomes)


def cross_validate_mlp(
    config: MLPConfig,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 10,
    seed: int = 42,
    n_jobs: int = -1,
) -> TrialResults:
    """K-fold estratificado: cada partição é o `test` de uma rodada, com z-score próprio.

    A seed de inicialização é a mesma em todos os folds, de modo que a variação observada
    venha das partições, não dos pesos iniciais.
    """
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    empty_X, empty_y = X.iloc[:0], y.iloc[:0]
    jobs = []
    for fold, (train_index, test_index) in enumerate(folds.split(X, y), start=1):
        splits = DataSplits(
            X.iloc[train_index], empty_X, X.iloc[test_index],
            y.iloc[train_index], empty_y, y.iloc[test_index],
        )
        jobs.append((Trial(f"fold {fold}", config, seed), splits))
    outcomes = Parallel(n_jobs=n_jobs)(
        delayed(_run_trial)(trial, splits, ("test",)) for trial, splits in jobs
    )
    return _collect(outcomes)


@_memory.cache
def measure_epoch_time(
    trials: Sequence[Trial], splits: DataSplits, epochs: int = 30, repeats: int = 5
) -> pd.DataFrame:
    """Milissegundos por época, um treino por vez e em uma thread.

    Os tempos de `run_trials` não servem para comparar configurações: dependem de quantos
    processos dividem a máquina naquele momento. Aqui as repetições são intercaladas entre as
    configurações e vale o mínimo, porque a interferência externa só aumenta o tempo. Só o
    treino entra na medida, sem monitoramento.
    """
    prepared = [_prepare_data(trial, splits, ()) for trial in trials]
    samples: list[list[float]] = [[] for _ in trials]
    with threadpool_limits(limits=1):
        for seed in range(repeats):
            for trial, (X_train, y_train, _), times in zip(trials, prepared, samples, strict=True):
                config = replace(trial.config, max_epochs=epochs, target_loss=0.0)
                run = train_mlp(config, X_train, y_train, seed)
                times.append(1000 * run.train_seconds / epochs)
    return pd.DataFrame(
        {
            "label": [trial.label for trial in trials],
            "n_train": [len(X_train) for X_train, _, _ in prepared],
            "n_features": [X_train.shape[1] for X_train, _, _ in prepared],
            "ms_per_epoch": [min(times) for times in samples],
        }
    )


def aggregate(summary: pd.DataFrame, by: str = "label") -> pd.DataFrame:
    """Média e desvio-padrão por configuração, na ordem em que as configurações aparecem."""
    numeric = [
        column
        for column in summary.select_dtypes("number").columns
        if column not in _NOT_AGGREGATED and column != by
    ]
    grouped = summary.groupby(by, sort=False)
    stats = grouped[numeric].agg(["mean", "std"])
    stats.columns = [f"{column}_{stat}" for column, stat in stats.columns]
    descriptors = [column for column in _DESCRIPTORS if column in summary]
    extras = grouped[descriptors].first().assign(
        runs=grouped.size(),
        runs_diverged=grouped["diverged"].sum(),
        runs_reached_target=grouped["reached_target"].sum(),
    )
    return extras.join(stats).reset_index()


def pick_balanced(
    aggregated: pd.DataFrame,
    quality: str = "val_loss_mean",
    speed: str = "epochs_to_aux_mean",
    tolerance: float = 0.05,
) -> pd.Series:
    """Entre as configurações estáveis e até `tolerance` da melhor qualidade, a mais rápida.

    Diferenças pequenas de perda ficam dentro do ruído entre seeds; nesse caso vale preferir a
    configuração que converge antes. Se nenhuma atingir o limiar de velocidade, desempata pela
    qualidade.
    """
    stable = aggregated[(aggregated["runs_diverged"] == 0) & aggregated[quality].notna()]
    candidates = stable[stable[quality] <= stable[quality].min() * (1 + tolerance)]
    return candidates.sort_values([speed, quality], na_position="last").iloc[0]


def top_k(aggregated: pd.DataFrame, k: int = 4, metric: str = "val_f1") -> pd.DataFrame:
    """As `k` configurações com maior média menos desvio: pontua desempenho e estabilidade."""
    score = aggregated[f"{metric}_mean"] - aggregated[f"{metric}_std"].fillna(0)
    return aggregated.assign(score=score).nlargest(k, "score").reset_index(drop=True)


def permutation_ranking(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int, n_repeats: int = 10
) -> pd.DataFrame:
    """Queda de F1-macro ao embaralhar cada atributo, em ordem decrescente de importância."""
    result = permutation_importance(
        model, X, y, scoring="f1_macro", n_repeats=n_repeats, random_state=seed
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


@_memory.cache
def _rank_with_seed(config: MLPConfig, splits: DataSplits, seed: int) -> pd.DataFrame:
    run = train_mlp(config, splits.X_train, splits.y_train, seed)
    return permutation_ranking(run.model, splits.X_val, splits.y_val, seed)


def rank_features(
    config: MLPConfig, splits: DataSplits, seeds: Sequence[int], n_jobs: int = -1
) -> pd.DataFrame:
    """Importância por permutação na validação, com média entre redes de seeds diferentes."""
    rankings = Parallel(n_jobs=n_jobs)(
        delayed(_rank_with_seed)(config, splits, seed) for seed in seeds
    )
    return (
        pd.concat(rankings)
        .groupby("feature", as_index=False)
        .agg(importance=("importance", "mean"), importance_std=("importance", "std"))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def drop_redundant(
    X_train: pd.DataFrame, priority: Sequence[str], threshold: float = 0.95
) -> tuple[str, ...]:
    """Percorre `priority` e mantém um atributo só se |r| ≤ `threshold` com os já mantidos."""
    correlation = X_train.corr().abs()
    kept: list[str] = []
    for feature in priority:
        if all(correlation.loc[feature, other] <= threshold for other in kept):
            kept.append(feature)
    return tuple(kept)
