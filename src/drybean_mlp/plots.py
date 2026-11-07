"""Gráficos dos experimentos; cada função recebe dados prontos e devolve uma `Figure`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.metrics import confusion_matrix

# Paleta categórica validada (skill dataviz, modo claro); a ordem é fixa e nunca rotaciona.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
_INK = "#1a1a19"
_MUTED = "#6f6e68"
_GRID = "#e4e3de"
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#eff5fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)
_DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#184f95", "#6da7ec", "#ecebe6", "#f09a73", "#b23f14"]
)

METRIC_LABELS = {
    "train_loss": "Perda (treino)",
    "val_loss": "Perda (validação)",
    "test_loss": "Perda (teste)",
    "train_accuracy": "Acurácia (treino)",
    "val_accuracy": "Acurácia (validação)",
    "test_accuracy": "Acurácia (teste)",
    "train_f1": "Medida-F (treino)",
    "val_f1": "Medida-F (validação)",
    "test_f1": "Medida-F (teste)",
    "ms_per_epoch": "Tempo por época (ms)",
}


def set_style() -> None:
    """Grade discreta, sem moldura e tipografia legível quando impressa."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": _MUTED,
            "axes.labelcolor": _INK,
            "axes.titlecolor": _INK,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": _GRID,
            "grid.linewidth": 0.8,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "font.size": 10,
            "pdf.fonttype": 42,
        }
    )


def _legend_on_top(figure: plt.Figure, axes: np.ndarray, title: str | None = None) -> None:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=min(len(labels), 6),
            borderaxespad=0,
            title=title,
        )


def plot_curves(
    history: pd.DataFrame,
    hue: str,
    metrics: Sequence[str] = ("train_loss", "train_accuracy"),
    facet: str | None = None,
    hue_order: Sequence | None = None,
    facet_order: Sequence | None = None,
    log_y: Sequence[str] = (),
    reference_lines: Mapping[str, float] | None = None,
    min_epoch: int = 1,
    legend_title: str | None = None,
) -> plt.Figure:
    """Curvas por época: uma linha por nível de `hue`, com média e faixa de ±1 desvio.

    Cada métrica ocupa uma linha de painéis e cada nível de `facet`, uma coluna; nunca há dois
    eixos y no mesmo painel. `min_epoch` omite o transiente inicial, que achataria o resto da
    escala. Aceita no máximo oito séries.
    """
    hue_order = list(hue_order if hue_order is not None else history[hue].drop_duplicates())
    if len(hue_order) > len(SERIES):
        raise ValueError(f"Máximo de {len(SERIES)} séries; use painéis para separar o restante.")
    columns = [None] if facet is None else list(
        facet_order if facet_order is not None else history[facet].drop_duplicates()
    )
    figure, axes = plt.subplots(
        len(metrics),
        len(columns),
        figsize=(4.4 * len(columns) if len(columns) > 1 else 8.5, 2.9 * len(metrics) + 0.5),
        sharex=True,
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    reference_lines = reference_lines or {}
    for row_index, metric in enumerate(metrics):
        for column_index, level in enumerate(columns):
            ax = axes[row_index, column_index]
            data = history if facet is None else history[history[facet] == level]
            for color, series in zip(SERIES, hue_order, strict=False):
                stats = data[data[hue] == series].groupby("epoch")[metric].agg(["mean", "std"])
                stats = stats[stats.index >= min_epoch]
                ax.plot(stats.index, stats["mean"], color=color, label=str(series))
                if stats["std"].notna().any():
                    low, high = stats["mean"] - stats["std"], stats["mean"] + stats["std"]
                    if metric in log_y:  # a faixa não pode cruzar o zero em escala logarítmica
                        low = low.clip(lower=stats["mean"] / 2)
                    ax.fill_between(stats.index, low, high, color=color, alpha=0.15, linewidth=0)
            if metric in reference_lines:
                ax.axhline(reference_lines[metric], color=_MUTED, linestyle="--", linewidth=1)
            if metric in log_y:
                ax.set_yscale("log")
            if row_index == 0 and facet is not None:
                ax.set_title(str(level) if "=" in str(level) else f"{facet} = {level}")
            if column_index == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            if row_index == len(metrics) - 1:
                ax.set_xlabel("Época")
    _legend_on_top(figure, axes, legend_title)
    return figure


def plot_train_vs_eval(
    history: pd.DataFrame,
    labels: Sequence[str],
    eval_name: str = "val",
    metric: str = "loss",
    ncols: int = 2,
    min_epoch: int = 1,
) -> plt.Figure:
    """Um painel por configuração, com a curva de treino e a do conjunto avaliado."""
    nrows = int(np.ceil(len(labels) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 2.7 * nrows + 0.5),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    names = {"train": "Treino", "val": "Validação", "test": "Teste"}
    for ax, label in zip(axes.flat, labels, strict=False):
        data = history[history["label"] == label]
        for color, split in zip(SERIES, ("train", eval_name), strict=False):
            stats = data.groupby("epoch")[f"{split}_{metric}"].agg(["mean", "std"])
            stats = stats[stats.index >= min_epoch]
            ax.plot(stats.index, stats["mean"], color=color, label=names[split])
            if stats["std"].notna().any():
                low, high = stats["mean"] - stats["std"], stats["mean"] + stats["std"]
                ax.fill_between(stats.index, low, high, color=color, alpha=0.15, linewidth=0)
        ax.set_title(str(label))
    for ax in axes.flat[len(labels) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Época")
    for ax in axes[:, 0]:
        ax.set_ylabel("Perda" if metric == "loss" else "Acurácia")
    _legend_on_top(figure, axes)
    return figure


def plot_heatmaps(
    pivots: Mapping[str, pd.DataFrame],
    fmt: str | Mapping[str, str] = ".3f",
    reverse: Sequence[str] = (),
    log_color: Sequence[str] = (),
    ncols: int | None = None,
) -> plt.Figure:
    """Heatmaps sequenciais lado a lado.

    Títulos em `reverse` têm o escuro no menor valor. Títulos em `log_color` usam cor em escala
    logarítmica, para que um único valor extremo não apague o contraste do restante.
    """
    ncols = ncols or len(pivots)
    nrows = int(np.ceil(len(pivots) / ncols))
    first = next(iter(pivots.values()))
    panel_width = max(4.2, 0.62 * first.shape[1] + 1.6)
    panel_height = 0.55 * first.shape[0] + 1.5
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False,
        layout="constrained",
    )
    for ax, (title, pivot) in zip(axes.flat, pivots.items(), strict=False):
        cmap = _SEQUENTIAL.reversed() if title in reverse else _SEQUENTIAL
        sns.heatmap(
            pivot, annot=True, fmt=fmt if isinstance(fmt, str) else fmt[title], cmap=cmap,
            norm=LogNorm() if title in log_color else None, linewidths=2, linecolor="white",
            cbar=False, annot_kws={"fontsize": 8}, ax=ax,
        )
        ax.set_title(title)
        ax.tick_params(length=0, rotation=0)
    return figure


def plot_by_size(
    aggregated: pd.DataFrame,
    x: str,
    metrics_by_panel: Mapping[str, Sequence[str]],
    x_label: str,
) -> plt.Figure:
    """Métricas em função de `x`: média ± desvio ou, sem `<metric>_mean`, o valor da coluna."""
    figure, axes = plt.subplots(
        1,
        len(metrics_by_panel),
        figsize=(4.2 * len(metrics_by_panel), 3.4),
        squeeze=False,
        layout="constrained",
    )
    for ax, (title, metrics) in zip(axes.flat, metrics_by_panel.items(), strict=True):
        for color, metric in zip(SERIES, metrics, strict=False):
            has_spread = f"{metric}_mean" in aggregated
            ax.errorbar(
                aggregated[x],
                aggregated[f"{metric}_mean" if has_spread else metric],
                yerr=aggregated[f"{metric}_std"] if has_spread else None,
                color=color, marker="o", markersize=5, capsize=2.5,
                label=METRIC_LABELS.get(metric, metric),
            )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.legend(loc="best")
    return figure


def plot_dots(
    table: pd.DataFrame,
    label: str,
    value: str,
    error: str | None = None,
    x_label: str = "",
    reference: float | None = None,
    height_per_row: float = 0.3,
) -> plt.Figure:
    """Comparação de valores próximos: como o eixo não parte de zero, usa pontos, não barras."""
    ordered = table.iloc[::-1]
    figure, ax = plt.subplots(
        figsize=(7.5, height_per_row * len(table) + 1.1), layout="constrained"
    )
    ax.errorbar(
        ordered[value], ordered[label].astype(str), xerr=None if error is None else ordered[error],
        fmt="o", color=SERIES[0], markersize=6, capsize=2.5, linewidth=1.2,
    )
    if reference is not None:
        ax.axvline(reference, color=_MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return figure


def plot_confusion(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> plt.Figure:
    """Matriz de confusão: a cor é a taxa por classe real; o texto, a contagem."""
    counts = confusion_matrix(y_true, y_pred, labels=classes)
    rates = counts / counts.sum(axis=1, keepdims=True)
    figure, ax = plt.subplots(figsize=(6.4, 5.2), layout="constrained")
    sns.heatmap(
        rates, annot=counts, fmt="d", cmap=_SEQUENTIAL, vmin=0, vmax=1, linewidths=2,
        linecolor="white", xticklabels=classes, yticklabels=classes,
        cbar_kws={"label": "Taxa por classe real", "shrink": 0.8}, annot_kws={"fontsize": 9}, ax=ax,
    )
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.tick_params(length=0)
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="y", rotation=0)
    return figure


def plot_correlation(X: pd.DataFrame) -> plt.Figure:
    """Correlação de Pearson entre atributos, com escala divergente centrada em zero."""
    figure, ax = plt.subplots(figsize=(8.2, 6.8), layout="constrained")
    sns.heatmap(
        X.corr(), annot=True, fmt=".2f", cmap=_DIVERGING, vmin=-1, vmax=1, linewidths=1,
        linecolor="white", square=True, annot_kws={"fontsize": 6.5},
        cbar_kws={"label": "r de Pearson", "shrink": 0.75}, ax=ax,
    )
    ax.tick_params(length=0)
    return figure


def plot_class_counts(y: pd.Series) -> plt.Figure:
    """Exemplos por classe, com a contagem escrita ao lado de cada barra."""
    counts = y.value_counts().sort_values()
    figure, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    ax.barh(counts.index, counts.to_numpy(), color=SERIES[0], height=0.6)
    for position, value in enumerate(counts.to_numpy()):
        ax.text(value + 40, position, f"{value:,}".replace(",", "."), va="center", color=_INK)
    ax.set_xlabel("Exemplos")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, counts.max() * 1.1)
    return figure
