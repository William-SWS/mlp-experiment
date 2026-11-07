import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from drybean_mlp import plots
from drybean_mlp.reporting import save_figure


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def history() -> pd.DataFrame:
    rows = [
        {
            "label": label,
            "seed": seed,
            "group": group,
            "epoch": epoch,
            "train_loss": 1 / epoch + seed * 0.01,
            "train_accuracy": 1 - 1 / (epoch + 1),
            "val_loss": 1 / epoch + 0.1,
            "val_accuracy": 0.9,
            "test_loss": 1 / epoch + 0.2,
            "test_accuracy": 0.8,
        }
        for label, group in (("a", 1), ("b", 2), ("c", 1))
        for seed in (0, 1)
        for epoch in range(1, 6)
    ]
    return pd.DataFrame(rows)


def test_plot_curves_draws_one_line_per_hue_in_each_panel(history):
    figure = plots.plot_curves(history, hue="label", metrics=("train_loss", "val_loss"))
    assert len(figure.axes) == 2
    assert len(figure.axes[0].lines) == 3


def test_plot_curves_supports_facets_log_scale_and_reference_lines(history):
    figure = plots.plot_curves(
        history, hue="label", facet="group", metrics=("train_loss",), log_y=("train_loss",),
        reference_lines={"train_loss": 0.5}, min_epoch=3, legend_title="Rede",
    )
    assert len(figure.axes) == 2
    assert figure.axes[0].get_yscale() == "log"
    assert figure.axes[0].lines[0].get_xdata().min() == 3


def test_plot_curves_rejects_more_series_than_the_palette_has(history):
    many = pd.concat([history.assign(label=str(i)) for i in range(9)])
    with pytest.raises(ValueError, match="séries"):
        plots.plot_curves(many, hue="label")


@pytest.mark.parametrize("eval_name, metric", [("val", "loss"), ("test", "accuracy")])
def test_plot_train_vs_eval_has_one_panel_per_label(history, eval_name, metric):
    figure = plots.plot_train_vs_eval(history, ["a", "b", "c"], eval_name, metric, ncols=2)
    visible = [ax for ax in figure.axes if ax.get_visible()]
    assert [ax.get_title() for ax in visible] == ["a", "b", "c"]
    assert len(visible[0].lines) == 2


def test_plot_heatmaps_draws_each_pivot():
    pivot = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=[1, 2], columns=[10, 20])
    figure = plots.plot_heatmaps(
        {"A": pivot, "B": pivot}, fmt={"A": ".1f", "B": ".2f"}, reverse=("B",), log_color=("A",)
    )
    assert [ax.get_title() for ax in figure.axes] == ["A", "B"]


def test_plot_by_size_draws_one_panel_per_group():
    table = pd.DataFrame(
        {"n": [10, 20], "val_f1_mean": [0.8, 0.9], "val_f1_std": [0.01, 0.01],
         "test_f1_mean": [0.7, 0.8], "test_f1_std": [0.01, 0.01]}
    )
    figure = plots.plot_by_size(table, "n", {"F": ("val_f1", "test_f1"), "G": ("val_f1",)}, "N")
    assert len(figure.axes) == 2


def test_plot_by_size_accepts_a_plain_column_without_spread():
    table = pd.DataFrame({"n": [10, 20], "ms_per_epoch": [1.0, 2.0]})
    figure = plots.plot_by_size(table, "n", {"Tempo": ("ms_per_epoch",)}, "N")
    assert figure.axes[0].get_legend().get_texts()[0].get_text() == "Tempo por época (ms)"


@pytest.mark.parametrize("error", [None, "dp"])
def test_plot_dots_accepts_optional_error(error):
    table = pd.DataFrame({"name": ["a", "b"], "v": [0.1, 0.2], "dp": [0.01, 0.02]})
    figure = plots.plot_dots(table, "name", "v", error, x_label="x", reference=0.15)
    assert figure.axes[0].get_xlabel() == "x"


def test_plot_confusion_writes_counts_for_every_cell():
    y_true = ["a", "a", "b"]
    y_pred = ["a", "b", "b"]
    figure = plots.plot_confusion(y_true, y_pred, ["a", "b"])
    assert sorted(text.get_text() for text in figure.axes[0].texts) == ["0", "1", "1", "1"]


def test_data_overview_plots_render():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 3)), columns=["x", "y", "z"])
    y = pd.Series(["a"] * 15 + ["b"] * 5)
    assert plots.plot_correlation(X).axes
    assert plots.plot_class_counts(y).axes


def test_save_figure_writes_pdf_and_png(tmp_path):
    figure, _ = plt.subplots()
    save_figure(figure, "demo", figure_dir=tmp_path)
    assert (tmp_path / "demo.pdf").exists()
    assert (tmp_path / "demo.png").exists()
