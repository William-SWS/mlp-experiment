import numpy as np
import pandas as pd
import pytest

from drybean_mlp import experiments
from drybean_mlp.data import split_dataset
from drybean_mlp.experiments import (
    Trial,
    aggregate,
    cross_validate_mlp,
    drop_redundant,
    pick_balanced,
    rank_features,
    run_trials,
    top_k,
)
from drybean_mlp.reporting import save_table
from drybean_mlp.training import MLPConfig

FAST = MLPConfig(hidden_layers=(8,), learning_rate=0.05, momentum=0.9, max_epochs=3)


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    """Os testes não leem nem escrevem o cache dos experimentos reais."""
    monkeypatch.setattr(experiments, "_run_trial", experiments._run_trial.func)
    monkeypatch.setattr(experiments, "_rank_with_seed", experiments._rank_with_seed.func)
    monkeypatch.setattr(experiments, "measure_epoch_time", experiments.measure_epoch_time.func)


def test_run_trials_reads_only_the_requested_sets(toy_data):
    X, y = toy_data
    splits = split_dataset(X, y)
    results = run_trials([Trial("base", FAST, seed=0)], splits, monitor=(), n_jobs=1)

    assert not [c for c in results.summary.columns if c.startswith(("val_", "test_"))]
    assert not [c for c in results.histories.columns if c.startswith(("val_", "test_"))]


def test_run_trials_collects_one_row_per_trial_and_one_per_epoch(toy_data):
    X, y = toy_data
    splits = split_dataset(X, y)
    trials = [Trial("a", FAST, seed=0), Trial("a", FAST, seed=1)]
    results = run_trials(trials, splits, monitor=("val", "test"), n_jobs=1)

    assert len(results.summary) == 2
    assert len(results.histories) == 2 * FAST.max_epochs
    assert {"val_f1", "test_f1", "val_loss_min", "val_loss_min_epoch"} <= set(results.summary)
    assert len(results.predictions["a", 0, "test"]) == len(splits.y_test)


def test_train_fraction_changes_only_the_training_set(toy_data):
    X, y = toy_data
    splits = split_dataset(X, y)
    results = run_trials(
        [Trial("half", FAST, seed=0, train_fraction=0.5)], splits, monitor=("val",), n_jobs=1
    )
    assert results.summary.loc[0, "n_train"] == pytest.approx(len(splits.X_train) / 2, abs=1)
    assert len(results.predictions["half", 0, "val"]) == len(splits.X_val)


def test_feature_subset_and_noise_columns_define_the_input_size(toy_data):
    X, y = toy_data
    splits = split_dataset(X, y)
    trial = Trial("few", FAST, seed=0, features=("f0", "f1"), noise_features=3)
    results = run_trials([trial], splits, n_jobs=1)
    assert results.summary.loc[0, "n_features"] == 5


def test_cross_validation_uses_every_row_once_as_test(toy_data):
    X, y = toy_data
    results = cross_validate_mlp(FAST, X, y, k=5, n_jobs=1)

    assert len(results.summary) == 5
    assert results.summary["n_train"].tolist() == [800] * 5
    assert sum(len(v) for v in results.predictions.values()) == len(X)


def test_measure_epoch_time_reports_one_row_per_trial_with_its_data_size(toy_data):
    X, y = toy_data
    splits = split_dataset(X, y)
    trials = [Trial("half", FAST, 0, train_fraction=0.5), Trial("few", FAST, 0, features=("f0",))]
    timing = experiments.measure_epoch_time(trials, splits, epochs=2, repeats=2)

    assert timing["label"].tolist() == ["half", "few"]
    assert timing.loc[0, "n_train"] == pytest.approx(len(splits.X_train) / 2, abs=1)
    assert timing.loc[1, "n_features"] == 1
    assert (timing["ms_per_epoch"] > 0).all()


def test_aggregate_computes_mean_and_std_per_label():
    summary = pd.DataFrame(
        {
            "label": ["a", "a", "b", "b"],
            "seed": [0, 1, 0, 1],
            "hidden_layers": ["8"] * 4,
            "activation": ["relu"] * 4,
            "n_features": [16] * 4,
            "val_f1": [0.8, 0.9, 0.5, 0.5],
            "diverged": [False, True, False, False],
            "reached_target": [False] * 4,
        }
    )
    aggregated = aggregate(summary)
    assert aggregated["label"].tolist() == ["a", "b"]
    assert aggregated.loc[0, "val_f1_mean"] == pytest.approx(0.85)
    assert aggregated.loc[1, "val_f1_std"] == pytest.approx(0.0)
    assert aggregated["runs_diverged"].tolist() == [1, 0]


def test_pick_balanced_prefers_fastest_within_tolerance_and_skips_diverged():
    table = pd.DataFrame(
        {
            "label": ["best", "fast", "too_bad", "diverged"],
            "val_loss_mean": [0.200, 0.205, 0.300, 0.100],
            "epochs_to_aux_mean": [90.0, 20.0, 1.0, 1.0],
            "runs_diverged": [0, 0, 0, 1],
        }
    )
    assert pick_balanced(table)["label"] == "fast"


def test_pick_balanced_uses_quality_when_no_config_reaches_the_speed_threshold():
    table = pd.DataFrame(
        {
            "label": ["worse", "better"],
            "val_loss_mean": [0.205, 0.200],
            "epochs_to_aux_mean": [np.nan, np.nan],
            "runs_diverged": [0, 0],
        }
    )
    assert pick_balanced(table)["label"] == "better"


def test_top_k_penalizes_unstable_configurations():
    table = pd.DataFrame(
        {
            "label": ["unstable", "steady", "weak"],
            "val_f1_mean": [0.95, 0.93, 0.80],
            "val_f1_std": [0.05, 0.002, 0.001],
        }
    )
    assert top_k(table, k=2)["label"].tolist() == ["steady", "unstable"]


def test_rank_features_lists_every_feature_once_sorted_by_importance(toy_data):
    X, y = toy_data
    splits = split_dataset(X, y)
    ranking = rank_features(FAST, splits, seeds=(0, 1), n_jobs=1)

    assert sorted(ranking["feature"]) == sorted(X.columns)
    assert ranking["importance"].is_monotonic_decreasing


def test_drop_redundant_keeps_the_highest_priority_feature_of_each_group():
    base = np.arange(50.0)
    noise = np.random.default_rng(0).normal(size=50)
    X = pd.DataFrame({"a": base, "a_copy": base * 2 + 1, "b": noise})
    assert drop_redundant(X, priority=["a_copy", "a", "b"]) == ("a_copy", "b")


def test_save_table_applies_digits_per_column_only_to_latex(tmp_path):
    table = pd.DataFrame({"ms": [35.678, np.nan], "F1": [0.93812, 0.9]})
    save_table(table, "demo", column_digits={"ms": 1}, csv_dir=tmp_path, table_dir=tmp_path)

    latex = (tmp_path / "demo.tex").read_text(encoding="utf-8")
    assert "35,7 & 0,9381" in latex
    assert "-- & 0,9000" in latex
    assert "35.678" in (tmp_path / "demo.csv").read_text(encoding="utf-8")


def test_save_table_writes_csv_and_latex_with_decimal_comma(tmp_path):
    table = pd.DataFrame({"Taxa_x": [0.5], "Nome": ["a"], "Exemplos": [13543]})
    save_table(table, "demo", digits=2, csv_dir=tmp_path, table_dir=tmp_path)

    assert (tmp_path / "demo.csv").exists()
    latex = (tmp_path / "demo.tex").read_text(encoding="utf-8")
    assert "0,50" in latex
    assert r"Taxa\_x" in latex
    assert "13543" in latex
