import pandas as pd
import pytest

from drybean_mlp.data import corrupt_labels, load_dry_bean, split_dataset, subsample


def test_load_removes_duplicates_and_separates_target(tmp_path):
    path = tmp_path / "beans.xlsx"
    pd.DataFrame(
        {"Area": [1, 1, 2], "Perimeter": [3, 3, 4], "Class": ["SEKER", "SEKER", "SIRA"]}
    ).to_excel(path, index=False)

    X, y = load_dry_bean(path)

    assert list(X.columns) == ["Area", "Perimeter"]
    assert y.tolist() == ["SEKER", "SIRA"]


def test_split_is_disjoint_complete_and_sized(toy_data):
    X, y = toy_data
    splits = split_dataset(X, y, val_size=0.15, test_size=0.15)

    train, val, test = (set(s.index) for s in (splits.X_train, splits.X_val, splits.X_test))
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(X.index)
    assert len(train) == pytest.approx(700, abs=2)
    assert len(val) == pytest.approx(150, abs=2)
    assert len(test) == pytest.approx(150, abs=2)


def test_split_preserves_class_proportions(toy_data):
    X, y = toy_data
    splits = split_dataset(X, y)
    expected = y.value_counts(normalize=True)
    for part in (splits.y_train, splits.y_val, splits.y_test):
        observed = part.value_counts(normalize=True)
        assert (observed - expected).abs().max() < 0.02


def test_split_is_reproducible(toy_data):
    X, y = toy_data
    assert split_dataset(X, y, seed=1).y_test.index.equals(split_dataset(X, y, seed=1).y_test.index)


def test_subsample_full_fraction_returns_everything(toy_data):
    X, y = toy_data
    X_sample, y_sample = subsample(X, y, 1.0, seed=0)
    assert len(X_sample) == len(y_sample) == len(X)


def test_stratified_subsample_keeps_proportions(toy_data):
    X, y = toy_data
    _, y_sample = subsample(X, y, 0.2, seed=0)
    assert len(y_sample) == 200
    expected = y.value_counts(normalize=True)
    assert (y_sample.value_counts(normalize=True) - expected).abs().max() < 0.02


def test_corrupt_labels_changes_exactly_the_requested_rate(toy_data):
    _, y = toy_data
    noisy = corrupt_labels(y, rate=0.1, seed=0)
    assert (noisy != y).sum() == 100
    assert noisy.index.equals(y.index)
    assert set(noisy) <= set(y)
