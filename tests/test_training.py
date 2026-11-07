import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import _multilayer_perceptron as sk_mlp

from drybean_mlp.training import MLPConfig, evaluate, train_mlp

FAST = MLPConfig(hidden_layers=(8,), learning_rate=0.05, momentum=0.9, max_epochs=5)


def test_history_has_one_row_per_epoch_and_every_monitored_set(toy_data):
    X, y = toy_data
    run = train_mlp(FAST, X[:800], y[:800], seed=0, eval_sets={"val": (X[800:], y[800:])})

    assert run.epochs == 5
    assert run.history["epoch"].tolist() == [1, 2, 3, 4, 5]
    assert {"train_loss", "train_accuracy", "val_loss", "val_accuracy"} <= set(run.history.columns)
    assert not run.reached_target and not run.diverged
    assert run.train_seconds > 0


def test_scaler_is_fitted_only_on_training_data(toy_data):
    X, y = toy_data
    run = train_mlp(FAST, X[:800], y[:800], seed=0, eval_sets={"val": (X[800:], y[800:])})
    assert run.model.named_steps["scaler"].mean_ == pytest.approx(X[:800].mean().to_numpy())


def test_stops_as_soon_as_target_loss_is_reached(toy_data):
    X, y = toy_data
    config = MLPConfig(hidden_layers=(8,), target_loss=100.0, max_epochs=50)
    run = train_mlp(config, X, y, seed=0)
    assert run.reached_target
    assert run.epochs == 1


def test_same_seed_reproduces_history_and_other_seed_changes_it(toy_data):
    X, y = toy_data
    first = train_mlp(FAST, X, y, seed=7).history
    pd.testing.assert_frame_equal(first, train_mlp(FAST, X, y, seed=7).history)
    assert not np.allclose(first["train_loss"], train_mlp(FAST, X, y, seed=8).history["train_loss"])


def test_optimizer_is_created_once_so_momentum_survives_between_epochs(toy_data, monkeypatch):
    X, y = toy_data
    created = []
    original = sk_mlp.SGDOptimizer

    def counting_optimizer(*args, **kwargs):
        created.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(sk_mlp, "SGDOptimizer", counting_optimizer)
    train_mlp(FAST, X, y, seed=0)
    assert len(created) == 1


def test_samples_are_reshuffled_every_epoch(toy_data, monkeypatch):
    X, y = toy_data
    orders = []
    original = sk_mlp.shuffle

    def recording_shuffle(indices, random_state=None):
        result = original(indices, random_state=random_state)
        orders.append(np.array(result))
        return result

    monkeypatch.setattr(sk_mlp, "shuffle", recording_shuffle)
    train_mlp(FAST, X, y, seed=0)
    assert len(orders) == FAST.max_epochs
    assert not np.array_equal(orders[0], orders[1])


def test_classical_momentum_is_used(toy_data):
    X, y = toy_data
    classifier = train_mlp(FAST, X, y, seed=0).model.named_steps["mlp"]
    assert classifier.nesterovs_momentum is False
    assert classifier.momentum == FAST.momentum


def test_divergence_is_reported_without_raising(toy_data):
    X, y = toy_data
    config = MLPConfig(hidden_layers=(32, 32), learning_rate=1e8, momentum=0.9, max_epochs=20)
    run = train_mlp(config, X, y, seed=0)
    assert run.diverged
    assert not run.reached_target
    assert run.epochs < config.max_epochs


def test_epochs_to_loss_returns_first_hit_or_nan(toy_data):
    X, y = toy_data
    run = train_mlp(MLPConfig(hidden_layers=(16,), learning_rate=0.1, max_epochs=20), X, y, seed=0)
    assert run.epochs_to_loss(100.0) == 1
    assert np.isnan(run.epochs_to_loss(0.0))


def test_evaluate_returns_macro_metrics_for_the_original_labels(toy_data):
    X, y = toy_data
    config = MLPConfig(hidden_layers=(16,), learning_rate=0.1, max_epochs=30)
    run = train_mlp(config, X[:800], y[:800], seed=0)
    metrics = evaluate(run.model, X[800:], y[800:])
    assert set(metrics) == {"loss", "accuracy", "f1", "precision", "recall"}
    assert 0.5 < metrics["accuracy"] <= 1.0
    assert set(run.model.predict(X[800:])) <= {"a", "b", "c"}
