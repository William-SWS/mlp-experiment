import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    """Base sintética desbalanceada; os testes não dependem do download do UCI."""
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_classes=3,
        weights=[0.6, 0.3, 0.1],
        random_state=0,
    )
    columns = [f"f{i}" for i in range(X.shape[1])]
    labels = pd.Series(y).map({0: "a", 1: "b", 2: "c"}).rename("Class")
    return pd.DataFrame(X, columns=columns), labels
