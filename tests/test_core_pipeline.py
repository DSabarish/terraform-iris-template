import pandas as pd
from sklearn.datasets import load_iris

from src import clean_raw_data, transform_data, train


def test_clean_data_removes_duplicates_and_invalid_targets():
    data = {
        "sepal_length": [5.1, 5.1, 6.2],
        "sepal_width": [3.5, 3.5, 3.4],
        "petal_length": [1.4, 1.4, 5.4],
        "petal_width": [0.2, 0.2, 2.3],
        "target": [0, 0, 99],  # 99 should be dropped
    }
    df = pd.DataFrame(data)
    # Add an explicit duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = clean_raw_data.clean_data(df)

    # Only the valid class 0 rows should remain, with duplicates removed
    assert set(cleaned["target"].unique()) == {0}
    assert len(cleaned) == 1


def test_transform_adds_engineered_features_and_scales():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]

    transformed, scaler = transform_data.transform(df)

    for col in ["petal_area", "sepal_area"]:
        assert col in transformed.columns

    means = transformed[transform_data.FEATURE_COLS].mean().abs()
    assert (means < 1e-6).all()
    assert scaler is not None


def test_train_and_evaluate_produces_metrics():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]

    transformed, _ = transform_data.transform(df)

    model, metrics = train.train_and_evaluate(transformed)

    assert model is not None
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

