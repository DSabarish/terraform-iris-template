# tests/test_pipeline.py
import io
import pickle
import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clean_raw_data import clean_data
from transform_data import transform, FEATURE_COLS


# ─────────────────────────────────────────────────
# clean_raw_data tests
# ─────────────────────────────────────────────────
class TestCleanData:
    def _make_df(self):
        return pd.DataFrame({
            "sepal_length": [5.1, 4.9, None, 5.1],
            "sepal_width":  [3.5, 3.0, 2.8, 3.5],
            "petal_length": [1.4, 1.4, 4.7, 1.4],
            "petal_width":  [0.2, 0.2, 1.4, 0.2],
            "target":       [0, 1, 2, 0],          # row 3 is duplicate of row 0
            "target_name":  ["setosa", "versicolor", "virginica", "setosa"],
        })

    def test_removes_duplicates(self):
        df = self._make_df()
        clean = clean_data(df)
        # Rows 0 and 3 are identical; after dedup + dropna we should have 2
        assert len(clean) == 2

    def test_removes_nulls(self):
        df = self._make_df()
        clean = clean_data(df)
        assert clean.isnull().sum().sum() == 0

    def test_casts_types(self):
        df = self._make_df()
        clean = clean_data(df)
        for col in FEATURE_COLS:
            assert clean[col].dtype == float

    def test_invalid_targets_removed(self):
        df = pd.DataFrame({
            "sepal_length": [5.1, 6.0],
            "sepal_width":  [3.5, 2.9],
            "petal_length": [1.4, 4.5],
            "petal_width":  [0.2, 1.5],
            "target":       [0, 99],   # 99 is invalid
            "target_name":  ["setosa", "unknown"],
        })
        clean = clean_data(df)
        assert all(clean["target"].isin([0, 1, 2]))


# ─────────────────────────────────────────────────
# transform_data tests
# ─────────────────────────────────────────────────
class TestTransform:
    def _make_clean_df(self):
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
        return df

    def test_scaled_mean_near_zero(self):
        df = self._make_clean_df()
        df_t, scaler = transform(df)
        for col in FEATURE_COLS:
            assert abs(df_t[col].mean()) < 0.1, f"{col} mean not near zero"

    def test_engineered_features_exist(self):
        df = self._make_clean_df()
        df_t, _ = transform(df)
        assert "petal_area" in df_t.columns
        assert "sepal_area" in df_t.columns

    def test_scaler_is_picklable(self):
        df = self._make_clean_df()
        _, scaler = transform(df)
        data = pickle.dumps(scaler)
        loaded = pickle.loads(data)
        assert hasattr(loaded, "transform")


# ─────────────────────────────────────────────────
# inference tests (mocked GCS)
# ─────────────────────────────────────────────────
class TestInference:
    def test_predict_returns_valid_class(self):
        """Train a quick model inline and verify predict() contract."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import load_iris

        iris = load_iris()
        scaler = StandardScaler().fit(iris.data)
        X_scaled = scaler.transform(iris.data)
        petal_area = X_scaled[:, 2] * X_scaled[:, 3]
        sepal_area = X_scaled[:, 0] * X_scaled[:, 1]
        X_full = np.column_stack([X_scaled, petal_area, sepal_area])
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_full, iris.target)

        os.environ["GCS_BUCKET"] = "fake-bucket"
        os.environ["MODEL_PREFIX"] = "artifacts/models/latest"
        os.environ["SCALER_PREFIX"] = "artifacts/scalers"

        import inference
        inference.load_model.cache_clear()
        inference.load_scaler.cache_clear()

        with patch.object(inference, "_download_pkl") as mock_dl:
            def side_effect(bucket, path):
                if "model" in path:
                    return model
                return scaler
            mock_dl.side_effect = side_effect

            result = inference.predict([5.1, 3.5, 1.4, 0.2])
            assert "class_id" in result
            assert "class_name" in result
            assert result["class_name"] in ["setosa", "versicolor", "virginica"]
            assert abs(sum(result["probabilities"].values()) - 1.0) < 0.001
