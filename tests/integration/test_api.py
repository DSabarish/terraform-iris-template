# tests/integration/test_api.py
"""
Integration tests run against a live Cloud Run endpoint.
Usage: pytest tests/integration/ --base-url=https://iris-ml-api-dev-xxx.run.app
"""

import pytest
import requests


def pytest_addoption(parser):
    parser.addoption("--base-url", action="store", required=True, help="Cloud Run service base URL")


@pytest.fixture(scope="session")
def base_url(request):
    return request.config.getoption("--base-url").rstrip("/")


class TestHealthEndpoint:
    def test_health_returns_ok(self, base_url):
        r = requests.get(f"{base_url}/health", timeout=30)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestPredictEndpoint:
    SETOSA_SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    VIRGINICA_SAMPLE = {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}

    def test_predict_setosa(self, base_url):
        r = requests.post(f"{base_url}/predict", json=self.SETOSA_SAMPLE, timeout=60)
        assert r.status_code == 200
        result = r.json()
        assert result["class_name"] == "setosa"
        assert result["probabilities"]["setosa"] > 0.7

    def test_predict_virginica(self, base_url):
        r = requests.post(f"{base_url}/predict", json=self.VIRGINICA_SAMPLE, timeout=60)
        assert r.status_code == 200
        result = r.json()
        assert result["class_name"] == "virginica"

    def test_predict_probabilities_sum_to_one(self, base_url):
        r = requests.post(f"{base_url}/predict", json=self.SETOSA_SAMPLE, timeout=60)
        probs = r.json()["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_invalid_input_rejected(self, base_url):
        r = requests.post(f"{base_url}/predict", json={"sepal_length": -999}, timeout=30)
        assert r.status_code == 422

    def test_batch_predict(self, base_url):
        payload = {"instances": [self.SETOSA_SAMPLE, self.VIRGINICA_SAMPLE]}
        r = requests.post(f"{base_url}/predict/batch", json=payload, timeout=60)
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) == 2
        assert preds[0]["class_name"] == "setosa"
        assert preds[1]["class_name"] == "virginica"


class TestModelInfo:
    def test_model_info_returns_env(self, base_url):
        r = requests.get(f"{base_url}/model-info", timeout=30)
        assert r.status_code == 200
        info = r.json()
        assert "environment" in info
        assert "model_bucket" in info
