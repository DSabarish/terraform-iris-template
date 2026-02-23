"""
Configuration loader for Iris ML project.

Loads cfg/base.yaml and overrides with environment variables where applicable.
Usage:
    from cfg import get_config
    c = get_config()
    c["gcp"]["project_id"]
    c["gcs"]["paths"]["raw"]
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Directory containing base.yaml
_CFG_DIR = Path(__file__).resolve().parent
_BASE_PATH = _CFG_DIR / "base.yaml"

_base_config: Dict[str, Any] | None = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _env_overrides() -> Dict[str, Any]:
    """Build config overrides from current environment."""
    overrides = {}
    if os.environ.get("GCP_PROJECT"):
        overrides.setdefault("gcp", {})["project_id"] = os.environ["GCP_PROJECT"]
    if os.environ.get("GOOGLE_CLOUD_PROJECT") and not overrides.get("gcp", {}).get("project_id"):
        overrides.setdefault("gcp", {})["project_id"] = os.environ["GOOGLE_CLOUD_PROJECT"]
    if os.environ.get("VERTEX_REGION"):
        overrides.setdefault("gcp", {})["region"] = os.environ["VERTEX_REGION"]
    if os.environ.get("REGION") and "region" not in overrides.get("gcp", {}):
        overrides.setdefault("gcp", {})["region"] = os.environ["REGION"]
    if os.environ.get("GCS_BUCKET"):
        b = os.environ["GCS_BUCKET"].strip().replace("gs://", "").split("/")[0]
        overrides.setdefault("gcs", {})["bucket"] = b
    if os.environ.get("MODEL_PREFIX"):
        overrides.setdefault("gcs", {}).setdefault("paths", {})["model_latest"] = os.environ["MODEL_PREFIX"].rstrip("/")
    if os.environ.get("SCALER_PREFIX"):
        overrides.setdefault("gcs", {}).setdefault("paths", {})["scaler"] = os.environ["SCALER_PREFIX"].rstrip("/")
    if os.environ.get("GIT_COMMIT_SHA"):
        overrides.setdefault("pipeline", {})["version_key"] = os.environ["GIT_COMMIT_SHA"]
    if os.environ.get("PORT"):
        try:
            overrides.setdefault("app", {})["port"] = int(os.environ["PORT"])
        except ValueError:
            pass
    if os.environ.get("ENVIRONMENT"):
        overrides.setdefault("app", {})["environment"] = os.environ["ENVIRONMENT"]
    return overrides


def get_config(reload: bool = False) -> Dict[str, Any]:
    """
    Return the application config: base.yaml with current env overrides.
    Base YAML is cached; env is re-applied on each call so tests/CI can set env and see it.
    """
    global _base_config
    if _base_config is None or reload:
        if not _BASE_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {_BASE_PATH}")
        _base_config = _load_yaml(_BASE_PATH)
    overrides = _env_overrides()
    if not overrides:
        return _base_config
    return _deep_merge(_base_config, overrides)


def get_gcs_path(key: str) -> str:
    """Return a single GCS path prefix from config (e.g. 'raw', 'clean', 'model')."""
    c = get_config()
    return c["gcs"]["paths"].get(key, key)


def get_model_setting(key: str) -> Any:
    """Return a value from config['model'] (e.g. 'feature_cols', 'target_col')."""
    c = get_config()
    return c["model"].get(key)
