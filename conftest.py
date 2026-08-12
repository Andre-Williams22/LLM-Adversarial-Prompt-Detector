"""Pytest bootstrap.

The detector module instantiates a global ``FastAdversarialDetector`` at import
time so that the production server has its models resident before the first
request. That is the right behaviour in production and the wrong behaviour in a
test run, so we neutralise the two side effects -- Hugging Face pipeline
construction and MLflow tracking -- before the module is ever imported.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlflow  # noqa: E402
import transformers  # noqa: E402


def _stub_pipeline(*_args, **_kwargs):
    """Stand in for a transformers pipeline, returning a neutral (safe) score."""
    return MagicMock(return_value=[{"label": "LABEL_0", "score": 0.0}])


transformers.pipeline = _stub_pipeline

for _name in ("set_tracking_uri", "set_experiment", "log_param", "log_metric",
              "log_text", "end_run"):
    setattr(mlflow, _name, MagicMock())
mlflow.start_run = MagicMock()
mlflow.get_experiment_by_name = MagicMock(return_value=None)
