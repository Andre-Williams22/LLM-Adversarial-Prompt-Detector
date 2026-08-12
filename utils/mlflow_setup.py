"""MLflow tracking configuration.

Both the detector and the web layer log to the same experiment, so the setup
lives in one place. Tracking is best-effort: if the tracking server is
unreachable the service must still serve traffic, because an observability
outage is not a safety outage.
"""
import os
from pathlib import Path

import mlflow

from utils.logging_config import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "adversarial_detection_system"

_configured = False


def resolve_tracking_uri() -> str:
    """Remote tracking server if configured, otherwise a local file store."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    project_root = Path(__file__).resolve().parent.parent
    return f"file://{project_root / 'mlruns'}"


def configure_mlflow(experiment_name: str = EXPERIMENT_NAME) -> bool:
    """Point MLflow at the tracking store and select the shared experiment.

    Returns True when tracking is usable, False when it is not. Callers should
    treat False as "carry on without tracking", never as a fatal error.
    """
    global _configured
    if _configured:
        return True

    try:
        tracking_uri = resolve_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow tracking enabled: %s (experiment=%s)",
                    tracking_uri, experiment_name)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info("Experiment %s will be created on the first run",
                        experiment_name)
        else:
            logger.debug("Experiment %s resolved to id=%s",
                         experiment_name, experiment.experiment_id)

        _configured = True
        return True
    except Exception as exc:
        logger.warning("MLflow setup failed (%s); continuing without tracking", exc)
        return False
