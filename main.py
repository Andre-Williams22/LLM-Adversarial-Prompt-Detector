"""Application entry point.

Assembles the service from the pieces in utils/ and owns nothing else:

    utils/detector_registry.py  model lifecycle
    utils/api_routes.py         HTTP surface
    utils/chat_handler.py       request-path detection logic
    utils/gradio_ui.py          demo interface
    utils/prometheus_metrics.py instrumentation
    utils/mongodb_manager.py    interaction logging

Run locally with:  uvicorn main:app --host 0.0.0.0 --port 8080
"""
import os
from contextlib import asynccontextmanager

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI

from utils.api_routes import detection, operations
from utils.detector_registry import registry
from utils.gradio_ui import build_interface
from utils.logging_config import configure_logging, get_logger
from utils.mlflow_setup import configure_mlflow
from utils.mongodb_manager import mongodb_manager
from utils.prometheus_metrics import cleanup, initialize_metrics

load_dotenv()
configure_logging()

logger = get_logger(__name__)

# Loading the models before the server binds gives the first user an instant
# response but risks tripping a platform startup probe on a slow cold start.
# Background loading binds first and gates traffic on /ready instead.
EAGER_MODEL_LOAD = os.getenv("EAGER_MODEL_LOAD", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start dependencies on boot and release them on shutdown."""
    logger.info("Starting Adversarial Prompt Detector")

    configure_mlflow()
    initialize_metrics()

    if EAGER_MODEL_LOAD:
        registry.load()
    else:
        registry.start_background_load()

    try:
        await mongodb_manager.connect()
    except Exception as exc:
        logger.warning("MongoDB unavailable, continuing without logging: %s", exc)

    yield

    logger.info("Shutting down")
    await mongodb_manager.close()
    cleanup()


def create_app() -> FastAPI:
    """Build the FastAPI application with the Gradio demo mounted at /chat."""
    app = FastAPI(
        title="Adversarial Prompt Detector",
        description=(
            "Input-side safeguard classifier that screens prompts for "
            "jailbreaks and prompt injection before they reach an assistant."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(operations)
    app.include_router(detection)

    return gr.mount_gradio_app(app, build_interface(), path="/chat")


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    logger.info("Serving on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")
