"""HTTP API surface.

Split into two routers so the operational endpoints (health, readiness,
metrics) stay separate from the ones that do classification work.

Liveness vs readiness: /health answers as soon as the process is up, so the
platform does not kill a container that is still loading weights. /ready only
answers 200 once the models are resident, so no traffic is routed to a replica
that cannot classify yet.
"""
import time

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from utils.chat_handler import classify
from utils.detector_registry import registry
from utils.logging_config import get_logger

logger = get_logger(__name__)

API_VERSION = "1.0.0"

# 1x1 transparent PNG, served so browsers stop generating 404s in the logs.
FAVICON = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08"
    b"\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00"
    b"\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
)

operations = APIRouter(tags=["operations"])
detection = APIRouter(tags=["detection"])


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000,
                      description="Prompt to screen")
    sensitivity_mode: str = Field(
        None, description="Override the policy preset for this process"
    )


class DetectResponse(BaseModel):
    is_adversarial: bool
    reason: str
    scores: list
    threshold: float
    inference_time: float = None


@operations.get("/")
def root() -> dict:
    """Service description and endpoint index."""
    return {
        "service": "Adversarial Prompt Detector",
        "description": (
            "Input-side safeguard classifier that screens prompts for "
            "jailbreaks and prompt injection before they reach an assistant."
        ),
        "version": API_VERSION,
        "status": "running",
        "detector": registry.status(),
        "endpoints": {
            "chat_interface": "/chat",
            "detect": "/detect",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "statistics": "/stats",
            "docs": "/docs",
        },
    }


@operations.get("/health")
def health() -> dict:
    """Liveness: the process is up and serving. Always 200 while running."""
    return {
        "status": "healthy" if registry.is_ready else "starting",
        "version": API_VERSION,
        "timestamp": time.time(),
        **registry.status(),
    }


@operations.get("/ready")
def ready():
    """Readiness: 200 only once the models can actually classify."""
    if registry.is_ready:
        return {"status": "ready", **registry.status()}

    payload = {
        "status": "error" if registry.error else "loading",
        **registry.status(),
    }
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content=payload)


@operations.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape target."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@operations.get("/stats")
async def stats() -> dict:
    """Aggregated interaction statistics from the application database."""
    from utils.mongodb_manager import mongodb_manager

    try:
        return {
            "chat_statistics": await mongodb_manager.get_chat_statistics(days=7),
            "model_performance": await mongodb_manager.get_model_performance_stats(days=7),
            "time_period": "last_7_days",
        }
    except Exception as exc:
        logger.error("Failed to retrieve statistics: %s", exc)
        return {
            "error": f"Failed to retrieve statistics: {exc}",
            "chat_statistics": {},
            "model_performance": [],
        }


@operations.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(content=FAVICON, media_type="image/png")


@operations.get("/manifest.json", include_in_schema=False)
def manifest() -> dict:
    return {
        "name": "Adversarial Prompt Detector",
        "short_name": "AdversarialDetector",
        "description": "Prompt screening with adversarial detection and safety filtering",
        "start_url": "/chat",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#007BFF",
        "icons": [{"src": "/favicon.ico", "sizes": "16x16", "type": "image/x-icon"}],
    }


@detection.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest):
    """Classify a single prompt and return the ensemble's reasoning."""
    if not registry.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "loading", **registry.status()},
        )

    if request.sensitivity_mode:
        registry.set_sensitivity(request.sensitivity_mode)

    is_adversarial, reasoning = classify(request.text)
    return DetectResponse(
        is_adversarial=is_adversarial,
        reason=reasoning.get("reason", ""),
        scores=reasoning.get("scores", []),
        threshold=reasoning.get("threshold", 0.5),
        inference_time=reasoning.get("inference_time"),
    )


@detection.post("/warm-up", include_in_schema=False)
def warm_up() -> dict:
    """Force model loading. Useful right after a cold deploy."""
    registry.start_background_load()
    return {"status": "warming" if not registry.is_ready else "ready",
            **registry.status()}
