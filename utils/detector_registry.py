"""Lifecycle management for the single shared detector instance.

The detector holds roughly 540MB of transformer weights and takes ~10s to
build, so exactly one instance is shared across all requests. Loading it at
import time (the previous design) blocked the process before the HTTP server
could bind, which made platform startup probes fail on slow cold starts. Here
the load runs on a background thread: the server binds and answers /health
immediately, and /ready reports false until the weights are resident.
"""
import os
import threading
import time
from typing import Optional

from utils.fast_detection import FastAdversarialDetector
from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_SENSITIVITY = os.getenv("FAST_DETECTION_SENSITIVITY", "balanced")


class DetectorRegistry:
    """Owns the detector singleton and reports its loading state."""

    def __init__(self, sensitivity_mode: str = DEFAULT_SENSITIVITY):
        self.sensitivity_mode = sensitivity_mode
        self._detector: Optional[FastAdversarialDetector] = None
        self._error: Optional[str] = None
        self._loading = False
        self._load_seconds: Optional[float] = None
        self._lock = threading.Lock()
        self._loaded = threading.Event()

    @property
    def is_ready(self) -> bool:
        return self._detector is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def error(self) -> Optional[str]:
        return self._error

    def status(self) -> dict:
        """Serialisable state for the health and readiness endpoints."""
        return {
            "models_loaded": self.is_ready,
            "models_loading": self.is_loading,
            "model_count": len(self._detector.models) if self._detector else 0,
            "sensitivity_mode": self.sensitivity_mode,
            "load_seconds": self._load_seconds,
            "error": self._error,
        }

    def load(self) -> Optional[FastAdversarialDetector]:
        """Build the detector synchronously. Idempotent."""
        with self._lock:
            if self._detector is not None:
                return self._detector
            self._loading = True
            self._error = None

        started = time.time()
        try:
            logger.info("Loading detection models (sensitivity=%s)",
                        self.sensitivity_mode)
            detector = FastAdversarialDetector(sensitivity_mode=self.sensitivity_mode)
            self._load_seconds = round(time.time() - started, 2)
            logger.info("Detection models ready in %.2fs (%d models)",
                        self._load_seconds, len(detector.models))
            with self._lock:
                self._detector = detector
                self._loading = False
            self._loaded.set()
            return detector
        except Exception as exc:
            logger.exception("Detection model loading failed")
            with self._lock:
                self._error = str(exc)
                self._loading = False
            self._loaded.set()
            return None

    def start_background_load(self) -> None:
        """Kick off loading without blocking the caller (e.g. app startup)."""
        if self.is_ready or self.is_loading:
            return
        threading.Thread(
            target=self.load,
            name="detector-loader",
            daemon=True,
        ).start()

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the load finishes. Returns True if a detector exists."""
        self._loaded.wait(timeout=timeout)
        return self.is_ready

    def get(self) -> FastAdversarialDetector:
        """Return the detector, loading it on demand.

        Raises RuntimeError if the models could not be loaded, so callers get a
        clear failure rather than an AttributeError on None.
        """
        if self._detector is None:
            self.load()
        if self._detector is None:
            raise RuntimeError(f"Detection models unavailable: {self._error}")
        return self._detector

    def set_sensitivity(self, sensitivity_mode: str) -> FastAdversarialDetector:
        """Rebuild the detector under a different policy preset.

        Model weights are reloaded, so this is a control-plane operation and
        not something to call per request.
        """
        if sensitivity_mode == self.sensitivity_mode and self._detector is not None:
            return self._detector
        logger.info("Switching sensitivity mode: %s -> %s",
                    self.sensitivity_mode, sensitivity_mode)
        with self._lock:
            self._detector = None
        self.sensitivity_mode = sensitivity_mode
        self._loaded.clear()
        return self.get()


registry = DetectorRegistry()
