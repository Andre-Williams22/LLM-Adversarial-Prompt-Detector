"""Request-path logic for the chat interface.

The flow is: classify the prompt, block it or answer it, emit metrics, and
record the interaction. Keeping this out of main.py means the decision path can
be exercised without standing up FastAPI or Gradio.
"""
import asyncio
import os
import threading
import time
import uuid
from typing import Dict, List, Tuple

from utils.detector_registry import registry
from utils.logging_config import get_logger
from utils.mongodb_manager import mongodb_manager
from utils.prometheus_metrics import (
    track_chat_failure,
    track_chat_request,
    track_concurrent_request_end,
    track_concurrent_request_start,
    track_model_error,
    track_model_inference,
    track_prompt_result,
)

logger = get_logger(__name__)

# Order matters: it lines up with the score vector the detector returns.
MODEL_NAMES = ["keyword_detector", "toxic_bert", "hate_classifier", "safety_classifier"]

BLOCKED_MESSAGE = "Adversarial prompt detected. The request was not processed."
ERROR_MESSAGE = "An error occurred while processing your request."
LOADING_MESSAGE = (
    "The detection models are still loading. Please retry in a few seconds."
)

# Last-resort patterns used only when the ensemble itself raises. Failing
# closed on these keeps the obvious attacks blocked during an outage.
FALLBACK_KEYWORDS = [
    "ignore", "jailbreak", "bypass", "override", "previous instructions",
    "act as", "pretend", "roleplay", "system prompt", "admin", "root",
    "prompt injection", "escape", "break out", "sudo", "developer mode",
]

DEFAULT_REASONING = {
    "reason": "Unknown",
    "scores": [0.0, 0.0, 0.0, 0.0],
    "threshold": 0.5,
}


def _fallback_detection(text: str, error: Exception) -> Tuple[bool, Dict]:
    """Degrade to keyword matching when the ensemble is unavailable."""
    lowered = text.lower()
    matched = any(keyword in lowered for keyword in FALLBACK_KEYWORDS)
    reason = "Basic keyword match" if matched else "No obvious patterns"
    return matched, {
        "reason": f"Fallback detection - {reason}: {error}",
        "scores": [0.8 if matched else 0.1, 0.0, 0.0, 0.0],
        "threshold": 0.5,
    }


def classify(text: str) -> Tuple[bool, Dict]:
    """Classify a prompt, falling back to keywords if the ensemble fails."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        detector = registry.get()
        is_adversarial, reasoning = detector.detect_adversarial_sync(text)
        track_prompt_result(is_adversarial, model_name="fast_ensemble")
        return is_adversarial, reasoning
    except Exception as exc:
        logger.error("Detection failed, using fallback: %s", exc)
        track_model_error("fast_detector", "detection_failure")
        is_adversarial, reasoning = _fallback_detection(text, exc)
        track_prompt_result(is_adversarial, model_name="fallback_detector")
        return is_adversarial, reasoning


def _record_model_metrics(reasoning: Dict, duration: float) -> None:
    """Emit one Prometheus sample per ensemble member."""
    threshold = reasoning.get("threshold", 0.5)
    for name, score in zip(MODEL_NAMES, reasoning.get("scores", [])):
        track_model_inference(
            model_name=name,
            model_type="fast_detector",
            duration=duration,
            is_adversarial=score > threshold,
        )


def _log_interaction_async(**kwargs) -> None:
    """Persist the interaction to MongoDB without blocking the response.

    Gradio invokes the handler from a worker thread with no running event loop,
    so the coroutine gets its own loop on a throwaway daemon thread. Logging is
    fire-and-forget: a database outage must never fail a request.
    """
    def _run() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(mongodb_manager.log_chat_interaction(**kwargs))
            logger.debug("Interaction logged to MongoDB")
        except Exception as exc:
            logger.error("MongoDB logging failed: %s", exc)
        finally:
            loop.close()

    threading.Thread(target=_run, name="mongodb-logger", daemon=True).start()


def _generate_reply(user_message: str, history: List[Tuple[str, str]]) -> str:
    """Produce the assistant reply for a prompt that passed the classifier.

    The upstream LLM call is stubbed. The point of this project is the
    safeguard in front of the model, so the generator is deliberately a
    placeholder rather than a billed API dependency.
    """
    return (
        f"This is a mock response to your message: '{user_message}'. "
        "The adversarial detection system determined this prompt is safe."
    )


def _format_verdict(is_adversarial: bool, reasoning: Dict) -> str:
    colour = "red" if is_adversarial else "green"
    headline = (
        "Adversarial prompt detected"
        if is_adversarial
        else "No adversarial prompt detected"
    )
    return (
        f"<p style='color:{colour};font-weight:bold;'>"
        f"{headline}<br>"
        f"Reason: {reasoning.get('reason', 'Unknown')}<br>"
        f"Threshold: {reasoning.get('threshold', 0.0):.2f}<br>"
        f"Scores: {reasoning.get('scores', [])}</p>"
    )


def chat_and_detect(user_message: str, history):
    """Gradio handler: returns (chatbot, state, verdict_html)."""
    started = time.time()
    history = history or []
    history.append(("User", user_message))
    session_id = str(uuid.uuid4())

    track_concurrent_request_start()
    track_chat_request()

    bot_response = ERROR_MESSAGE
    verdict = "<p style='color:orange;'>Processing error</p>"
    is_adversarial = False
    reasoning = dict(DEFAULT_REASONING)

    try:
        if not registry.is_ready and registry.is_loading:
            history.append(("Bot", LOADING_MESSAGE))
            return history, history, "<p style='color:#666;'>Models loading</p>"

        logger.info("Classifying prompt (%d chars)", len(user_message))
        detection_started = time.time()
        is_adversarial, reasoning = classify(user_message)
        detection_duration = time.time() - detection_started

        _record_model_metrics(reasoning, detection_duration)
        logger.info(
            "Verdict=%s reason=%r detection=%.3fs",
            "ADVERSARIAL" if is_adversarial else "SAFE",
            reasoning.get("reason"),
            detection_duration,
        )

        if is_adversarial:
            bot_response = BLOCKED_MESSAGE
        else:
            bot_response = _generate_reply(user_message, history)
        history.append(("Bot", bot_response))
        verdict = _format_verdict(is_adversarial, reasoning)

    except Exception as exc:
        logger.exception("Chat request failed")
        track_chat_failure("general_error")
        bot_response = ERROR_MESSAGE
        history.append(("Bot", bot_response))
        verdict = f"<p style='color:orange;font-weight:bold;'>Error: {exc}</p>"
    finally:
        track_concurrent_request_end()

    latency = time.time() - started
    logger.info("Chat request completed in %.3fs", latency)

    _log_interaction_async(
        user_message=user_message,
        bot_response=bot_response,
        detection_results={
            "is_adversarial": is_adversarial,
            "scores": reasoning.get("scores", []),
            "reason": reasoning.get("reason", ""),
            "threshold": reasoning.get("threshold", 0.0),
        },
        latency=latency,
        session_id=session_id,
    )

    return history, history, verdict
