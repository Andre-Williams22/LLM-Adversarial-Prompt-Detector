"""Central logging configuration.

Everything logs to stdout so that whichever platform is hosting the service
(Cloud Run, CapRover, a local shell) is the one responsible for collection and
retention. File logging is opt-in via LOG_FILE for local debugging only -- a
container that writes logs to its own filesystem loses them on every restart.
"""
import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_configured = False


def configure_logging(level: str = None, log_file: str = None) -> None:
    """Install the root logging handlers. Safe to call more than once."""
    global _configured
    if _configured:
        return

    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_file = log_file or os.getenv("LOG_FILE")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=DEFAULT_FORMAT,
        handlers=handlers,
        force=True,
    )

    # These libraries log every HTTP call at INFO, which drowns out our own
    # detection decisions in production log search.
    for noisy in ("httpx", "urllib3", "botocore", "asyncio", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root logger on first use."""
    configure_logging()
    return logging.getLogger(name)
