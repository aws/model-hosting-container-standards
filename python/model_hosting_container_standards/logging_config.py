"""Logging configuration for model hosting container standards."""

import logging
import os
import sys
from typing import Union


def parse_level(level: str) -> Union[int, str]:
    """Parse a log level string into a valid logging level.

    Args:
        level: Log level string to parse.

    Returns:
        Valid logging level.
    """
    # Convert level to uppercase
    level = level.upper()

    # Convert numeric log level string to int so `setLevel` can recognize it
    try:
        return int(level)
    except ValueError:
        return level


def _get_env_log_level() -> Union[int, str]:
    """Get the log level from environment variables, defaulting to ERROR."""
    level = os.getenv(
        "SAGEMAKER_CONTAINER_LOG_LEVEL", os.getenv("LOG_LEVEL", "ERROR")
    )
    return parse_level(level)


def _safe_set_level(logger: logging.Logger, level: Union[int, str]) -> None:
    """Set log level on a logger, falling back to ERROR on failure."""
    try:
        logger.setLevel(level)
    except (ValueError, AttributeError, TypeError):
        logger.setLevel(logging.ERROR)


def configure_root_logger() -> None:
    """Enforce SAGEMAKER_CONTAINER_LOG_LEVEL on the root logger.

    Ensures the environment variable is respected even when another
    framework has already initialized the root logger.
    """
    level = _get_env_log_level()
    root = logging.getLogger()
    _safe_set_level(root, level)

    resolved = root.level
    for handler in root.handlers:
        if handler.level < resolved:
            handler.setLevel(resolved)


def get_logger(name: str = "model_hosting_container_standards") -> logging.Logger:
    """Get a configured logger for the package.

    The logger uses SAGEMAKER_CONTAINER_LOG_LEVEL (or LOG_LEVEL) to determine the log level.
    If not set, defaults to ERROR level, which effectively disables most package logging.

    Returns:
        Configured logger instance for the package.
    """
    logger = logging.getLogger(name)

    level = _get_env_log_level()

    # Always apply the log level, even if handlers already exist
    _safe_set_level(logger, level)

    # Only add our handler once to avoid duplicate log lines
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s - %(filename)s:%(lineno)d: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


configure_root_logger()

# Package logger instance
logger = get_logger()
