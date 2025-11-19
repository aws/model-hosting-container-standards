"""Logging configuration for model hosting container standards."""

import logging
import os
import sys


def get_logger(name: str = "model_hosting_container_standards") -> logging.Logger:
    """Get a configured logger for the package.

    The logger uses SAGEMAKER_CONTAINER_LOG_LEVEL (or LOG_LEVEL) to determine the log level.
    If not set, defaults to ERROR level, which effectively disables most package logging.

    Returns:
        Configured logger instance for the package.
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Get log level from environment, default to ERROR (effectively disabled)
        # Convert level to uppercase
        level = os.getenv(
            "SAGEMAKER_CONTAINER_LOG_LEVEL", os.getenv("LOG_LEVEL", "ERROR")
        ).upper()

        # Set up handler with consistent format
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s - %(filename)s:%(lineno)d: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Convert numeric log level string to int so `setLevel` can recognize it
        # Will raise error if the set log level is not registered in
        # logging.getLevelNamesMapping()
        logger.setLevel(int(level) if level.isdigit() else level.upper())

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger


# Package logger instance
logger = get_logger()
