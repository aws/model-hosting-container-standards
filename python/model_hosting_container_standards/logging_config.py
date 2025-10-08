"""Logging configuration for model hosting container standards."""

import logging
import os
import sys


def get_logger(name: str = "model_hosting_container_standards") -> logging.Logger:
    """Get a configured logger for the package."""
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Get log level from environment or default to INFO
        level = os.getenv(
            "SAGEMAKER_CONTAINER_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")
        )

        # Set up handler with consistent format
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s - %(filename)s:%(lineno)d: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))

    return logger


# Package logger instance
logger = get_logger()
