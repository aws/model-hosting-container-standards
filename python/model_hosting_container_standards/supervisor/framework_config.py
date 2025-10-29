"""
Framework-specific configuration and command mapping for supervisor.

This module provides framework detection and default command mapping
for different ML frameworks supported by the supervisor system.
"""

import os
from typing import Optional

from ..logging_config import get_logger
from .config import FrameworkName

logger = get_logger(__name__)


# Supported framework names for validation
SUPPORTED_FRAMEWORKS = {framework.value for framework in FrameworkName}


def get_framework_command() -> Optional[str]:
    """Get the framework command from environment variables.

    Returns:
        Optional[str]: Framework command to execute, or None if not available
    """
    # Check for explicit framework command
    framework_command = os.getenv("FRAMEWORK_COMMAND")
    if framework_command:
        command = framework_command.strip()
        if command:
            return command
        else:
            logger.warning("FRAMEWORK_COMMAND environment variable is set but empty")

    # If no explicit command, log error and return None
    logger.error(
        "No framework command available. Set FRAMEWORK_COMMAND environment variable with your framework's start command."
    )
    return None


def validate_framework_command(command: str) -> bool:
    """Validate that a framework command appears to be executable.

    Args:
        command: The framework command to validate

    Returns:
        bool: True if command appears valid, False otherwise
    """
    if not command or not command.strip():
        return False

    # Basic validation - command should start with an executable
    parts = command.strip().split()
    if not parts:
        return False

    executable = parts[0]

    # Check for common executable patterns
    if executable in ("python", "python3", "java", "node", "bash", "sh"):
        return True

    # Check if it's a path to an executable
    if executable.startswith("/") or executable.startswith("./"):
        return True

    # Check if it's a module execution pattern
    if "python" in executable or "-m" in command:
        return True

    # Allow other patterns but warn
    logger.warning(f"Framework command executable '{executable}' may not be valid")
    return True


def get_supported_frameworks() -> set[str]:
    """Get a set of supported framework names for validation.

    Returns:
        set[str]: Set of supported framework names
    """
    return SUPPORTED_FRAMEWORKS
