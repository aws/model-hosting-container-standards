"""
Framework-specific configuration and command mapping for supervisor.

This module provides framework detection and default command mapping
for different ML frameworks supported by the supervisor system.
"""

import os
from typing import Dict, Optional

from ..logging_config import get_logger
from .config import FrameworkName, get_framework_name

logger = get_logger(__name__)


# Default framework commands mapping
DEFAULT_FRAMEWORK_COMMANDS: Dict[FrameworkName, str] = {
    FrameworkName.VLLM: "python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080",
    FrameworkName.TENSORRT_LLM: "python /path/to/tensorrt_llm_server --host 0.0.0.0 --port 8080",
}


def get_framework_command() -> Optional[str]:
    """Get the framework command from environment or default.

    Returns:
        Optional[str]: Framework command to execute, or None if not available

    Raises:
        ConfigurationError: If no framework command can be determined
    """
    # Check for explicit framework command first
    framework_command = os.getenv("FRAMEWORK_COMMAND")
    if framework_command:
        command = framework_command.strip()
        if command:
            return command
        else:
            logger.warning("FRAMEWORK_COMMAND environment variable is set but empty")

    # Try to get default command for detected framework
    framework = get_framework_name()
    if framework:
        if framework in DEFAULT_FRAMEWORK_COMMANDS:
            return DEFAULT_FRAMEWORK_COMMANDS[framework]
        else:
            logger.error(
                f"Framework '{framework.value}' detected but no default command available"
            )
            return None

    # If no explicit command and no framework name, this is an error
    logger.error(
        "No framework command available. Either set FRAMEWORK_COMMAND or FRAMEWORK_NAME environment variable"
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


def get_supported_frameworks() -> Dict[str, str]:
    """Get a mapping of supported framework names to their default commands.

    Returns:
        Dict[str, str]: Mapping of framework names to default commands
    """
    return {
        framework.value: command
        for framework, command in DEFAULT_FRAMEWORK_COMMANDS.items()
    }
