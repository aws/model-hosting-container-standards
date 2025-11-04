"""Configuration management for supervisor process management."""

import os
from dataclasses import dataclass
from typing import Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""

    pass


@dataclass
class SupervisorConfig:
    """Configuration for supervisor process management system."""

    auto_recovery: bool = True
    max_start_retries: int = 3
    recovery_backoff_seconds: int = (
        10  # Currently unused - supervisord doesn't support backoff
    )
    launch_command: Optional[str] = None
    config_path: str = "/tmp/supervisord.conf"
    log_level: str = "info"


def _parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ("true", "1", "yes", "on")


def _get_env_int(name: str, default: int, min_val: int = 0, max_val: int = 100) -> int:
    """Get integer from environment with validation."""
    value = os.getenv(name)
    if not value:
        return default

    try:
        parsed = int(value)
        if not (min_val <= parsed <= max_val):
            raise ConfigurationError(
                f"{name} must be between {min_val} and {max_val}, got {parsed}"
            )
        return parsed
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer, got '{value}'")


def _get_env_str(name: str, default: str, allowed: Optional[list] = None) -> str:
    """Get string from environment with validation."""
    value = os.getenv(name, default).strip()
    if allowed and value.lower() not in allowed:
        raise ConfigurationError(f"{name} must be one of {allowed}, got '{value}'")
    return value


def parse_environment_variables() -> SupervisorConfig:
    """Parse environment variables and return SupervisorConfig instance."""
    try:
        return SupervisorConfig(
            auto_recovery=_parse_bool(os.getenv("ENGINE_AUTO_RECOVERY", "true")),
            max_start_retries=_get_env_int("ENGINE_MAX_START_RETRIES", 3),
            recovery_backoff_seconds=_get_env_int(
                "ENGINE_RECOVERY_BACKOFF_SECONDS", 10, 0, 3600
            ),
            launch_command=os.getenv("LAUNCH_COMMAND"),
            config_path=_get_env_str("SUPERVISOR_CONFIG_PATH", "/tmp/supervisord.conf"),
            log_level=_get_env_str(
                "SUPERVISOR_LOG_LEVEL",
                "info",
                ["debug", "info", "warn", "error", "critical"],
            ),
        )
    except ConfigurationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def get_launch_command() -> Optional[str]:
    """Get the launch command from environment variables."""
    command = os.getenv("LAUNCH_COMMAND")
    return command.strip() if command and command.strip() else None
