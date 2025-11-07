"""Configuration management for supervisor process management."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""

    pass


@dataclass
class SupervisorConfig:
    """Configuration for supervisor process management system.

    Hybrid Environment Variable Design:
    - Application config: Simple names (AUTO_RECOVERY, MAX_START_RETRIES, LOG_LEVEL)
    - Supervisord config: SUPERVISOR_{SECTION}_{KEY} pattern for custom overrides
    - Section names with colons: Use double underscore __ to represent colon :

    Examples:
    - AUTO_RECOVERY=false (application behavior)
    - MAX_START_RETRIES=5 (application behavior)
    - LOG_LEVEL=debug (application behavior)
    - SUPERVISOR_PROGRAM_STARTSECS=10 (supervisord [program] section override)
    - SUPERVISOR_SUPERVISORD_LOGLEVEL=debug (supervisord [supervisord] section override)
    - SUPERVISOR_PROGRAM__WEB_COMMAND="gunicorn app:app" (supervisord [program:web] section)
    - SUPERVISOR_RPCINTERFACE__SUPERVISOR_FACTORY=... (supervisord [rpcinterface:supervisor] section)
    """

    auto_recovery: bool = True
    max_start_retries: int = 3
    config_path: str = "/tmp/supervisord.conf"
    log_level: str = "info"
    custom_sections: Dict[str, Dict[str, str]] = field(default_factory=dict)


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
        # Parse custom SUPERVISOR_* configuration sections
        custom_sections = _parse_supervisor_custom_sections()

        return SupervisorConfig(
            auto_recovery=_parse_bool(os.getenv("AUTO_RECOVERY", "true")),
            max_start_retries=_get_env_int("MAX_START_RETRIES", 3),
            config_path=_get_env_str("SUPERVISOR_CONFIG_PATH", "/tmp/supervisord.conf"),
            log_level=_get_env_str(
                "LOG_LEVEL",
                "info",
                ["debug", "info", "warn", "error", "critical"],
            ),
            custom_sections=custom_sections,
        )
    except ConfigurationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def _parse_supervisor_custom_sections() -> Dict[str, Dict[str, str]]:
    """
    Parse SUPERVISOR_{SECTION}_{KEY}=VALUE environment variables for supervisord configuration.

    Pattern: SUPERVISOR_SECTION_KEY -> [section] key=value
    Special handling for section names with colons:
    - Double underscore __ in section name becomes colon :

    Examples:
    - SUPERVISOR_PROGRAM_STARTSECS=10 -> [program] startsecs=10
    - SUPERVISOR_SUPERVISORD_LOGLEVEL=debug -> [supervisord] loglevel=debug
    - SUPERVISOR_PROGRAM__WEB_COMMAND="gunicorn app:app" -> [program:web] command=gunicorn app:app
    - SUPERVISOR_RPCINTERFACE__SUPERVISOR_FACTORY=... -> [rpcinterface:supervisor] factory=...

    Skips SUPERVISOR_CONFIG_PATH (used for file path, not supervisord config).

    Returns:
        Dictionary mapping section names to their key-value configurations
    """
    custom_sections: Dict[str, Dict[str, str]] = {}

    for env_var, value in os.environ.items():
        if not env_var.startswith("SUPERVISOR_"):
            continue

        # Skip the config path variable
        if env_var == "SUPERVISOR_CONFIG_PATH":
            continue

        # Remove SUPERVISOR_ prefix
        remaining = env_var[11:]  # len("SUPERVISOR_") = 11

        # Find the last underscore to separate key from section
        last_underscore = remaining.rfind("_")
        if last_underscore == -1:
            logger.warning(
                f"Invalid SUPERVISOR_ environment variable format: '{env_var}'. "
                f"Expected format: SUPERVISOR_SECTION_KEY=value"
            )
            continue

        section_part = remaining[:last_underscore]
        key_name = remaining[last_underscore + 1 :].lower()

        # Convert double underscores to colons in section name first
        section_name = section_part.replace("__", ":").lower()

        # Validate section and key are not empty after processing
        # Also check for invalid section names (starting with underscore indicates empty section before __)
        if (
            not section_name
            or section_name.startswith(":")
            or section_name.endswith(":")
            or section_name.startswith("_")
        ):
            logger.warning(
                f"Invalid SUPERVISOR_ environment variable: '{env_var}' has invalid section name. "
                f"Expected format: SUPERVISOR_SECTION_KEY=value"
            )
            continue

        if not key_name:
            logger.warning(
                f"Invalid SUPERVISOR_ environment variable: '{env_var}' has empty key name. "
                f"Expected format: SUPERVISOR_SECTION_KEY=value"
            )
            continue

        # Initialize section if it doesn't exist
        if section_name not in custom_sections:
            custom_sections[section_name] = {}

        # Store the custom configuration
        custom_sections[section_name][key_name] = value.strip()
        logger.debug(
            f"Found custom supervisor configuration: [{section_name}] {key_name}={value}"
        )

    return custom_sections
