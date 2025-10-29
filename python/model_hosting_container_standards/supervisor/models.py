"""
Configuration management for supervisor process management.

This module provides configuration dataclasses and environment variable
parsing for the supervisord-based process management system.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..logging_config import get_logger

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""

    pass


@dataclass
class SupervisorConfig:
    """Configuration for supervisor process management system.

    This dataclass holds all configuration options for the supervisord-based
    process management system, with defaults that can be overridden by
    environment variables.

    Attributes:
        auto_recovery: Enable/disable automatic restart of framework processes
        max_recovery_attempts: Maximum number of restart attempts before giving up
        recovery_backoff_seconds: Wait time in seconds between restart attempts (currently unused)
        launch_command: Custom command to run the framework process
        config_path: Path where supervisord configuration files are stored
        log_level: Logging level for supervisord (debug, info, warn, error, critical)

    """

    auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_backoff_seconds: int = (
        10  # NOTE: Currently unused - supervisord doesn't support backoff natively
    )
    launch_command: Optional[str] = None
    config_path: str = "/tmp/supervisord.conf"
    log_level: str = "info"


def validate_environment_variable(
    var_name: str,
    value: str,
    var_type: type = str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    allowed_values: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate an environment variable value.

    Args:
        var_name: Name of the environment variable
        value: Value to validate
        var_type: Expected type (int, str, bool)
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        allowed_values: List of allowed string values

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if var_type == int:
            parsed_value = int(value)
            if min_value is not None and parsed_value < min_value:
                return False, f"{var_name} must be >= {min_value}, got {parsed_value}"
            if max_value is not None and parsed_value > max_value:
                return False, f"{var_name} must be <= {max_value}, got {parsed_value}"
            return True, None
        elif var_type == bool:
            if value.lower() not in (
                "true",
                "false",
                "1",
                "0",
                "yes",
                "no",
                "on",
                "off",
            ):
                return False, f"{var_name} must be a boolean value, got '{value}'"
            return True, None
        elif var_type == str:
            if not value.strip():
                return False, f"{var_name} cannot be empty"
            if allowed_values and value.lower() not in allowed_values:
                return (
                    False,
                    f"{var_name} must be one of {allowed_values}, got '{value}'",
                )
            return True, None
        else:
            return True, None
    except (ValueError, TypeError) as e:
        return False, f"{var_name} has invalid format: {str(e)}"


def get_validated_env_var(
    var_name: str,
    default_value=None,
    var_type: type = str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    allowed_values: Optional[List[str]] = None,
    required: bool = False,
):
    """Get and validate an environment variable value.

    Args:
        var_name: Name of the environment variable
        default_value: Default value if env var is not set
        var_type: Expected type (int, str, bool)
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        allowed_values: List of allowed string values
        required: Whether the variable is required

    Returns:
        Validated and parsed value

    Raises:
        ConfigurationError: If validation fails and no default provided
    """
    var_value = os.getenv(var_name)

    if var_value is None:
        if required:
            raise ConfigurationError(
                f"Required environment variable {var_name} is not set"
            )
        return default_value

    try:
        if var_type == int:
            parsed_value = int(var_value)
            if min_value is not None and parsed_value < min_value:
                raise ConfigurationError(
                    f"{var_name} must be >= {min_value}, got {parsed_value}"
                )
            if max_value is not None and parsed_value > max_value:
                raise ConfigurationError(
                    f"{var_name} must be <= {max_value}, got {parsed_value}"
                )
            return parsed_value
        elif var_type == bool:
            if var_value.lower() not in ("true", "false", "1", "0"):
                raise ConfigurationError(
                    f"{var_name} must be a boolean value (true/false, 1/0), got '{var_value}'"
                )
            return var_value.lower() in ("true", "1")
        elif var_type == str:
            if allowed_values and var_value.lower() not in allowed_values:
                raise ConfigurationError(
                    f"{var_name} must be one of {allowed_values}, got '{var_value}'"
                )
            if not var_value.strip():
                raise ConfigurationError(f"{var_name} cannot be empty")
            return var_value.strip()
        else:
            return var_value
    except (ValueError, TypeError) as e:
        raise ConfigurationError(f"{var_name} has invalid format: {str(e)}")


def parse_environment_variables() -> SupervisorConfig:
    """Parse environment variables and return SupervisorConfig instance with validation.

    Returns:
        SupervisorConfig: Validated configuration instance

    Raises:
        ConfigurationError: If critical configuration validation fails
    """
    config = SupervisorConfig()

    try:
        config.auto_recovery = get_validated_env_var(
            "ENGINE_AUTO_RECOVERY", default_value=config.auto_recovery, var_type=bool
        )

        config.max_recovery_attempts = get_validated_env_var(
            "ENGINE_MAX_RECOVERY_ATTEMPTS",
            default_value=config.max_recovery_attempts,
            var_type=int,
            min_value=0,
            max_value=100,
        )

        config.recovery_backoff_seconds = get_validated_env_var(
            "ENGINE_RECOVERY_BACKOFF_SECONDS",
            default_value=config.recovery_backoff_seconds,
            var_type=int,
            min_value=0,
            max_value=3600,
        )  # NOTE: Currently unused - supervisord doesn't support backoff natively

        config.launch_command = get_validated_env_var(
            "LAUNCH_COMMAND",
            default_value=config.launch_command,
            var_type=str,
        )

        config.config_path = get_validated_env_var(
            "SUPERVISOR_CONFIG_PATH",
            default_value=config.config_path,
            var_type=str,
        )

        config.log_level = get_validated_env_var(
            "SUPERVISOR_LOG_LEVEL",
            default_value=config.log_level,
            var_type=str,
            allowed_values=["debug", "info", "warn", "error", "critical"],
        )

    except ConfigurationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    return config


def get_launch_command() -> Optional[str]:
    """Get the launch command from environment variables.

    Returns:
        Optional[str]: Launch command to execute, or None if not available
    """
    command = os.getenv("LAUNCH_COMMAND")
    if command and command.strip():
        return command.strip()
    return None


def validate_config_directory(config_path: str) -> Tuple[bool, Optional[str]]:
    """Validate that the configuration directory can be created and is writable.

    Args:
        config_path: Path to the configuration file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        config_dir = os.path.dirname(config_path)

        # Check if directory exists or can be created
        if not os.path.exists(config_dir):
            try:
                os.makedirs(config_dir, mode=0o755, exist_ok=True)
                logger.debug(f"Created configuration directory: {config_dir}")
            except OSError as e:
                return (
                    False,
                    f"Cannot create configuration directory '{config_dir}': {str(e)}",
                )

        # Check if directory is writable
        if not os.access(config_dir, os.W_OK):
            return False, f"Configuration directory '{config_dir}' is not writable"

        # Check if config file exists and is writable, or can be created
        if os.path.exists(config_path):
            if not os.access(config_path, os.W_OK):
                return (
                    False,
                    f"Configuration file '{config_path}' exists but is not writable",
                )
        else:
            # Try to create a test file to verify write permissions
            try:
                test_file = os.path.join(config_dir, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except OSError as e:
                return (
                    False,
                    f"Cannot write to configuration directory '{config_dir}': {str(e)}",
                )

        return True, None

    except Exception as e:
        return (
            False,
            f"Unexpected error validating configuration path '{config_path}': {str(e)}",
        )
