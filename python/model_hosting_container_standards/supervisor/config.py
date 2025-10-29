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
        recovery_backoff_seconds: Wait time in seconds between restart attempts
        framework_command: Custom command to run the framework process
        config_path: Path where supervisord configuration files are stored
        log_level: Logging level for supervisord (debug, info, warn, error, critical)
        framework_name: Name of the ML framework being managed
    """

    auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_backoff_seconds: int = 10
    framework_command: Optional[str] = None
    config_path: str = "/opt/aws/supervisor/conf.d/supervisord.conf"
    log_level: str = "info"


def validate_environment_variable(
    var_name: str,
    var_value: str,
    var_type: type,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    allowed_values: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate an environment variable value.

    Args:
        var_name: Name of the environment variable
        var_value: Value to validate
        var_type: Expected type (int, str, bool)
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        allowed_values: List of allowed string values

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if var_type == int:
            parsed_value = int(var_value)
            if min_value is not None and parsed_value < min_value:
                return False, f"{var_name} must be >= {min_value}, got {parsed_value}"
            if max_value is not None and parsed_value > max_value:
                return False, f"{var_name} must be <= {max_value}, got {parsed_value}"
        elif var_type == bool:
            if var_value.lower() not in (
                "true",
                "false",
                "1",
                "0",
                "yes",
                "no",
                "on",
                "off",
            ):
                return (
                    False,
                    f"{var_name} must be a boolean value (true/false, 1/0, yes/no, on/off), got '{var_value}'",
                )
        elif var_type == str:
            if allowed_values and var_value.lower() not in allowed_values:
                return (
                    False,
                    f"{var_name} must be one of {allowed_values}, got '{var_value}'",
                )
            if not var_value.strip():
                return False, f"{var_name} cannot be empty"

        return True, None
    except (ValueError, TypeError) as e:
        return False, f"{var_name} has invalid format: {str(e)}"


def parse_environment_variables() -> SupervisorConfig:
    """Parse environment variables and return SupervisorConfig instance with validation.

    Returns:
        SupervisorConfig: Validated configuration instance

    Raises:
        ConfigurationError: If critical configuration validation fails
    """
    config = SupervisorConfig()
    validation_errors: List[str] = []
    validation_warnings = []

    # Parse boolean auto_recovery
    auto_recovery_str = os.getenv("ENGINE_AUTO_RECOVERY", "true")
    is_valid, error_msg = validate_environment_variable(
        "ENGINE_AUTO_RECOVERY", auto_recovery_str, bool
    )
    if is_valid:
        config.auto_recovery = auto_recovery_str.lower() in ("true", "1", "yes", "on")
    else:
        validation_warnings.append(
            f"Invalid ENGINE_AUTO_RECOVERY: {error_msg}. Using default: {config.auto_recovery}"
        )

    # Parse integer fields with validation
    max_attempts_str = os.getenv("ENGINE_MAX_RECOVERY_ATTEMPTS")
    if max_attempts_str:
        is_valid, error_msg = validate_environment_variable(
            "ENGINE_MAX_RECOVERY_ATTEMPTS",
            max_attempts_str,
            int,
            min_value=0,
            max_value=100,
        )
        if is_valid:
            config.max_recovery_attempts = int(max_attempts_str)
        else:
            validation_warnings.append(
                f"Invalid ENGINE_MAX_RECOVERY_ATTEMPTS: {error_msg}. Using default: {config.max_recovery_attempts}"
            )

    backoff_str = os.getenv("ENGINE_RECOVERY_BACKOFF_SECONDS")
    if backoff_str:
        is_valid, error_msg = validate_environment_variable(
            "ENGINE_RECOVERY_BACKOFF_SECONDS",
            backoff_str,
            int,
            min_value=0,
            max_value=3600,
        )
        if is_valid:
            config.recovery_backoff_seconds = int(backoff_str)
        else:
            validation_warnings.append(
                f"Invalid ENGINE_RECOVERY_BACKOFF_SECONDS: {error_msg}. Using default: {config.recovery_backoff_seconds}"
            )

    # Parse string fields with validation
    framework_command = os.getenv("FRAMEWORK_COMMAND")
    if framework_command:
        is_valid, error_msg = validate_environment_variable(
            "FRAMEWORK_COMMAND", framework_command, str
        )
        if is_valid:
            config.framework_command = framework_command.strip()
        else:
            validation_warnings.append(f"Invalid FRAMEWORK_COMMAND: {error_msg}")

    config_path = os.getenv("SUPERVISOR_CONFIG_PATH")
    if config_path:
        is_valid, error_msg = validate_environment_variable(
            "SUPERVISOR_CONFIG_PATH", config_path, str
        )
        if is_valid:
            config.config_path = config_path.strip()
        else:
            validation_warnings.append(
                f"Invalid SUPERVISOR_CONFIG_PATH: {error_msg}. Using default: {config.config_path}"
            )

    # Parse log level with validation
    log_level = os.getenv("SUPERVISOR_LOG_LEVEL", "info")
    allowed_log_levels = ["debug", "info", "warn", "error", "critical"]
    is_valid, error_msg = validate_environment_variable(
        "SUPERVISOR_LOG_LEVEL", log_level, str, allowed_values=allowed_log_levels
    )
    if is_valid:
        config.log_level = log_level.lower().strip()
    else:
        validation_warnings.append(
            f"Invalid SUPERVISOR_LOG_LEVEL: {error_msg}. Using default: {config.log_level}"
        )

    # Log all validation warnings
    for warning in validation_warnings:
        logger.warning(warning)

    # Raise error if there are critical validation failures
    if validation_errors:
        error_msg = "Critical configuration validation errors:\n" + "\n".join(
            validation_errors
        )
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    return config


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
