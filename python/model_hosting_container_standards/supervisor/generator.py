"""
Supervisord configuration generation for ML framework process management.

This module provides functionality to generate supervisord configuration files
based on environment variables and framework-specific settings.
"""

import os

from ..logging_config import get_logger
from .models import ConfigurationError, SupervisorConfig, validate_config_directory

logger = get_logger(__name__)


# Supervisord configuration template for LLM service monitoring
#
# Key behavior: LLM services are expected to run indefinitely. Any exit is considered an error.
# - exitcodes=255: Only exit code 255 is "expected" - all other exits (0,1,2...) trigger restart
# - startsecs=1: Process must run at least 1 second to be considered successfully started
# - autorestart=unexpected: Only restart on unexpected exit codes (not 255)
#   When ENGINE_AUTO_RECOVERY=false, autorestart=false to disable all restarts
# - startretries=N: Maximum restart attempts before entering FATAL state
#
# When a program enters FATAL state (too many restart failures), the entrypoint script
# will detect this and exit with code 1 to signal container failure.
SUPERVISORD_CONFIG_TEMPLATE = """[unix_http_server]
file=/tmp/supervisor-{program_name}.sock

[supervisord]
nodaemon=true
loglevel={log_level}
logfile=/dev/stdout
logfile_maxbytes=0
pidfile=/tmp/supervisord-{program_name}.pid

[supervisorctl]
serverurl=unix:///tmp/supervisor-{program_name}.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[program:{program_name}]
command={framework_command}
autostart=true
autorestart={auto_restart}
startretries={max_start_retries}
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
exitcodes=255
startsecs=1
"""


def generate_supervisord_config(
    config: SupervisorConfig,
    program_name: str = "llm-engine",
) -> str:
    """Generate supervisord configuration content with validation and logging.

    Creates a supervisord configuration file content based on the provided
    configuration.

    Args:
        config: SupervisorConfig instance with supervisor settings.
        program_name: Name for the supervisord program section

    Returns:
        str: Complete supervisord configuration file content

    Raises:
        ConfigurationError: If configuration validation fails
        ValueError: If required parameters are invalid
    """
    # Validate required parameters
    if not program_name or not program_name.strip():
        error_msg = "Program name cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate launch command from config
    if not config.launch_command or not config.launch_command.strip():
        error_msg = "Launch command in configuration cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Convert boolean auto_recovery to supervisord format
    auto_restart = "true" if config.auto_recovery else "false"

    try:
        # Generate configuration content
        config_content = SUPERVISORD_CONFIG_TEMPLATE.format(
            log_level=config.log_level,
            program_name=program_name,
            framework_command=config.launch_command,
            auto_restart=auto_restart,
            max_start_retries=config.max_start_retries,
        )

        return config_content

    except Exception as e:
        error_msg = f"Failed to generate supervisord configuration: {str(e)}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e


def write_supervisord_config(
    config_path: str,
    config: SupervisorConfig,
    program_name: str = "llm-engine",
) -> None:
    """Write supervisord configuration to file with comprehensive error handling.

    Generates supervisord configuration content and writes it to the
    specified file path. Creates parent directories if they don't exist.

    Args:
        config_path: Path where the configuration file should be written
        config: SupervisorConfig instance with supervisor settings.
        program_name: Name for the supervisord program section

    Raises:
        ConfigurationError: If configuration generation or validation fails
        OSError: If the configuration file cannot be written
        ValueError: If required parameters are invalid
    """
    # Validate config path
    if not config_path or not config_path.strip():
        error_msg = "Configuration path cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate that we can write to the configuration directory
    is_valid, validation_error = validate_config_directory(config_path)
    if not is_valid:
        logger.error(f"Configuration directory validation failed: {validation_error}")
        raise ConfigurationError(f"Cannot write configuration: {validation_error}")

    try:
        # Generate configuration content
        config_content = generate_supervisord_config(config, program_name)

        # Create parent directories if they don't exist
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, mode=0o755, exist_ok=True)

        # Write configuration to file
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        # Verify the file was written successfully
        if not os.path.exists(config_path):
            error_msg = f"Configuration file was not created: {config_path}"
            logger.error(error_msg)
            raise OSError(error_msg)

        file_size = os.path.getsize(config_path)
        logger.info(
            f"Successfully wrote supervisord configuration ({file_size} bytes) to '{config_path}'"
        )

    except (OSError, IOError) as e:
        error_msg = f"Failed to write configuration file '{config_path}': {str(e)}"
        logger.error(error_msg)
        raise OSError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error writing configuration: {str(e)}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e
