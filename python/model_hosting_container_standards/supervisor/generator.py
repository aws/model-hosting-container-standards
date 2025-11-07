"""
Supervisord configuration generation for ML framework process management.

This module provides functionality to generate supervisord configuration files
based on environment variables and framework-specific settings.
"""

from pathlib import Path

from ..logging_config import get_logger
from .models import ConfigurationError, SupervisorConfig

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
# FATAL state examples (supervisorctl status output):
#   app                       FATAL     Exited too quickly (process log may have details)
#   app                       FATAL     can't find command '/path/to/missing/binary'
#   app                       FATAL     spawn error
#
# When a program enters FATAL state (too many restart failures), the entrypoint script
# will detect this and exit with code 1 to signal container failure.
def get_base_config_template(
    program_name: str,
    log_level: str,
    framework_command: str,
    auto_restart: str,
    max_start_retries: int,
) -> dict:
    """Get base supervisord configuration as dictionary structure.

    Note: We don't use supervisorctl for process management, but supervisord
    still needs minimal RPC configuration for its internal operations.
    """
    return {
        "supervisord": {
            "nodaemon": "true",
            "loglevel": log_level,
            "logfile": f"/tmp/supervisord-{program_name}.log",
            "logfile_maxbytes": "50MB",
            "logfile_backups": "3",
            "pidfile": f"/tmp/supervisord-{program_name}.pid",
        },
        f"program:{program_name}": {
            "command": framework_command,
            "autostart": "true",
            "autorestart": auto_restart,
            "startretries": str(max_start_retries),
            "stdout_logfile": "/dev/stdout",
            "stdout_logfile_maxbytes": "0",
            "stderr_logfile": "/dev/stderr",
            "stderr_logfile_maxbytes": "0",
            "exitcodes": "255",
            "startsecs": "1",
            "stopsignal": "TERM",
            "stopwaitsecs": "30",
            "stopasgroup": "true",
            "killasgroup": "true",
        },
    }


def generate_supervisord_config(
    config: SupervisorConfig,
    launch_command: str,
    program_name: str = "app",
) -> str:
    """Generate supervisord configuration content with validation and logging.

    Creates a supervisord configuration file content based on the provided
    configuration and launch command. Merges custom SUPERVISOR_* configuration
    with the base template.

    Args:
        config: SupervisorConfig instance with supervisor settings.
        launch_command: Command to execute in the supervised program
        program_name: Name for the supervisord program section

    Returns:
        str: Complete supervisord configuration file content

    Raises:
        ConfigurationError: If configuration generation fails
        ValueError: If required parameters are invalid
    """
    # Validate required parameters
    if not program_name or not program_name.strip():
        error_msg = "Program name cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Validate launch command parameter
    if not launch_command or not launch_command.strip():
        error_msg = "Launch command cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Convert boolean auto_recovery to supervisord format
    # Use "true" to always restart (except for exitcodes=255 which is "expected")
    auto_restart = "true" if config.auto_recovery else "false"

    try:
        # Get base configuration as dictionary
        base_config = get_base_config_template(
            program_name=program_name,
            log_level=config.log_level,
            framework_command=launch_command,
            auto_restart=auto_restart,
            max_start_retries=config.max_start_retries,
        )

        # Merge custom configuration sections
        merged_config = _merge_custom_sections(base_config, config.custom_sections)

        # Convert to INI format string
        return _dict_to_ini_string(merged_config)

    except Exception as e:
        error_msg = f"Failed to generate supervisord configuration: {str(e)}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e


def write_supervisord_config(
    config_path: str,
    config: SupervisorConfig,
    launch_command: str,
    program_name: str = "app",
) -> None:
    """Write supervisord configuration to file with comprehensive error handling.

    Generates supervisord configuration content and writes it to the
    specified file path. Creates parent directories if they don't exist.

    Args:
        config_path: Path where the configuration file should be written
        config: SupervisorConfig instance with supervisor settings.
        launch_command: Command to execute in the supervised program
        program_name: Name for the supervisord program section

    Raises:
        ConfigurationError: If configuration generation or validation fails
        OSError: If the configuration file cannot be written
        ValueError: If required parameters are invalid
    """
    try:
        # Generate configuration content
        config_content = generate_supervisord_config(
            config, launch_command, program_name
        )

        # Create parent directories if they don't exist
        Path(config_path).parent.mkdir(parents=True, exist_ok=True, mode=0o755)

        # Write configuration to file
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

        logger.info(f"Successfully wrote supervisord configuration to '{config_path}'")

    except (OSError, IOError) as e:
        error_msg = f"Failed to write configuration file '{config_path}': {str(e)}"
        logger.error(error_msg)
        raise OSError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error writing configuration: {str(e)}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e


def _merge_custom_sections(base_config: dict, custom_sections: dict) -> dict:
    """Merge custom configuration sections with base configuration.

    Args:
        base_config: Base configuration dictionary
        custom_sections: Custom configuration sections to merge

    Returns:
        dict: Merged configuration dictionary
    """
    if not custom_sections:
        return base_config

    # Merge custom sections directly into base config
    for section_name, custom_config in custom_sections.items():
        if section_name in base_config:
            # Update existing section
            for key, value in custom_config.items():
                if key in base_config[section_name]:
                    logger.info(f"Overrode setting in [{section_name}]: {key}={value}")
                else:
                    logger.info(
                        f"Added custom setting to [{section_name}]: {key}={value}"
                    )
                base_config[section_name][key] = value
        else:
            # Add new section
            base_config[section_name] = custom_config.copy()
            logger.info(
                f"Added new custom section [{section_name}] with {len(custom_config)} settings"
            )

    return base_config


def _dict_to_ini_string(config_dict: dict) -> str:
    """Convert configuration dictionary to INI format string using configparser.

    Args:
        config_dict: Configuration dictionary

    Returns:
        str: INI format configuration string
    """
    import configparser
    from io import StringIO

    config = configparser.ConfigParser()

    # Add sections and their key-value pairs
    for section_name, section_config in config_dict.items():
        config.add_section(section_name)
        for key, value in section_config.items():
            config.set(section_name, key, str(value))

    # Write to string buffer
    output = StringIO()
    config.write(output)

    return output.getvalue()
