#!/usr/bin/env python3
"""
Supervisor Configuration Generator Script

Simple script to generate supervisord configuration files for ML frameworks.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the package to Python path for imports
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir.parent))

try:
    from model_hosting_container_standards.logging_config import get_logger
    from model_hosting_container_standards.supervisor.config import (
        ConfigurationError,
        parse_environment_variables,
    )
    from model_hosting_container_standards.supervisor.framework_config import (
        get_framework_command,
        validate_framework_command,
    )
    from model_hosting_container_standards.supervisor.supervisor_config import (
        write_supervisord_config,
    )
except ImportError as e:
    print(f"ERROR: Failed to import supervisor modules: {e}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    """Main entry point with comprehensive error handling and logging."""
    parser = argparse.ArgumentParser(description="Generate supervisord configuration")

    parser.add_argument(
        "-o", "--output", required=True, help="Output path for config file"
    )
    parser.add_argument(
        "-c", "--command", help="Framework command (overrides env vars)"
    )
    parser.add_argument(
        "-p", "--program-name", default="framework", help="Program name"
    )
    parser.add_argument(
        "--log-level",
        choices=["ERROR", "INFO", "DEBUG"],
        default="ERROR",
        help="Log level",
    )

    args = parser.parse_args()

    # Set up logging based on command line argument
    logger = get_logger(__name__)
    if args.log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif args.log_level == "INFO":
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    try:
        # Get framework command
        framework_command = args.command or get_framework_command()

        if not framework_command:
            error_msg = "No framework command available. Set FRAMEWORK_COMMAND environment variable."
            logger.error(error_msg)
            print(f"ERROR: {error_msg}", file=sys.stderr)
            return 1

        # Validate framework command
        if not validate_framework_command(framework_command):
            logger.warning(f"Framework command may not be valid: '{framework_command}'")

        # Parse configuration from environment
        config = parse_environment_variables()

        # Generate and write configuration
        write_supervisord_config(
            args.output, framework_command, config, args.program_name
        )

        if args.log_level != "ERROR":
            print(f"Configuration written to: {args.output}")

        return 0

    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        print(f"ERROR: Configuration error: {e}", file=sys.stderr)
        return 1
    except (OSError, IOError) as e:
        logger.error(f"File I/O error: {str(e)}")
        print(f"ERROR: File I/O error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
