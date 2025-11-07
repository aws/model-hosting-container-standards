#!/usr/bin/env python3
"""
Supervisor Configuration Generator Script

Simple script to generate supervisord configuration files for ML frameworks.
"""

import argparse
import logging
import sys

from model_hosting_container_standards.logging_config import get_logger
from model_hosting_container_standards.supervisor.generator import (
    write_supervisord_config,
)
from model_hosting_container_standards.supervisor.models import (
    ConfigurationError,
    parse_environment_variables,
)


def main() -> int:
    """Main entry point with comprehensive error handling and logging."""
    parser = argparse.ArgumentParser(description="Generate supervisord configuration")

    parser.add_argument(
        "-o", "--output", required=True, help="Output path for config file"
    )

    parser.add_argument("-p", "--program-name", default="app", help="Program name")
    parser.add_argument(
        "--log-level",
        choices=["ERROR", "INFO", "DEBUG"],
        default="ERROR",
        help="Log level",
    )
    parser.add_argument("command", nargs="+", help="Launch command and arguments")

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
        # Parse configuration from environment
        config = parse_environment_variables()

        # Get launch command from CLI arguments
        launch_command = " ".join(args.command)

        # Generate and write configuration
        write_supervisord_config(args.output, config, launch_command, args.program_name)

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
