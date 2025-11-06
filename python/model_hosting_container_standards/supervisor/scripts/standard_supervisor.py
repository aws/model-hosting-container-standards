#!/usr/bin/env python3
"""
Standard Supervisor CLI Script

Simplified CLI command that wraps and manages user launch processes under supervision.
Users can prepend 'standard-supervisor' to their existing launch commands.

Usage:
    standard-supervisor <launch_command> [args...]

Example:
    standard-supervisor vllm serve model --host 0.0.0.0 --port 8080
"""

import logging
import sys
from typing import List

from model_hosting_container_standards.logging_config import get_logger


def parse_arguments() -> List[str]:
    """
    Parse command-line arguments to extract launch command.

    Returns:
        List of launch command and arguments

    Raises:
        SystemExit: If no launch command is provided
    """
    # Get all command line arguments except the script name
    launch_command = sys.argv[1:]

    # Validate that launch command is provided
    if not launch_command:
        # Set up basic logging for error reporting
        logger = get_logger(__name__)
        error_msg = "No launch command provided"
        logger.error(error_msg)
        print(f"ERROR: {error_msg}", file=sys.stderr)
        print("Usage: standard-supervisor <launch_command> [args...]", file=sys.stderr)
        print(
            "Example: standard-supervisor vllm serve model --host 0.0.0.0 --port 8080",
            file=sys.stderr,
        )
        sys.exit(1)

    return launch_command


def main() -> int:
    """
    Main entry point for standard-supervisor CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command-line arguments
    launch_command = parse_arguments()

    # Set up logging with default INFO level
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(f"Starting: {' '.join(launch_command)}")

    # TODO: In future tasks, this will integrate with supervisor configuration and execution
    # For now, we just validate and log the command
    print(f"Standard supervisor would execute: {' '.join(launch_command)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
