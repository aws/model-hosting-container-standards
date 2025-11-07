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
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from model_hosting_container_standards.logging_config import get_logger
from model_hosting_container_standards.supervisor.generator import (
    write_supervisord_config,
)
from model_hosting_container_standards.supervisor.models import (
    ConfigurationError,
    parse_environment_variables,
)


class ProcessManager:
    """Manages supervisord process lifecycle without supervisorctl dependency."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process: Optional[subprocess.Popen] = None

    def check_tools_available(self) -> tuple[bool, str]:
        """Check if supervisord is available."""
        if not shutil.which("supervisord"):
            return False, "supervisord"
        return True, ""

    def start(self, config_path: str) -> subprocess.Popen:
        """Start supervisord process with the given configuration."""
        self.logger.info("Starting supervisord...")

        self.process = subprocess.Popen(["supervisord", "-c", config_path])
        time.sleep(1.0)  # Give supervisord time to start

        if self.process.poll() is not None:
            error_msg = (
                f"Supervisord failed to start. Exit code: {self.process.returncode}"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.logger.info(f"Supervisord started with PID: {self.process.pid}")
        return self.process

    def terminate(self) -> None:
        """Terminate the supervisord process."""
        if not self.process:
            return

        try:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.logger.info("Supervisord terminated")
        except subprocess.TimeoutExpired:
            self.logger.warning("Termination timed out, force killing...")
            self.process.kill()
            self.process.wait()
            self.logger.info("Supervisord force killed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


class SignalHandler:
    """Handles process signals for graceful shutdown."""

    def __init__(self, process_manager: ProcessManager, logger: logging.Logger):
        self.process_manager = process_manager
        self.logger = logger
        self._original_handlers: Dict[int, Any] = {}

    def setup(self) -> None:
        """Set up signal handlers."""

        def signal_handler(signum: int, frame) -> None:
            self.logger.info(f"Received signal {signum}, shutting down...")
            self._restore_default_handlers()
            self.process_manager.terminate()
            sys.exit(0)

        # Store original handlers and set new ones
        self._original_handlers[signal.SIGTERM] = signal.signal(
            signal.SIGTERM, signal_handler
        )
        self._original_handlers[signal.SIGINT] = signal.signal(
            signal.SIGINT, signal_handler
        )

    def _restore_default_handlers(self) -> None:
        """Restore default signal handlers to prevent recursive calls."""
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


class StandardSupervisor:
    """Main supervisor orchestrator."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self._setup_logging()

        self.process_manager = ProcessManager(self.logger)
        self.signal_handler = SignalHandler(self.process_manager, self.logger)

    def _setup_logging(self) -> None:
        """Configure logging based on environment."""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))

    def parse_arguments(self) -> List[str]:
        """Parse command-line arguments to extract launch command."""
        launch_command = sys.argv[1:]

        if not launch_command:
            print("ERROR: No launch command provided", file=sys.stderr)
            print(
                "Usage: standard-supervisor <launch_command> [args...]", file=sys.stderr
            )
            print(
                "Example: standard-supervisor vllm serve model --host 0.0.0.0 --port 8080",
                file=sys.stderr,
            )
            sys.exit(1)

        return launch_command

    def run(self) -> int:
        """Main execution method."""
        launch_command = self.parse_arguments()
        self.logger.info(f"Starting: {' '.join(launch_command)}")

        # Check system requirements
        tools_available, missing_tool = self.process_manager.check_tools_available()
        if not tools_available:
            self.logger.error(f"{missing_tool} not found. Install supervisor package.")
            return 1

        # Parse configuration
        try:
            config = parse_environment_variables()
        except ConfigurationError as e:
            self.logger.error(f"Configuration error: {e}")
            return 1

        config_path = config.config_path
        program_name = "llm_engine"

        try:
            # Generate and start supervisor
            self.logger.info("Generating supervisor configuration...")
            write_supervisord_config(
                config_path=config_path,
                config=config,
                launch_command=" ".join(launch_command),
                program_name=program_name,
            )

            supervisord_process = self.process_manager.start(config_path)
            self.signal_handler.setup()

            # Wait for supervisord to exit using poll loop
            # This allows signal handlers to interrupt and respond quickly
            self.logger.info("Supervisord running, waiting for completion...")
            while supervisord_process.poll() is None:
                time.sleep(0.5)  # Check twice per second

            exit_code = supervisord_process.returncode
            self.logger.info(f"Supervisord exited with code: {exit_code}")
            return exit_code

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return 1
        finally:
            # Cleanup - only delete auto-generated temp files, not user-specified configs
            user_specified_config = os.getenv("SUPERVISOR_CONFIG_PATH")
            should_cleanup = (
                config_path.startswith("/tmp/")
                and os.path.exists(config_path)
                and not user_specified_config
            )
            if should_cleanup:
                try:
                    os.unlink(config_path)
                except OSError as e:
                    self.logger.warning(f"Failed to clean up config file: {e}")


def main() -> int:
    """Main entry point for standard-supervisor CLI."""
    supervisor = StandardSupervisor()
    return supervisor.run()


if __name__ == "__main__":
    sys.exit(main())
