"""
Unit tests for StandardSupervisor CLI components.

These tests focus on individual components of the standard-supervisor CLI
without requiring actual supervisor processes or system integration.
"""

import os
import signal
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from model_hosting_container_standards.supervisor.scripts.standard_supervisor import (
    ProcessManager,
    ProcessMonitor,
    SignalHandler,
    StandardSupervisor,
)


class TestProcessManager:
    """Test the ProcessManager class."""

    def test_init(self):
        """Test ProcessManager initialization."""
        logger = Mock()
        manager = ProcessManager(logger)

        assert manager.logger == logger
        assert manager.process is None

    @patch("shutil.which")
    def test_check_tools_available_success(self, mock_which):
        """Test successful tool availability check."""
        mock_which.return_value = "/usr/bin/supervisord"

        logger = Mock()
        manager = ProcessManager(logger)

        available, missing = manager.check_tools_available()

        assert available is True
        assert missing == ""
        assert mock_which.call_count == 2  # supervisord and supervisorctl

    @patch("shutil.which")
    def test_check_tools_available_missing_supervisord(self, mock_which):
        """Test tool availability check with missing supervisord."""

        def mock_which_side_effect(tool):
            if tool == "supervisord":
                return None
            return "/usr/bin/supervisorctl"

        mock_which.side_effect = mock_which_side_effect

        logger = Mock()
        manager = ProcessManager(logger)

        available, missing = manager.check_tools_available()

        assert available is False
        assert missing == "supervisord"

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    @patch("time.sleep")
    def test_start_success(self, mock_sleep, mock_run, mock_popen):
        """Test successful process start."""
        # Mock successful process start
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        logger = Mock()
        manager = ProcessManager(logger)

        result = manager.start("/tmp/test.conf")

        assert result == mock_process
        assert manager.process == mock_process
        mock_popen.assert_called_once_with(["supervisord", "-c", "/tmp/test.conf"])
        mock_sleep.assert_called_once_with(1.0)

    @patch("subprocess.Popen")
    @patch("time.sleep")
    def test_start_failure(self, mock_sleep, mock_popen):
        """Test process start failure."""
        # Mock failed process start
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process exited with error
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        logger = Mock()
        manager = ProcessManager(logger)

        with pytest.raises(RuntimeError, match="Supervisord failed to start"):
            manager.start("/tmp/test.conf")

    def test_terminate_no_process(self):
        """Test terminate when no process is running."""
        logger = Mock()
        manager = ProcessManager(logger)

        # Should not raise an exception
        manager.terminate()

    def test_terminate_success(self):
        """Test successful process termination."""
        mock_process = Mock()
        mock_process.terminate.return_value = None
        mock_process.wait.return_value = 0

        logger = Mock()
        manager = ProcessManager(logger)
        manager.process = mock_process

        manager.terminate()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)

    def test_terminate_timeout_and_kill(self):
        """Test process termination with timeout and force kill."""
        mock_process = Mock()
        mock_process.terminate.return_value = None
        mock_process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), 0]
        mock_process.kill.return_value = None

        logger = Mock()
        manager = ProcessManager(logger)
        manager.process = mock_process

        manager.terminate()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert mock_process.wait.call_count == 2


class TestProcessMonitor:
    """Test the ProcessMonitor class."""

    def test_init(self):
        """Test ProcessMonitor initialization."""
        logger = Mock()
        monitor = ProcessMonitor("/tmp/test.conf", "test-program", logger)

        assert monitor.config_path == "/tmp/test.conf"
        assert monitor.program_name == "test-program"
        assert monitor.logger == logger

    @patch("subprocess.run")
    def test_check_fatal_state_true(self, mock_run):
        """Test fatal state detection when process is FATAL."""
        mock_run.return_value = Mock(stdout="test-program FATAL Exited too quickly")

        logger = Mock()
        monitor = ProcessMonitor("/tmp/test.conf", "test-program", logger)

        result = monitor.check_fatal_state()

        assert result is True
        mock_run.assert_called_once_with(
            ["supervisorctl", "-c", "/tmp/test.conf", "status", "test-program"],
            capture_output=True,
            text=True,
            timeout=3,
        )

    @patch("subprocess.run")
    def test_check_fatal_state_false(self, mock_run):
        """Test fatal state detection when process is not FATAL."""
        mock_run.return_value = Mock(stdout="test-program RUNNING pid 12345")

        logger = Mock()
        monitor = ProcessMonitor("/tmp/test.conf", "test-program", logger)

        result = monitor.check_fatal_state()

        assert result is False

    @patch("subprocess.run")
    def test_check_fatal_state_exception(self, mock_run):
        """Test fatal state detection when supervisorctl fails."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 3)

        logger = Mock()
        monitor = ProcessMonitor("/tmp/test.conf", "test-program", logger)

        result = monitor.check_fatal_state()

        assert result is False  # Should return False on exception


class TestSignalHandler:
    """Test the SignalHandler class."""

    def test_init(self):
        """Test SignalHandler initialization."""
        process_manager = Mock()
        logger = Mock()
        handler = SignalHandler(process_manager, logger)

        assert handler.process_manager == process_manager
        assert handler.logger == logger
        assert handler._original_handlers == {}

    @patch("signal.signal")
    def test_setup(self, mock_signal):
        """Test signal handler setup."""
        process_manager = Mock()
        logger = Mock()
        handler = SignalHandler(process_manager, logger)

        # Mock original handlers
        original_term = Mock()
        original_int = Mock()
        mock_signal.side_effect = [original_term, original_int]

        handler.setup()

        # Verify signal handlers were set
        assert mock_signal.call_count == 2
        calls = mock_signal.call_args_list
        assert calls[0][0][0] == signal.SIGTERM
        assert calls[1][0][0] == signal.SIGINT

        # Verify original handlers were stored
        assert handler._original_handlers[signal.SIGTERM] == original_term
        assert handler._original_handlers[signal.SIGINT] == original_int


class TestStandardSupervisor:
    """Test the StandardSupervisor main class."""

    def test_init(self):
        """Test StandardSupervisor initialization."""
        supervisor = StandardSupervisor()

        assert supervisor.logger is not None
        assert supervisor.process_manager is not None
        assert supervisor.signal_handler is not None

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_setup_logging_debug(self):
        """Test logging setup with DEBUG level."""
        supervisor = StandardSupervisor()

        # Logger should be set to DEBUG level
        assert supervisor.logger.level <= 10  # DEBUG is 10

    @patch.dict(os.environ, {"LOG_LEVEL": "ERROR"})
    def test_setup_logging_error(self):
        """Test logging setup with ERROR level."""
        supervisor = StandardSupervisor()

        # Logger should be set to ERROR level
        assert supervisor.logger.level >= 40  # ERROR is 40

    def test_parse_arguments_valid(self):
        """Test argument parsing with valid arguments."""
        supervisor = StandardSupervisor()

        with patch.object(sys, "argv", ["standard-supervisor", "echo", "hello"]):
            result = supervisor.parse_arguments()
            assert result == ["echo", "hello"]

    def test_parse_arguments_complex(self):
        """Test argument parsing with complex command."""
        supervisor = StandardSupervisor()

        with patch.object(
            sys,
            "argv",
            ["standard-supervisor", "vllm", "serve", "model", "--host", "0.0.0.0"],
        ):
            result = supervisor.parse_arguments()
            assert result == ["vllm", "serve", "model", "--host", "0.0.0.0"]

    def test_parse_arguments_empty(self):
        """Test argument parsing with no arguments."""
        supervisor = StandardSupervisor()

        with patch.object(sys, "argv", ["standard-supervisor"]):
            with pytest.raises(SystemExit) as exc_info:
                supervisor.parse_arguments()
            assert exc_info.value.code == 1

    @patch(
        "model_hosting_container_standards.supervisor.scripts.standard_supervisor.parse_environment_variables"
    )
    @patch(
        "model_hosting_container_standards.supervisor.scripts.standard_supervisor.write_supervisord_config"
    )
    def test_run_success_flow(self, mock_write_config, mock_parse_env):
        """Test successful run flow."""
        # Mock configuration
        mock_config = Mock()
        mock_config.config_path = "/tmp/test.conf"
        mock_parse_env.return_value = mock_config

        # Mock process manager
        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, 0]  # Running, then exit
        mock_process.wait.return_value = 0

        supervisor = StandardSupervisor()
        supervisor.process_manager.check_tools_available = Mock(return_value=(True, ""))
        supervisor.process_manager.start = Mock(return_value=mock_process)
        supervisor.signal_handler.setup = Mock()

        # Mock monitor
        with patch(
            "model_hosting_container_standards.supervisor.scripts.standard_supervisor.ProcessMonitor"
        ) as mock_monitor_class:
            mock_monitor = Mock()
            mock_monitor.check_fatal_state.return_value = False
            mock_monitor_class.return_value = mock_monitor

            with patch.object(sys, "argv", ["standard-supervisor", "echo", "test"]):
                with patch("time.sleep"):  # Mock sleep to speed up test
                    result = supervisor.run()

        assert result == 0
        mock_write_config.assert_called_once()
        supervisor.process_manager.start.assert_called_once()

    def test_run_missing_tools(self):
        """Test run with missing supervisor tools."""
        supervisor = StandardSupervisor()
        supervisor.process_manager.check_tools_available = Mock(
            return_value=(False, "supervisord")
        )

        with patch.object(sys, "argv", ["standard-supervisor", "echo", "test"]):
            result = supervisor.run()

        assert result == 1

    @patch(
        "model_hosting_container_standards.supervisor.scripts.standard_supervisor.parse_environment_variables"
    )
    def test_run_configuration_error(self, mock_parse_env):
        """Test run with configuration error."""
        from model_hosting_container_standards.supervisor.models import (
            ConfigurationError,
        )

        mock_parse_env.side_effect = ConfigurationError("Invalid config")

        supervisor = StandardSupervisor()
        supervisor.process_manager.check_tools_available = Mock(return_value=(True, ""))

        with patch.object(sys, "argv", ["standard-supervisor", "echo", "test"]):
            result = supervisor.run()

        assert result == 1

    @patch(
        "model_hosting_container_standards.supervisor.scripts.standard_supervisor.parse_environment_variables"
    )
    @patch(
        "model_hosting_container_standards.supervisor.scripts.standard_supervisor.write_supervisord_config"
    )
    def test_run_fatal_state_detection(self, mock_write_config, mock_parse_env):
        """Test run with FATAL state detection."""
        # Mock configuration
        mock_config = Mock()
        mock_config.config_path = "/tmp/test.conf"
        mock_parse_env.return_value = mock_config

        # Mock process that keeps running
        mock_process = Mock()
        mock_process.poll.return_value = None  # Always running

        supervisor = StandardSupervisor()
        supervisor.process_manager.check_tools_available = Mock(return_value=(True, ""))
        supervisor.process_manager.start = Mock(return_value=mock_process)
        supervisor.process_manager.terminate = Mock()
        supervisor.signal_handler.setup = Mock()

        # Mock monitor that detects FATAL state
        with patch(
            "model_hosting_container_standards.supervisor.scripts.standard_supervisor.ProcessMonitor"
        ) as mock_monitor_class:
            mock_monitor = Mock()
            mock_monitor.check_fatal_state.return_value = True  # FATAL detected
            mock_monitor_class.return_value = mock_monitor

            with patch.object(sys, "argv", ["standard-supervisor", "echo", "test"]):
                with patch("time.sleep"):  # Mock sleep to speed up test
                    result = supervisor.run()

        assert result == 1
        supervisor.process_manager.terminate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
