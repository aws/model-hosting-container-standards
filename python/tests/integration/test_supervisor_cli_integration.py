"""
Integration tests for standard-supervisor CLI functionality.

Tests verify:
1. CLI argument parsing and validation
2. Supervisor configuration generation with custom SUPERVISOR_* variables
3. End-to-end CLI execution with simple test commands
"""

import os
import subprocess

import pytest

from model_hosting_container_standards.supervisor.models import (
    parse_environment_variables,
)
from model_hosting_container_standards.supervisor.scripts.standard_supervisor import (
    StandardSupervisor,
)


class TestStandardSupervisorCLI:
    """Test CLI argument parsing and validation."""

    def test_cli_argument_parsing_valid_command(self):
        """Test CLI parsing with valid command arguments."""
        supervisor = StandardSupervisor()

        # Mock sys.argv for testing
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["standard-supervisor", "echo", "hello", "world"]
            launch_command = supervisor.parse_arguments()
            assert launch_command == ["echo", "hello", "world"]
        finally:
            sys.argv = original_argv

    def test_cli_argument_parsing_single_command(self):
        """Test CLI parsing with single command."""
        supervisor = StandardSupervisor()

        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["standard-supervisor", "python", "--version"]
            launch_command = supervisor.parse_arguments()
            assert launch_command == ["python", "--version"]
        finally:
            sys.argv = original_argv

    def test_cli_argument_parsing_complex_command(self):
        """Test CLI parsing with complex command including flags and arguments."""
        supervisor = StandardSupervisor()

        import sys

        original_argv = sys.argv
        try:
            sys.argv = [
                "standard-supervisor",
                "vllm",
                "serve",
                "model",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--dtype",
                "auto",
            ]
            launch_command = supervisor.parse_arguments()
            expected = [
                "vllm",
                "serve",
                "model",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--dtype",
                "auto",
            ]
            assert launch_command == expected
        finally:
            sys.argv = original_argv

    def test_cli_argument_parsing_no_command_error(self):
        """Test CLI parsing fails appropriately when no command is provided."""
        supervisor = StandardSupervisor()

        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["standard-supervisor"]
            with pytest.raises(SystemExit) as exc_info:
                supervisor.parse_arguments()
            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv

    def test_cli_command_line_interface(self):
        """Test the actual CLI command interface."""
        # Test with no arguments - should fail
        result = subprocess.run(
            [
                "python",
                "-m",
                "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            cwd="python",  # Run from python directory where the package is
        )

        assert result.returncode == 1
        assert "No launch command provided" in result.stderr
        assert "Usage: standard-supervisor" in result.stderr


class TestSupervisorConfigurationGeneration:
    """Test supervisor configuration generation with custom SUPERVISOR_* variables."""

    def test_configuration_with_custom_supervisor_variables(self):
        """Test configuration generation with custom SUPERVISOR_* environment variables."""
        # Set up test environment variables
        test_env = {
            "SUPERVISOR_PROGRAM_STARTRETRIES": "5",
            "SUPERVISOR_PROGRAM_STARTSECS": "10",
            "SUPERVISOR_PROGRAM_STOPWAITSECS": "30",
            "SUPERVISOR_SUPERVISORD_LOGLEVEL": "debug",
        }

        # Backup existing environment
        env_backup = {}
        for key in test_env:
            if key in os.environ:
                env_backup[key] = os.environ[key]

        try:
            # Set test environment
            os.environ.update(test_env)

            # Parse configuration
            config = parse_environment_variables()

            # Verify custom sections are parsed correctly
            assert config.custom_sections["program"]["startretries"] == "5"
            assert config.custom_sections["program"]["startsecs"] == "10"
            assert config.custom_sections["program"]["stopwaitsecs"] == "30"
            assert config.custom_sections["supervisord"]["loglevel"] == "debug"

        finally:
            # Clean up test environment
            for key in test_env:
                if key in env_backup:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)

    def test_configuration_with_default_values(self):
        """Test configuration generation with default values when no custom variables are set."""
        # Clear any existing SUPERVISOR_ environment variables
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("SUPERVISOR_"):
                env_backup[key] = os.environ.pop(key)

        try:
            config = parse_environment_variables()

            # Verify defaults
            assert config.auto_recovery is True
            assert config.max_start_retries == 3
            assert config.log_level == "info"
            assert config.custom_sections == {}

        finally:
            # Restore environment
            os.environ.update(env_backup)

    def test_configuration_with_mixed_variables(self):
        """Test configuration with both application-level and SUPERVISOR_* variables."""
        test_env = {
            "AUTO_RECOVERY": "false",
            "MAX_START_RETRIES": "7",
            "LOG_LEVEL": "debug",
            "SUPERVISOR_PROGRAM_STARTSECS": "15",
            "SUPERVISOR_SUPERVISORD_NODAEMON": "true",
        }

        # Backup and set environment
        env_backup = {}
        for key in test_env:
            if key in os.environ:
                env_backup[key] = os.environ[key]

        try:
            os.environ.update(test_env)

            config = parse_environment_variables()

            # Verify application-level variables work
            assert config.auto_recovery is False
            assert config.max_start_retries == 7
            assert config.log_level == "debug"

            # Verify SUPERVISOR_* variables work
            assert config.custom_sections["program"]["startsecs"] == "15"
            assert config.custom_sections["supervisord"]["nodaemon"] == "true"

        finally:
            # Clean up
            for key in test_env:
                if key in env_backup:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)

    def test_configuration_with_program_specific_variables(self):
        """Test configuration with program-specific SUPERVISOR_PROGRAM__LLM_ENGINE_* variables."""
        test_env = {
            "SUPERVISOR_PROGRAM__LLM_ENGINE_STARTSECS": "20",
            "SUPERVISOR_PROGRAM__LLM_ENGINE_STOPWAITSECS": "45",
            "SUPERVISOR_PROGRAM_STARTSECS": "10",  # Generic program setting
        }

        # Backup and set environment
        env_backup = {}
        for key in test_env:
            if key in os.environ:
                env_backup[key] = os.environ[key]

        try:
            os.environ.update(test_env)

            config = parse_environment_variables()

            # Verify program-specific variables work (double underscore becomes colon)
            # LLM_ENGINE becomes llm_engine in the section name
            assert config.custom_sections["program:llm_engine"]["startsecs"] == "20"
            assert config.custom_sections["program:llm_engine"]["stopwaitsecs"] == "45"

            # Verify generic program variables work
            assert config.custom_sections["program"]["startsecs"] == "10"

        finally:
            # Clean up
            for key in test_env:
                if key in env_backup:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)


class TestEndToEndCLIExecution:
    """Test end-to-end CLI execution with simple test commands."""

    @pytest.fixture
    def clean_environment(self):
        """Provide a clean environment for testing."""
        # Backup environment variables that might affect tests
        env_backup = {}
        supervisor_keys = [
            key for key in os.environ.keys() if key.startswith("SUPERVISOR_")
        ]
        app_level_keys = ["AUTO_RECOVERY", "MAX_START_RETRIES", "LOG_LEVEL"]

        for key in supervisor_keys + app_level_keys:
            if key in os.environ:
                env_backup[key] = os.environ[key]
                del os.environ[key]

        yield

        # Restore environment
        os.environ.update(env_backup)

    def test_cli_execution_with_simple_command(self, clean_environment):
        """Test CLI execution with a simple command that exits quickly."""
        # Set up minimal configuration for quick execution
        os.environ["MAX_START_RETRIES"] = "1"
        os.environ["SUPERVISOR_PROGRAM__LLM_ENGINE_STARTSECS"] = "1"

        # Use a command that will exit quickly
        result = subprocess.run(
            [
                "python",
                "-m",
                "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                "echo",
                "test message",
            ],
            capture_output=True,
            text=True,
            timeout=15,  # Allow time for supervisor setup and execution
            cwd="python",  # Run from python directory where the package is
        )

        # The command should execute and supervisor should handle the exit
        # Since echo exits immediately, supervisor will detect this and exit
        assert result.returncode in [
            0,
            1,
        ]  # 0 for success, 1 for expected exit after command completion

        # Verify supervisor started and processed the command
        assert (
            "Starting: echo test message" in result.stderr
            or "Starting: echo test message" in result.stdout
        )

    def test_cli_execution_with_python_command(self, clean_environment):
        """Test CLI execution with a Python command."""
        # Set up configuration for quick execution
        os.environ["MAX_START_RETRIES"] = "1"
        os.environ["SUPERVISOR_PROGRAM__LLM_ENGINE_STARTSECS"] = "1"

        result = subprocess.run(
            [
                "python",
                "-m",
                "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                "python",
                "-c",
                "print('Hello from supervised process'); import time; time.sleep(0.5)",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd="python",  # Run from python directory where the package is
        )

        # Should execute successfully
        assert result.returncode in [0, 1]

        # Verify supervisor started
        assert (
            "Starting: python -c" in result.stderr
            or "Starting: python -c" in result.stdout
        )

    def test_cli_execution_with_custom_configuration(self, clean_environment):
        """Test CLI execution with custom SUPERVISOR_* configuration."""
        # Set custom configuration (using recommended approach)
        os.environ["MAX_START_RETRIES"] = "2"
        os.environ["LOG_LEVEL"] = "debug"
        os.environ["SUPERVISOR_PROGRAM__LLM_ENGINE_STARTSECS"] = "2"

        result = subprocess.run(
            [
                "python",
                "-m",
                "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                "python",
                "-c",
                "print('Custom config test')",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd="python",  # Run from python directory where the package is
        )

        # Should execute with custom configuration
        assert result.returncode in [0, 1]

        # Verify supervisor started with custom config
        assert (
            "Starting: python -c" in result.stderr
            or "Starting: python -c" in result.stdout
        )

    def test_cli_execution_missing_supervisor_tools(
        self, clean_environment, monkeypatch
    ):
        """Test CLI execution when supervisor tools are missing."""

        # Mock shutil.which to simulate missing supervisord
        def mock_which(cmd):
            if cmd == "supervisord":
                return None
            return "/usr/bin/" + cmd  # Return path for other commands

        monkeypatch.setattr("shutil.which", mock_which)

        result = subprocess.run(
            [
                "python",
                "-c",
                "import sys; sys.path.insert(0, 'python'); "
                "from model_hosting_container_standards.supervisor.scripts.standard_supervisor import main; "
                "sys.argv = ['standard-supervisor', 'echo', 'test']; "
                "exit(main())",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 1

    def test_cli_execution_configuration_error(self, clean_environment):
        """Test CLI execution with invalid configuration."""
        # Set invalid configuration that should cause an error
        os.environ["MAX_START_RETRIES"] = "invalid_number"

        result = subprocess.run(
            [
                "python",
                "-m",
                "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                "echo",
                "test",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="python",  # Run from python directory where the package is
        )

        # Should fail due to configuration error
        assert result.returncode == 1

    def test_cli_execution_with_failing_command(self, clean_environment):
        """Test CLI execution with a command that fails immediately."""
        # Set up configuration for quick failure detection
        os.environ["MAX_START_RETRIES"] = "1"
        os.environ["SUPERVISOR_PROGRAM__LLM_ENGINE_STARTSECS"] = "1"

        result = subprocess.run(
            [
                "python",
                "-m",
                "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                "python",
                "-c",
                "import sys; sys.exit(1)",  # Command that fails immediately
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd="python",  # Run from python directory where the package is
        )

        # Should handle the failing command appropriately
        assert result.returncode == 1

        # Verify supervisor started and detected the failure
        assert (
            "Starting: python -c" in result.stderr
            or "Starting: python -c" in result.stdout
        )


class TestCLIIntegrationWithRealFrameworks:
    """Test CLI integration patterns that would be used with real ML frameworks."""

    def test_vllm_command_pattern(self):
        """Test CLI with vLLM-style command pattern (without actually running vLLM)."""
        supervisor = StandardSupervisor()

        import sys

        original_argv = sys.argv
        try:
            # Simulate typical vLLM command
            sys.argv = [
                "standard-supervisor",
                "vllm",
                "serve",
                "microsoft/DialoGPT-medium",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--dtype",
                "auto",
                "--max-model-len",
                "2048",
            ]

            launch_command = supervisor.parse_arguments()
            expected = [
                "vllm",
                "serve",
                "microsoft/DialoGPT-medium",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--dtype",
                "auto",
                "--max-model-len",
                "2048",
            ]
            assert launch_command == expected
        finally:
            sys.argv = original_argv

    def test_tensorrt_llm_command_pattern(self):
        """Test CLI with TensorRT-LLM-style command pattern."""
        supervisor = StandardSupervisor()

        import sys

        original_argv = sys.argv
        try:
            # Simulate typical TensorRT-LLM command
            sys.argv = [
                "standard-supervisor",
                "python",
                "-m",
                "tensorrt_llm.hlapi.llm_api",
                "--model-dir",
                "/opt/model",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
            ]

            launch_command = supervisor.parse_arguments()
            expected = [
                "python",
                "-m",
                "tensorrt_llm.hlapi.llm_api",
                "--model-dir",
                "/opt/model",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
            ]
            assert launch_command == expected
        finally:
            sys.argv = original_argv

    def test_custom_script_pattern(self):
        """Test CLI with custom script pattern."""
        supervisor = StandardSupervisor()

        import sys

        original_argv = sys.argv
        try:
            # Simulate custom script execution
            sys.argv = [
                "standard-supervisor",
                "./my-model-server.sh",
                "--config",
                "/app/config.json",
                "--workers",
                "4",
            ]

            launch_command = supervisor.parse_arguments()
            expected = [
                "./my-model-server.sh",
                "--config",
                "/app/config.json",
                "--workers",
                "4",
            ]
            assert launch_command == expected
        finally:
            sys.argv = original_argv

    def test_fastapi_uvicorn_pattern(self):
        """Test CLI with FastAPI/Uvicorn command pattern."""
        supervisor = StandardSupervisor()

        import sys

        original_argv = sys.argv
        try:
            # Simulate FastAPI with Uvicorn
            sys.argv = [
                "standard-supervisor",
                "uvicorn",
                "app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--workers",
                "1",
            ]

            launch_command = supervisor.parse_arguments()
            expected = [
                "uvicorn",
                "app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--workers",
                "1",
            ]
            assert launch_command == expected
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
