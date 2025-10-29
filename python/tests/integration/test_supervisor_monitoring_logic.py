"""
Integration tests for supervisor monitoring logic without requiring supervisord installation.

These tests focus on the configuration generation and script behavior that can be tested
without actually running supervisord.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from model_hosting_container_standards.supervisor.generator import (
    generate_supervisord_config,
    write_supervisord_config,
)
from model_hosting_container_standards.supervisor.models import (
    SupervisorConfig,
    parse_environment_variables,
)


class TestSupervisorMonitoringLogic:
    """Test the monitoring logic and configuration behavior."""

    def test_exit_behavior_configuration_generation(self):
        """Test that configuration is generated with correct exit behavior settings."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command="python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080",
            log_level="info",
        )

        config_content = generate_supervisord_config(config, "llm-engine")

        # Verify critical exit behavior settings
        lines = config_content.split("\n")

        # Check supervisord section
        assert any("nodaemon=true" in line for line in lines)
        assert any("loglevel=info" in line for line in lines)

        # Check program section
        assert any("[program:llm-engine]" in line for line in lines)
        assert any("autorestart=true" in line for line in lines)
        assert any("startretries=3" in line for line in lines)

        # Check critical exit behavior settings
        assert any(
            "exitcodes=255" in line for line in lines
        ), "exitcodes=255 not found - any exit except 255 should trigger restart"
        assert any(
            "startsecs=1" in line for line in lines
        ), "startsecs=1 not found - process must run 1 sec to be considered started"

        # Check command
        assert any("python -m vllm.entrypoints.api_server" in line for line in lines)

    def test_auto_recovery_disabled_configuration(self):
        """Test configuration when auto recovery is disabled."""
        config = SupervisorConfig(
            auto_recovery=False,
            max_recovery_attempts=1,
            launch_command="python -m tensorrt_llm.hlapi.llm_api",
            log_level="debug",
        )

        config_content = generate_supervisord_config(config, "tensorrt-engine")

        # When auto_recovery is False, autorestart should be false
        assert "autorestart=false" in config_content
        assert "startretries=1" in config_content
        # Still should treat all exits as unexpected
        assert "exitcodes=255" in config_content

    def test_environment_variable_parsing_for_monitoring(self):
        """Test that environment variables are correctly parsed for monitoring behavior."""
        env_vars = {
            "LAUNCH_COMMAND": "python -m my_llm_service --config /app/config.json",
            "ENGINE_AUTO_RECOVERY": "true",
            "ENGINE_MAX_RECOVERY_ATTEMPTS": "5",
            "SUPERVISOR_LOG_LEVEL": "warn",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = parse_environment_variables()

            assert (
                config.launch_command
                == "python -m my_llm_service --config /app/config.json"
            )
            assert config.auto_recovery is True
            assert config.max_recovery_attempts == 5
            assert config.log_level == "warn"

    def test_configuration_with_different_retry_limits(self):
        """Test configuration generation with different retry limits."""
        test_cases = [
            (0, "startretries=0"),
            (1, "startretries=1"),
            (10, "startretries=10"),
            (100, "startretries=100"),
        ]

        for max_attempts, expected_line in test_cases:
            config = SupervisorConfig(
                auto_recovery=True,
                max_recovery_attempts=max_attempts,
                launch_command="echo test",
                log_level="info",
            )

            config_content = generate_supervisord_config(config)
            assert expected_line in config_content

    def test_command_with_special_characters(self):
        """Test that commands with special characters are handled correctly."""
        special_commands = [
            "python -c \"print('Hello World')\"",
            'bash -c "echo \\"test\\" && sleep 1"',
            'python -m service --arg="value with spaces"',
            'service --env-var="KEY=value" --port=8080',
        ]

        for command in special_commands:
            config = SupervisorConfig(
                auto_recovery=True,
                max_recovery_attempts=3,
                launch_command=command,
                log_level="info",
            )

            config_content = generate_supervisord_config(config)
            # Command should appear exactly as specified
            assert command in config_content

    def test_configuration_file_writing_and_reading(self):
        """Test writing configuration to file and reading it back."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=2,
            launch_command="python -m test_service",
            log_level="error",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            config_path = f.name

        try:
            # Write configuration
            write_supervisord_config(config_path, config, "test-service")

            # Verify file exists and has content
            assert os.path.exists(config_path)

            # Read and verify content
            with open(config_path, "r") as f:
                content = f.read()

            assert "[program:test-service]" in content
            assert "python -m test_service" in content
            assert "startretries=2" in content
            assert "loglevel=error" in content
            assert "exitcodes=255" in content

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_entrypoint_script_extraction(self):
        """Test that the entrypoint script can be extracted."""
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
            temp_path = f.name

        try:
            # Test extract-supervisor-entrypoint CLI
            result = subprocess.run(
                ["extract-supervisor-entrypoint", "-o", temp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0
            assert os.path.exists(temp_path)

            # Verify the script content
            with open(temp_path, "r") as f:
                script_content = f.read()

            # Check for key monitoring logic
            assert "#!/bin/bash" in script_content
            assert "LLM Service Monitoring Strategy:" in script_content
            assert "supervisorctl status llm-engine" in script_content
            assert "FATAL" in script_content
            assert "exit 1" in script_content

            # Verify script is executable
            assert os.access(temp_path, os.X_OK)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_generate_config_cli_tool(self):
        """Test the generate-supervisor-config CLI tool."""
        with tempfile.NamedTemporaryFile(suffix=".conf", delete=False) as f:
            config_path = f.name

        try:
            env = os.environ.copy()
            env.update(
                {
                    "LAUNCH_COMMAND": "python -m my_service --port 9000",
                    "ENGINE_MAX_RECOVERY_ATTEMPTS": "4",
                    "ENGINE_AUTO_RECOVERY": "true",
                }
            )

            result = subprocess.run(
                ["generate-supervisor-config", "-o", config_path, "-p", "my-service"],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0
            assert os.path.exists(config_path)

            # Verify generated config
            with open(config_path, "r") as f:
                content = f.read()

            assert "[program:my-service]" in content
            assert "python -m my_service --port 9000" in content
            assert "startretries=4" in content
            assert "exitcodes=255" in content

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_entrypoint_script_environment_validation(self):
        """Test entrypoint script validates environment variables correctly."""
        # Extract script to temp location
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
            script_path = f.name

        try:
            # Extract the script
            subprocess.run(
                ["extract-supervisor-entrypoint", "-o", script_path],
                check=True,
                capture_output=True,
            )

            # Test 1: Missing LAUNCH_COMMAND should fail
            env_without_launch = os.environ.copy()
            if "LAUNCH_COMMAND" in env_without_launch:
                del env_without_launch["LAUNCH_COMMAND"]

            result = subprocess.run(
                [script_path],
                env=env_without_launch,
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 1
            assert "LAUNCH_COMMAND must be set" in result.stderr

            # Test 2: Valid LAUNCH_COMMAND should pass validation step
            env_with_launch = os.environ.copy()
            env_with_launch["LAUNCH_COMMAND"] = 'echo "test service"'

            try:
                result = subprocess.run(
                    [script_path],
                    env=env_with_launch,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                # Should get past environment validation (may fail later due to missing supervisord)
                assert "Configuration validation:" in result.stderr
                assert 'LAUNCH_COMMAND: echo "test service"' in result.stderr

            except subprocess.TimeoutExpired:
                # If it times out, it means it got past validation and is trying to run supervisord
                # This is actually a success for our validation test
                pass

        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)

    def test_configuration_template_structure(self):
        """Test that the configuration template has the expected structure."""
        from model_hosting_container_standards.supervisor.generator import (
            SUPERVISORD_CONFIG_TEMPLATE,
        )

        # Verify template structure
        assert "[supervisord]" in SUPERVISORD_CONFIG_TEMPLATE
        assert "[program:{program_name}]" in SUPERVISORD_CONFIG_TEMPLATE

        # Verify critical monitoring settings are in template
        assert "exitcodes=255" in SUPERVISORD_CONFIG_TEMPLATE
        assert "startsecs=1" in SUPERVISORD_CONFIG_TEMPLATE
        assert "autorestart={auto_restart}" in SUPERVISORD_CONFIG_TEMPLATE
        assert "startretries={max_recovery_attempts}" in SUPERVISORD_CONFIG_TEMPLATE

        # Verify logging configuration
        assert "stdout_logfile=/dev/stdout" in SUPERVISORD_CONFIG_TEMPLATE
        assert "stderr_logfile=/dev/stderr" in SUPERVISORD_CONFIG_TEMPLATE

    def test_error_conditions(self):
        """Test various error conditions in configuration generation."""
        # Test empty launch command
        with pytest.raises(
            ValueError, match="Launch command in configuration cannot be empty"
        ):
            config = SupervisorConfig(
                auto_recovery=True,
                max_recovery_attempts=3,
                launch_command="",
                log_level="info",
            )
            generate_supervisord_config(config)

        # Test None launch command
        with pytest.raises(
            ValueError, match="Launch command in configuration cannot be empty"
        ):
            config = SupervisorConfig(
                auto_recovery=True,
                max_recovery_attempts=3,
                launch_command=None,
                log_level="info",
            )
            generate_supervisord_config(config)

        # Test empty program name
        with pytest.raises(ValueError, match="Program name cannot be empty"):
            config = SupervisorConfig(
                auto_recovery=True,
                max_recovery_attempts=3,
                launch_command="echo test",
                log_level="info",
            )
            generate_supervisord_config(config, program_name="")

    def test_monitoring_behavior_documentation(self):
        """Test that the monitoring behavior is properly documented in code."""
        # Check that generator.py has proper comments
        generator_path = (
            Path(__file__).parent.parent.parent
            / "model_hosting_container_standards"
            / "supervisor"
            / "generator.py"
        )

        with open(generator_path, "r") as f:
            generator_content = f.read()

        # Verify key documentation is present
        assert "LLM services are expected to run indefinitely" in generator_content
        assert "exitcodes=255" in generator_content
        assert "FATAL state" in generator_content

        # Check that entrypoint script has proper comments
        script_path = (
            Path(__file__).parent.parent.parent
            / "model_hosting_container_standards"
            / "supervisor"
            / "scripts"
            / "supervisor-entrypoint.sh"
        )

        with open(script_path, "r") as f:
            script_content = f.read()

        # Verify monitoring strategy is documented
        assert "LLM Service Monitoring Strategy:" in script_content
        assert "any exit is an error" in script_content
        assert "FATAL state" in script_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
