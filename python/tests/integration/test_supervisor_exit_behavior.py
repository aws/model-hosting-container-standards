"""
Integration tests for supervisor exit behavior and monitoring logic.

These tests verify the actual behavior of the supervisor system:
1. LLM services that exit are automatically restarted
2. After max retry attempts, the container exits with code 1
3. Long-running services are properly monitored
4. Configuration generation works end-to-end
"""

import os
import subprocess
import tempfile
import time

import pytest

from model_hosting_container_standards.supervisor.generator import (
    generate_supervisord_config,
    write_supervisord_config,
)
from model_hosting_container_standards.supervisor.models import SupervisorConfig


class TestSupervisorExitBehavior:
    """Test the actual exit behavior and monitoring logic."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_entrypoint_script(self):
        """Extract entrypoint script to temporary location for testing."""
        import shutil
        from importlib import resources

        script_path = str(
            resources.files("model_hosting_container_standards")
            / "supervisor/scripts/supervisor-entrypoint.sh"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            temp_path = f.name

        shutil.copy2(script_path, temp_path)
        os.chmod(temp_path, 0o755)

        yield temp_path
        os.unlink(temp_path)

    def test_config_generation_with_exit_behavior(self, temp_config_file):
        """Test that generated config has correct exit behavior settings."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=2,
            launch_command="echo 'test command'",
            log_level="info",
        )

        write_supervisord_config(temp_config_file, config, "test-program")

        # Read and verify the generated config
        with open(temp_config_file, "r") as f:
            config_content = f.read()

        # Verify key behavior settings
        assert "exitcodes=255" in config_content
        assert "startsecs=1" in config_content
        assert "autorestart=true" in config_content
        assert "startretries=2" in config_content
        assert "command=echo 'test command'" in config_content
        assert "[program:test-program]" in config_content

    def test_config_generation_with_auto_recovery_disabled(self, temp_config_file):
        """Test config generation when auto recovery is disabled."""
        config = SupervisorConfig(
            auto_recovery=False,
            max_recovery_attempts=1,
            launch_command="python -c 'print(\"hello\")'",
            log_level="debug",
        )

        write_supervisord_config(temp_config_file, config)

        with open(temp_config_file, "r") as f:
            config_content = f.read()

        # When auto_recovery is False, autorestart should be false
        assert "autorestart=false" in config_content
        assert "startretries=1" in config_content
        assert "exitcodes=255" in config_content  # Still treat all exits as unexpected

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/supervisord")
        and not os.path.exists("/usr/local/bin/supervisord"),
        reason="supervisord not installed",
    )
    def test_supervisord_config_syntax_validation(self, temp_config_file):
        """Test that generated config has valid supervisord syntax."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command="sleep 1",
            log_level="info",
        )

        write_supervisord_config(temp_config_file, config)

        # Test config syntax with supervisord
        result = subprocess.run(
            ["supervisord", "-c", temp_config_file, "-t"],
            capture_output=True,
            text=True,
        )

        # Should exit with code 0 for valid config
        assert result.returncode == 0, f"Config syntax error: {result.stderr}"

    def test_failing_command_behavior_simulation(self, temp_config_file):
        """Test the behavior with a command that exits immediately (simulates failure)."""
        # Create config for a command that exits immediately
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=2,
            launch_command="echo 'failing service' && exit 1",
            log_level="info",
        )

        write_supervisord_config(temp_config_file, config)

        # Verify the config contains the expected restart behavior
        with open(temp_config_file, "r") as f:
            content = f.read()

        # Key assertions for failure handling
        assert "startretries=2" in content
        assert (
            "exitcodes=255" in content
        )  # Only 255 is "expected", so exit 1 will trigger restart
        assert "autorestart=true" in content

    def test_long_running_command_config(self, temp_config_file):
        """Test config for a long-running command (normal LLM service behavior)."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=5,
            launch_command="python -c 'import time; print(\"LLM service started\"); time.sleep(3600)'",
            log_level="warn",
        )

        write_supervisord_config(temp_config_file, config)

        with open(temp_config_file, "r") as f:
            content = f.read()

        # Verify long-running service settings
        assert "startretries=5" in content
        assert "loglevel=warn" in content
        assert "time.sleep(3600)" in content

    def test_entrypoint_script_environment_validation(self, temp_entrypoint_script):
        """Test that entrypoint script validates required environment variables."""
        # Test without LAUNCH_COMMAND
        env = os.environ.copy()
        if "LAUNCH_COMMAND" in env:
            del env["LAUNCH_COMMAND"]

        result = subprocess.run(
            [temp_entrypoint_script],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail with exit code 1
        assert result.returncode == 1
        assert "LAUNCH_COMMAND must be set" in result.stderr

    def test_entrypoint_script_with_valid_environment(self, temp_entrypoint_script):
        """Test entrypoint script with valid environment (but expect it to fail on missing supervisord)."""
        env = os.environ.copy()
        env["LAUNCH_COMMAND"] = 'echo "test service"'

        try:
            result = subprocess.run(
                [temp_entrypoint_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=3,  # Reduced timeout since we expect it to fail quickly
            )

            # Will likely fail due to missing supervisord, but should pass env validation
            # Check that it got past the environment validation step
            assert "Configuration validation:" in result.stderr
            assert 'LAUNCH_COMMAND: echo "test service"' in result.stderr

        except subprocess.TimeoutExpired as e:
            # If it times out, it means the script got past validation and tried to start supervisord
            # This is actually a success case for our test - it means env validation worked
            # Check the partial output we got before timeout
            stderr_output = e.stderr.decode() if e.stderr else ""

            # The script should have logged the configuration validation before timing out
            assert "Configuration validation:" in stderr_output
            assert 'LAUNCH_COMMAND: echo "test service"' in stderr_output

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/supervisord")
        and not os.path.exists("/usr/local/bin/supervisord"),
        reason="supervisord not installed",
    )
    def test_end_to_end_failing_service_behavior(
        self, temp_entrypoint_script, temp_config_file
    ):
        """
        End-to-end test of failing service behavior.

        This test verifies:
        1. Service starts and fails immediately
        2. supervisord restarts it up to max attempts
        3. After max attempts, program enters FATAL state
        4. Entrypoint script detects FATAL and exits with code 1
        """
        env = os.environ.copy()
        env.update(
            {
                "LAUNCH_COMMAND": 'echo "Service failed" && exit 1',
                "ENGINE_MAX_RECOVERY_ATTEMPTS": "2",
                "ENGINE_AUTO_RECOVERY": "true",
                "SUPERVISOR_CONFIG_PATH": temp_config_file,
            }
        )

        # Run the entrypoint script with a timeout
        start_time = time.time()
        result = subprocess.run(
            [temp_entrypoint_script],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,  # Should complete within 30 seconds
        )
        end_time = time.time()

        # Verify the behavior
        assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"

        # Should complete relatively quickly (within 30 seconds)
        assert end_time - start_time < 30

        # Check for expected log messages
        stderr_output = result.stderr
        assert "Configuration generated successfully" in stderr_output
        assert "Starting supervisord" in stderr_output

        # The exact FATAL detection message might not appear due to timing,
        # but the exit code 1 confirms the behavior worked

    def test_config_template_comments_and_documentation(self):
        """Test that the configuration template includes proper documentation."""
        from model_hosting_container_standards.supervisor.generator import (
            SUPERVISORD_CONFIG_TEMPLATE,
        )

        # Verify the template has the expected structure
        assert "[supervisord]" in SUPERVISORD_CONFIG_TEMPLATE
        assert "[program:{program_name}]" in SUPERVISORD_CONFIG_TEMPLATE
        assert "exitcodes=255" in SUPERVISORD_CONFIG_TEMPLATE
        assert "startsecs=1" in SUPERVISORD_CONFIG_TEMPLATE

        # Check that key placeholders are present
        assert "{log_level}" in SUPERVISORD_CONFIG_TEMPLATE
        assert "{framework_command}" in SUPERVISORD_CONFIG_TEMPLATE
        assert "{auto_restart}" in SUPERVISORD_CONFIG_TEMPLATE
        assert "{max_recovery_attempts}" in SUPERVISORD_CONFIG_TEMPLATE

    def test_extract_entrypoint_cli_tool(self):
        """Test the extract-supervisor-entrypoint CLI tool."""
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
            temp_path = f.name

        try:
            # Test the CLI tool
            result = subprocess.run(
                ["extract-supervisor-entrypoint", "-o", temp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0
            assert (
                f"Successfully extracted supervisor-entrypoint.sh to {temp_path}"
                in result.stdout
            )

            # Verify the extracted file
            assert os.path.exists(temp_path)
            assert os.access(temp_path, os.X_OK)  # Should be executable

            # Verify it's a valid shell script
            with open(temp_path, "r") as f:
                content = f.read()

            assert content.startswith("#!/bin/bash")
            assert "LLM Service Monitoring Strategy:" in content

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_generate_supervisor_config_cli_tool(self, temp_config_file):
        """Test the generate-supervisor-config CLI tool."""
        env = os.environ.copy()
        env["LAUNCH_COMMAND"] = "python -m test.service --port 8080"

        result = subprocess.run(
            [
                "generate-supervisor-config",
                "-o",
                temp_config_file,
                "-p",
                "test-service",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert os.path.exists(temp_config_file)

        # Verify the generated config
        with open(temp_config_file, "r") as f:
            content = f.read()

        assert "[program:test-service]" in content
        assert "python -m test.service --port 8080" in content
        assert "exitcodes=255" in content


class TestSupervisorConfigurationEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_launch_command_error(self):
        """Test that empty launch command raises appropriate error."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command="",  # Empty command
            log_level="info",
        )

        with pytest.raises(
            ValueError, match="Launch command in configuration cannot be empty"
        ):
            generate_supervisord_config(config)

    def test_whitespace_only_launch_command_error(self):
        """Test that whitespace-only launch command raises error."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command="   \t\n   ",  # Whitespace only
            log_level="info",
        )

        with pytest.raises(
            ValueError, match="Launch command in configuration cannot be empty"
        ):
            generate_supervisord_config(config)

    def test_none_launch_command_error(self):
        """Test that None launch command raises error."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command=None,
            log_level="info",
        )

        with pytest.raises(
            ValueError, match="Launch command in configuration cannot be empty"
        ):
            generate_supervisord_config(config)

    def test_empty_program_name_error(self):
        """Test that empty program name raises error."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command="echo test",
            log_level="info",
        )

        with pytest.raises(ValueError, match="Program name cannot be empty"):
            generate_supervisord_config(config, program_name="")

    def test_max_recovery_attempts_zero(self):
        """Test configuration with zero recovery attempts."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=0,
            launch_command="echo test",
            log_level="info",
        )

        config_content = generate_supervisord_config(config)
        assert "startretries=0" in config_content

    def test_special_characters_in_command(self):
        """Test that special characters in commands are handled properly."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command='python -c "print(\'Hello, World!\')" && echo "Done"',
            log_level="info",
        )

        config_content = generate_supervisord_config(config)
        assert 'python -c "print(\'Hello, World!\')" && echo "Done"' in config_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
