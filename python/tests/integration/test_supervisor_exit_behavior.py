"""
Integration tests for supervisor exit behavior and monitoring logic.

Tests verify:
1. Configuration generation with correct restart behavior
2. Entrypoint script validation and execution
3. CLI tools functionality
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from model_hosting_container_standards.supervisor.generator import (
    generate_supervisord_config,
    write_supervisord_config,
)
from model_hosting_container_standards.supervisor.models import SupervisorConfig


class TestSupervisorExitBehavior:
    """Test supervisor configuration and behavior."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_entrypoint_script(self):
        """Extract entrypoint script to temporary location for testing."""
        import shutil
        from importlib import resources

        script_path = (
            resources.files("model_hosting_container_standards")
            / "supervisor/scripts/supervisor-entrypoint.sh"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            temp_path = f.name

        shutil.copy2(str(script_path), temp_path)
        os.chmod(temp_path, 0o755)

        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_config_generation_basic(self, temp_config_file):
        """Test basic config generation with correct settings."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=2,
            launch_command="echo 'test command'",
            log_level="info",
        )

        write_supervisord_config(temp_config_file, config, "test-program")
        content = Path(temp_config_file).read_text()

        # Verify key settings
        assert "exitcodes=255" in content
        assert "autorestart=true" in content
        assert "startretries=2" in content
        assert "command=echo 'test command'" in content
        assert "[program:test-program]" in content

    def test_config_generation_auto_recovery_disabled(self, temp_config_file):
        """Test config generation when auto recovery is disabled."""
        config = SupervisorConfig(
            auto_recovery=False,
            max_start_retries=1,
            launch_command="python -c 'print(\"hello\")'",
            log_level="debug",
        )

        write_supervisord_config(temp_config_file, config)
        content = Path(temp_config_file).read_text()

        assert "autorestart=false" in content
        assert "startretries=1" in content
        assert "exitcodes=255" in content

    def test_entrypoint_script_validation(self, temp_entrypoint_script):
        """Test entrypoint script environment validation."""
        # Test without LAUNCH_COMMAND
        env = os.environ.copy()
        env.pop("LAUNCH_COMMAND", None)

        result = subprocess.run(
            [temp_entrypoint_script],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 1
        assert "LAUNCH_COMMAND must be set" in result.stderr

    def test_entrypoint_script_with_valid_environment(self, temp_entrypoint_script):
        """Test entrypoint script passes validation with valid environment."""
        import os
        import signal

        env = os.environ.copy()
        env["LAUNCH_COMMAND"] = 'echo "test service"'

        # Use process group to ensure we can kill the entire process tree
        process = subprocess.Popen(
            [temp_entrypoint_script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # Create new process group
        )

        stdout = ""
        stderr = ""

        try:
            # Give more time for CI environments (they can be slower)
            stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            # Script is running indefinitely (supervisord started) - kill process group
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                stdout, stderr = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                # Still not dead, force kill the entire process group
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate(timeout=3)
        finally:
            # Double insurance: kill any remaining processes
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        # Should pass validation regardless of whether supervisord starts successfully
        assert "Configuration validation:" in stderr
        assert 'LAUNCH_COMMAND: echo "test service"' in stderr

    def test_config_template_structure(self):
        """Test that configuration template has expected structure."""
        from model_hosting_container_standards.supervisor.generator import (
            SUPERVISORD_CONFIG_TEMPLATE,
        )

        # Verify template structure and placeholders
        expected_sections = ["[supervisord]", "[program:{program_name}]"]
        expected_settings = ["exitcodes=255", "startsecs=1"]
        expected_placeholders = [
            "{log_level}",
            "{framework_command}",
            "{auto_restart}",
            "{max_start_retries}",
        ]

        for item in expected_sections + expected_settings + expected_placeholders:
            assert item in SUPERVISORD_CONFIG_TEMPLATE

    def test_cli_tools(self, temp_config_file):
        """Test CLI tools functionality."""
        # Test extract-supervisor-entrypoint
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
            temp_script_path = f.name

        try:
            result = subprocess.run(
                ["extract-supervisor-entrypoint", "-o", temp_script_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0
            assert Path(temp_script_path).exists()
            assert os.access(temp_script_path, os.X_OK)

            content = Path(temp_script_path).read_text()
            assert content.startswith("#!/bin/bash")
            assert "LLM Service Monitoring Strategy:" in content

        finally:
            Path(temp_script_path).unlink(missing_ok=True)

        # Test generate-supervisor-config
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
        content = Path(temp_config_file).read_text()
        assert "[program:test-service]" in content
        assert "python -m test.service --port 8080" in content


class TestSupervisorConfigurationEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.parametrize("invalid_command", ["", "   \t\n   ", None])
    def test_invalid_launch_command_error(self, invalid_command):
        """Test that invalid launch commands raise appropriate errors."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            launch_command=invalid_command,
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
            max_start_retries=3,
            launch_command="echo test",
            log_level="info",
        )

        with pytest.raises(ValueError, match="Program name cannot be empty"):
            generate_supervisord_config(config, program_name="")

    def test_special_configurations(self):
        """Test edge case configurations."""
        # Zero retries
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=0,
            launch_command="echo test",
            log_level="info",
        )
        content = generate_supervisord_config(config)
        assert "startretries=0" in content

        # Special characters in command
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            launch_command='python -c "print(\'Hello, World!\')" && echo "Done"',
            log_level="info",
        )
        content = generate_supervisord_config(config)
        assert 'python -c "print(\'Hello, World!\')" && echo "Done"' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
