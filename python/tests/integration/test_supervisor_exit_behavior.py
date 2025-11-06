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
            log_level="info",
        )

        write_supervisord_config(
            temp_config_file, config, "echo 'test command'", "test-program"
        )
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
            log_level="debug",
        )

        write_supervisord_config(
            temp_config_file, config, "python -c 'print(\"hello\")'", "llm-engine"
        )
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
            get_base_config_template,
        )

        # Generate a sample template to verify structure
        template = get_base_config_template(
            program_name="test-program",
            log_level="info",
            framework_command="echo test",
            auto_restart="true",
            max_start_retries=3,
        )

        # Verify expected sections exist
        expected_sections = [
            "supervisord",
            "program:test-program",
            "unix_http_server",
            "supervisorctl",
            "rpcinterface:supervisor",
        ]

        for section in expected_sections:
            assert section in template

        # Verify critical settings in program section
        program_section = template["program:test-program"]
        assert program_section["exitcodes"] == "255"
        assert program_section["startsecs"] == "1"
        assert program_section["command"] == "echo test"

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
            log_level="info",
        )

        with pytest.raises(ValueError, match="Launch command cannot be empty"):
            generate_supervisord_config(config, invalid_command)

    def test_empty_program_name_error(self):
        """Test that empty program name raises error."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            log_level="info",
        )

        with pytest.raises(ValueError, match="Program name cannot be empty"):
            generate_supervisord_config(config, "echo test", program_name="")

    def test_special_configurations(self):
        """Test edge case configurations."""
        # Zero retries
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=0,
            log_level="info",
        )
        content = generate_supervisord_config(config, "echo test")
        assert "startretries=0" in content

        # Special characters in command
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            log_level="info",
        )
        content = generate_supervisord_config(
            config, 'python -c "print(\'Hello, World!\')" && echo "Done"'
        )
        assert 'python -c "print(\'Hello, World!\')" && echo "Done"' in content


class TestCustomConfigurationMerging:
    """Test custom SUPERVISOR_* configuration merging functionality."""

    def test_custom_configuration_merging_basic(self):
        """Test basic custom configuration merging."""
        custom_sections = {
            "program:llm-engine": {
                "startsecs": "10",
                "stopwaitsecs": "30",
            },
            "supervisord": {
                "loglevel": "debug",
            },
        }

        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            log_level="info",
            custom_sections=custom_sections,
        )

        content = generate_supervisord_config(config, "echo test", "llm-engine")

        # Verify custom settings are applied
        assert "startsecs=10" in content
        assert "stopwaitsecs=30" in content
        assert "loglevel=debug" in content

    def test_custom_configuration_new_section(self):
        """Test adding completely new sections via custom configuration."""
        custom_sections = {
            "eventlistener:memmon": {
                "command": "memmon -a 200MB -m mail@example.com",
                "events": "PROCESS_STATE_FATAL",
            }
        }

        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            log_level="info",
            custom_sections=custom_sections,
        )

        content = generate_supervisord_config(config, "echo test", "llm-engine")

        # Verify new section is added
        assert "[eventlistener:memmon]" in content
        assert "command=memmon -a 200MB -m mail@example.com" in content
        assert "events=PROCESS_STATE_FATAL" in content

    def test_custom_configuration_override_any_setting(self):
        """Test that any setting can be overridden (user responsibility)."""
        # Test overriding any settings - user is responsible for correctness
        custom_sections = {
            "program:llm-engine": {
                "command": "custom command",
                "exitcodes": "0",
                "nodaemon": "false",
            },
            "supervisord": {
                "nodaemon": "false",
            },
        }

        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            log_level="info",
            custom_sections=custom_sections,
        )

        # Should work without validation errors - user responsibility
        content = generate_supervisord_config(config, "echo test", "llm-engine")

        # Verify overrides are applied
        assert "command=custom command" in content
        assert "exitcodes=0" in content
        assert "nodaemon=false" in content

    def test_custom_configuration_empty_sections(self):
        """Test behavior with empty custom sections."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            log_level="info",
            custom_sections={},
        )

        content = generate_supervisord_config(config, "echo test", "llm-engine")

        # Should work normally without custom sections
        assert "[program:llm-engine]" in content
        assert "command=echo test" in content

    def test_custom_configuration_override_existing_settings(self):
        """Test overriding existing non-critical settings."""
        custom_sections = {
            "program:llm-engine": {
                "startsecs": "5",  # Override default startsecs=1
                "priority": "999",  # Add new setting
            }
        }

        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            log_level="info",
            custom_sections=custom_sections,
        )

        content = generate_supervisord_config(config, "echo test", "llm-engine")

        # Verify override worked
        assert "startsecs=5" in content
        assert "startsecs=1" not in content  # Original should be replaced
        assert "priority=999" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
