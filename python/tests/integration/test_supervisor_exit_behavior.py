"""
Integration tests for supervisor exit behavior and monitoring logic.

Tests verify:
1. Configuration generation with correct restart behavior
2. Entrypoint script validation and execution
3. CLI tools functionality
"""

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
            temp_config_file, config, "python -c 'print(\"hello\")'", "llm_engine"
        )
        content = Path(temp_config_file).read_text()

        assert "autorestart=false" in content
        assert "startretries=1" in content
        assert "exitcodes=255" in content

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
        # Test generate-supervisor-config via Python module
        result = subprocess.run(
            [
                "python",
                "-m",
                "model_hosting_container_standards.supervisor.scripts.generate_supervisor_config",
                "-o",
                temp_config_file,
                "-p",
                "test-service",
                "echo",
                "test",
                "command",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="python",
        )

        assert result.returncode == 0
        content = Path(temp_config_file).read_text()
        assert "[program:test-service]" in content
        assert "echo test command" in content


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
            "program:llm_engine": {
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

        content = generate_supervisord_config(config, "echo test", "llm_engine")

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

        content = generate_supervisord_config(config, "echo test", "llm_engine")

        # Verify new section is added
        assert "[eventlistener:memmon]" in content
        assert "command=memmon -a 200MB -m mail@example.com" in content
        assert "events=PROCESS_STATE_FATAL" in content

    def test_custom_configuration_override_any_setting(self):
        """Test that any setting can be overridden (user responsibility)."""
        # Test overriding any settings - user is responsible for correctness
        custom_sections = {
            "program:llm_engine": {
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
        content = generate_supervisord_config(config, "echo test", "llm_engine")

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

        content = generate_supervisord_config(config, "echo test", "llm_engine")

        # Should work normally without custom sections
        assert "[program:llm_engine]" in content
        assert "command=echo test" in content

    def test_custom_configuration_override_existing_settings(self):
        """Test overriding existing non-critical settings."""
        custom_sections = {
            "program:llm_engine": {
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

        content = generate_supervisord_config(config, "echo test", "llm_engine")

        # Verify override worked
        assert "startsecs=5" in content
        assert "startsecs=1" not in content  # Original should be replaced
        assert "priority=999" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
