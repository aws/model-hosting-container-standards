"""
Unit tests for supervisor configuration generator.

These tests focus on the configuration generation logic
without requiring actual file I/O or supervisor processes.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from model_hosting_container_standards.supervisor.generator import (
    _dict_to_ini_string,
    _merge_custom_sections,
    generate_supervisord_config,
    get_base_config_template,
    write_supervisord_config,
)
from model_hosting_container_standards.supervisor.models import (
    ConfigurationError,
    SupervisorConfig,
)


class TestGetBaseConfigTemplate:
    """Test the base configuration template generation."""

    def test_basic_template_structure(self):
        """Test that basic template has all required sections."""
        template = get_base_config_template(
            program_name="test_program",
            log_level="info",
            framework_command="echo test",
            auto_restart="true",
            max_start_retries=3,
        )

        # Check all required sections exist
        expected_sections = [
            "unix_http_server",
            "supervisorctl",
            "supervisord",
            "rpcinterface:supervisor",
            "program:test_program",
        ]

        for section in expected_sections:
            assert section in template

    def test_program_section_configuration(self):
        """Test program section has correct configuration."""
        template = get_base_config_template(
            program_name="llm_engine",
            log_level="debug",
            framework_command="vllm serve model",
            auto_restart="false",
            max_start_retries=5,
        )

        program_section = template["program:llm_engine"]

        assert program_section["command"] == "vllm serve model"
        assert program_section["autorestart"] == "false"
        assert program_section["startretries"] == "5"
        assert program_section["exitcodes"] == "255"
        assert program_section["startsecs"] == "1"
        assert program_section["stdout_logfile"] == "/dev/stdout"
        assert program_section["stderr_logfile"] == "/dev/stderr"

    def test_supervisord_section_configuration(self):
        """Test supervisord section has correct configuration."""
        template = get_base_config_template(
            program_name="test_program",
            log_level="debug",
            framework_command="echo test",
            auto_restart="true",
            max_start_retries=3,
        )

        supervisord_section = template["supervisord"]

        assert supervisord_section["nodaemon"] == "true"
        assert supervisord_section["loglevel"] == "debug"
        assert "test_program" in supervisord_section["logfile"]
        assert "test_program" in supervisord_section["pidfile"]


class TestMergeCustomSections:
    """Test custom configuration section merging."""

    def test_merge_empty_custom_sections(self):
        """Test merging with empty custom sections."""
        base_config = {"program:test": {"command": "echo test", "autorestart": "true"}}
        custom_sections = {}

        result = _merge_custom_sections(base_config, custom_sections)

        assert result == base_config

    def test_merge_override_existing_setting(self):
        """Test overriding existing settings in base config."""
        base_config = {
            "program:test": {
                "command": "echo test",
                "autorestart": "true",
                "startsecs": "1",
            }
        }
        custom_sections = {"program:test": {"startsecs": "10", "stopwaitsecs": "30"}}

        result = _merge_custom_sections(base_config, custom_sections)

        expected = {
            "program:test": {
                "command": "echo test",
                "autorestart": "true",
                "startsecs": "10",  # Overridden
                "stopwaitsecs": "30",  # Added
            }
        }
        assert result == expected

    def test_merge_add_new_section(self):
        """Test adding completely new sections."""
        base_config = {"program:test": {"command": "echo test"}}
        custom_sections = {
            "eventlistener:memmon": {
                "command": "memmon -a 200MB",
                "events": "PROCESS_STATE_FATAL",
            }
        }

        result = _merge_custom_sections(base_config, custom_sections)

        assert "program:test" in result
        assert "eventlistener:memmon" in result
        assert result["eventlistener:memmon"]["command"] == "memmon -a 200MB"

    def test_merge_preserves_original(self):
        """Test that merging doesn't modify the original base config."""
        base_config = {"program:test": {"command": "echo test", "autorestart": "true"}}
        original_base = base_config.copy()

        custom_sections = {"program:test": {"startsecs": "10"}}

        _merge_custom_sections(base_config, custom_sections)

        # Original should be unchanged
        assert base_config == original_base


class TestDictToIniString:
    """Test INI string generation from dictionary."""

    def test_simple_config(self):
        """Test simple configuration conversion."""
        config_dict = {
            "section1": {"key1": "value1", "key2": "value2"},
            "section2": {"key3": "value3"},
        }

        result = _dict_to_ini_string(config_dict)

        assert "[section1]" in result
        assert "key1=value1" in result
        assert "key2=value2" in result
        assert "[section2]" in result
        assert "key3=value3" in result

    def test_empty_config(self):
        """Test empty configuration conversion."""
        config_dict = {}
        result = _dict_to_ini_string(config_dict)
        assert result == ""

    def test_section_ordering(self):
        """Test that sections are properly separated."""
        config_dict = {"section1": {"key1": "value1"}, "section2": {"key2": "value2"}}

        result = _dict_to_ini_string(config_dict)
        lines = result.split("\n")

        # Should have empty lines between sections
        section1_idx = lines.index("[section1]")

        # There should be an empty line after section1's content
        assert lines[section1_idx + 2] == ""


class TestGenerateSupervisordConfig:
    """Test the main configuration generation function."""

    def test_basic_generation(self):
        """Test basic configuration generation."""
        config = SupervisorConfig(
            auto_recovery=True, max_start_retries=3, log_level="info"
        )

        result = generate_supervisord_config(config, "echo test", "test_program")

        assert "[program:test_program]" in result
        assert "command=echo test" in result
        assert "autorestart=true" in result
        assert "startretries=3" in result

    def test_auto_recovery_disabled(self):
        """Test configuration with auto recovery disabled."""
        config = SupervisorConfig(
            auto_recovery=False, max_start_retries=1, log_level="debug"
        )

        result = generate_supervisord_config(config, "python script.py", "my_program")

        assert "autorestart=false" in result
        assert "startretries=1" in result
        assert "loglevel=debug" in result

    def test_custom_sections_integration(self):
        """Test integration with custom sections."""
        custom_sections = {
            "program:llm_engine": {"startsecs": "15", "stopwaitsecs": "45"},
            "supervisord": {"logfile_maxbytes": "100MB"},
        }

        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=5,
            log_level="info",
            custom_sections=custom_sections,
        )

        result = generate_supervisord_config(config, "vllm serve model", "llm_engine")

        assert "startsecs=15" in result
        assert "stopwaitsecs=45" in result
        assert "logfile_maxbytes=100MB" in result
        assert "startretries=5" in result

    def test_empty_launch_command_error(self):
        """Test error handling for empty launch command."""
        config = SupervisorConfig()

        with pytest.raises(ValueError, match="Launch command cannot be empty"):
            generate_supervisord_config(config, "", "test_program")

        with pytest.raises(ValueError, match="Launch command cannot be empty"):
            generate_supervisord_config(config, "   ", "test_program")

    def test_empty_program_name_error(self):
        """Test error handling for empty program name."""
        config = SupervisorConfig()

        with pytest.raises(ValueError, match="Program name cannot be empty"):
            generate_supervisord_config(config, "echo test", "")

        with pytest.raises(ValueError, match="Program name cannot be empty"):
            generate_supervisord_config(config, "echo test", "   ")

    def test_special_characters_in_command(self):
        """Test handling of special characters in launch command."""
        config = SupervisorConfig()

        command_with_quotes = "python -c \"print('Hello World')\""
        result = generate_supervisord_config(
            config, command_with_quotes, "test_program"
        )

        assert command_with_quotes in result

    @patch(
        "model_hosting_container_standards.supervisor.generator.get_base_config_template"
    )
    def test_exception_handling(self, mock_get_template):
        """Test exception handling in configuration generation."""
        mock_get_template.side_effect = Exception("Template error")

        config = SupervisorConfig()

        with pytest.raises(
            ConfigurationError, match="Failed to generate supervisord configuration"
        ):
            generate_supervisord_config(config, "echo test", "test_program")


class TestWriteSupervisordConfig:
    """Test configuration file writing."""

    def test_successful_write(self):
        """Test successful configuration file writing."""
        config = SupervisorConfig(
            auto_recovery=True, max_start_retries=2, log_level="info"
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name

        try:
            write_supervisord_config(temp_path, config, "echo test", "test_program")

            # Verify file was created and has content
            content = Path(temp_path).read_text()
            assert "[program:test_program]" in content
            assert "command=echo test" in content
            assert "startretries=2" in content

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_directory_creation(self):
        """Test that parent directories are created if they don't exist."""
        config = SupervisorConfig()

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "dir" / "config.conf"

            write_supervisord_config(
                str(nested_path), config, "echo test", "test_program"
            )

            assert nested_path.exists()
            content = nested_path.read_text()
            assert "[program:test_program]" in content

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_write_permission_error(self, mock_open):
        """Test handling of file write permission errors."""
        config = SupervisorConfig()

        with pytest.raises(OSError, match="Failed to write configuration file"):
            write_supervisord_config(
                "/invalid/path/config.conf", config, "echo test", "test_program"
            )

    def test_invalid_launch_command_propagation(self):
        """Test that validation errors are properly propagated."""
        config = SupervisorConfig()

        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(
                ConfigurationError, match="Launch command cannot be empty"
            ):
                write_supervisord_config(f.name, config, "", "test_program")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
