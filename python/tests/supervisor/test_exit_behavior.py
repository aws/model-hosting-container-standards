"""
Unit tests specifically for the SupervisorConfig model and configuration parsing.

These tests focus on the configuration model without testing the generator
which will be updated in a separate task.
"""

import os

import pytest

from model_hosting_container_standards.supervisor.models import (
    SupervisorConfig,
    parse_environment_variables,
)


class TestSupervisorConfigModel:
    """Test the SupervisorConfig model and environment parsing."""

    def test_supervisor_config_creation(self):
        """Test that SupervisorConfig can be created with default values."""
        config = SupervisorConfig()

        assert config.auto_recovery is True
        assert config.max_start_retries == 3
        assert config.recovery_backoff_seconds == 10
        assert config.config_path == "/tmp/supervisord.conf"
        assert config.log_level == "info"
        assert config.custom_sections == {}

    def test_supervisor_config_with_custom_values(self):
        """Test SupervisorConfig creation with custom values."""
        config = SupervisorConfig(
            auto_recovery=False,
            max_start_retries=5,
            log_level="debug",
            custom_sections={"program": {"startsecs": "10"}},
        )

        assert config.auto_recovery is False
        assert config.max_start_retries == 5
        assert config.log_level == "debug"
        assert config.custom_sections == {"program": {"startsecs": "10"}}

    def test_parse_environment_variables_defaults(self):
        """Test parsing environment variables with defaults."""
        # Clear any existing SUPERVISOR_ environment variables that might affect the test
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("SUPERVISOR_"):
                env_backup[key] = os.environ.pop(key)

        try:
            config = parse_environment_variables()

            assert config.auto_recovery is True
            assert config.max_start_retries == 3
            assert config.log_level == "info"
            assert config.custom_sections == {}
        finally:
            # Restore environment
            os.environ.update(env_backup)

    def test_parse_environment_variables_custom(self):
        """Test parsing custom environment variables with simple design."""
        # Set test environment variables
        test_env = {
            "AUTO_RECOVERY": "false",
            "MAX_START_RETRIES": "5",
            "LOG_LEVEL": "debug",
            "SUPERVISOR_PROGRAM_STARTSECS": "10",
            "SUPERVISOR_PROGRAM_STOPWAITSECS": "30",
            "SUPERVISOR_SUPERVISORD_LOGLEVEL": "info",
        }

        # Backup existing environment
        env_backup = {}
        for key in test_env:
            if key in os.environ:
                env_backup[key] = os.environ[key]

        try:
            # Set test environment
            os.environ.update(test_env)

            config = parse_environment_variables()

            assert config.auto_recovery is False
            assert config.max_start_retries == 5
            assert config.log_level == "debug"

            # Check custom sections
            expected_custom = {
                "program": {"startsecs": "10", "stopwaitsecs": "30"},
                "supervisord": {"loglevel": "info"},
            }
            assert config.custom_sections == expected_custom

        finally:
            # Clean up test environment
            for key in test_env:
                if key in env_backup:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)

    def test_custom_sections_parsing(self):
        """Test parsing of SUPERVISOR_{SECTION}_{KEY} environment variables including colon sections."""
        test_env = {
            "SUPERVISOR_PROGRAM_AUTORESTART": "true",
            "SUPERVISOR_PROGRAM_STARTRETRIES": "5",
            "SUPERVISOR_SUPERVISORD_NODAEMON": "true",
            "SUPERVISOR_PROGRAM__WEB_COMMAND": "gunicorn app:app",
            "SUPERVISOR_RPCINTERFACE__SUPERVISOR_FACTORY": "supervisor.rpcinterface:make_main_rpcinterface",
        }

        # Backup and set environment
        env_backup = {}
        for key in test_env:
            if key in os.environ:
                env_backup[key] = os.environ[key]

        try:
            os.environ.update(test_env)

            config = parse_environment_variables()

            # Verify custom sections are parsed correctly
            assert config.custom_sections == {
                "program": {"autorestart": "true", "startretries": "5"},
                "supervisord": {"nodaemon": "true"},
                "program:web": {"command": "gunicorn app:app"},
                "rpcinterface:supervisor": {
                    "factory": "supervisor.rpcinterface:make_main_rpcinterface"
                },
            }

            # Check that we have the expected sections
            assert "program" in config.custom_sections
            assert "supervisord" in config.custom_sections
            assert "program:web" in config.custom_sections
            assert "rpcinterface:supervisor" in config.custom_sections

            assert config.custom_sections["program"]["autorestart"] == "true"
            assert config.custom_sections["program"]["startretries"] == "5"
            assert config.custom_sections["supervisord"]["nodaemon"] == "true"
            assert (
                config.custom_sections["program:web"]["command"] == "gunicorn app:app"
            )
            assert (
                config.custom_sections["rpcinterface:supervisor"]["factory"]
                == "supervisor.rpcinterface:make_main_rpcinterface"
            )

        finally:
            # Clean up
            for key in test_env:
                if key in env_backup:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)

    def test_double_underscore_to_colon_conversion(self):
        """Test that double underscores in section names are converted to colons."""
        test_env = {
            "SUPERVISOR_PROGRAM__WEB_COMMAND": "gunicorn app:app",
            "SUPERVISOR_PROGRAM__API_DIRECTORY": "/app/api",
            "SUPERVISOR_RPCINTERFACE__SUPERVISOR_FACTORY": "supervisor.rpcinterface:make_main_rpcinterface",
            "SUPERVISOR_EVENTLISTENER__MEMMON_COMMAND": "memmon",
        }

        # Backup and set environment
        env_backup = {}
        for key in test_env:
            if key in os.environ:
                env_backup[key] = os.environ[key]

        try:
            os.environ.update(test_env)

            config = parse_environment_variables()

            # Verify double underscores are converted to colons
            assert "program:web" in config.custom_sections
            assert "program:api" in config.custom_sections
            assert "rpcinterface:supervisor" in config.custom_sections
            assert "eventlistener:memmon" in config.custom_sections

            assert (
                config.custom_sections["program:web"]["command"] == "gunicorn app:app"
            )
            assert config.custom_sections["program:api"]["directory"] == "/app/api"
            assert (
                config.custom_sections["rpcinterface:supervisor"]["factory"]
                == "supervisor.rpcinterface:make_main_rpcinterface"
            )
            assert config.custom_sections["eventlistener:memmon"]["command"] == "memmon"

        finally:
            # Clean up
            for key in test_env:
                if key in env_backup:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
