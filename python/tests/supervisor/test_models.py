"""
Unit tests for supervisor models module.

Tests configuration parsing, validation functions, and error handling.
"""

import os
from unittest.mock import patch

import pytest

from model_hosting_container_standards.supervisor.models import (
    ConfigurationError,
    SupervisorConfig,
    _get_env_int,
    _get_env_str,
    _parse_bool,
    _parse_supervisor_custom_sections,
    parse_environment_variables,
)


class TestSupervisorConfig:
    """Test the SupervisorConfig dataclass."""

    def test_default_values(self):
        """Test SupervisorConfig with default values."""
        config = SupervisorConfig()

        assert config.enable_supervisor is False
        assert config.auto_recovery is True
        assert config.max_start_retries == 3
        assert config.config_path == "/tmp/supervisord.conf"
        assert config.log_level == "info"
        assert config.custom_sections == {}

    def test_custom_values(self):
        """Test SupervisorConfig with custom values."""
        custom_sections = {"program": {"startsecs": "10"}}
        config = SupervisorConfig(
            enable_supervisor=True,
            auto_recovery=False,
            max_start_retries=5,
            config_path="/custom/path.conf",
            log_level="debug",
            custom_sections=custom_sections,
        )

        assert config.enable_supervisor is True
        assert config.auto_recovery is False
        assert config.max_start_retries == 5
        assert config.config_path == "/custom/path.conf"
        assert config.log_level == "debug"
        assert config.custom_sections == custom_sections


class TestParseBool:
    """Test the _parse_bool helper function."""

    def test_true_values(self):
        """Test values that should parse to True."""
        true_values = ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]
        for value in true_values:
            assert _parse_bool(value) is True

    def test_false_values(self):
        """Test values that should parse to False."""
        false_values = ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF", ""]
        for value in false_values:
            assert _parse_bool(value) is False

    def test_mixed_case(self):
        """Test mixed case values."""
        assert _parse_bool("TrUe") is True
        assert _parse_bool("FaLsE") is False
        assert _parse_bool("YeS") is True
        assert _parse_bool("nO") is False


class TestGetEnvInt:
    """Test the _get_env_int helper function."""

    def test_default_value(self):
        """Test returning default when env var not set."""
        result = _get_env_int("NONEXISTENT_VAR", 42)
        assert result == 42

    def test_valid_integer(self):
        """Test parsing valid integer from environment."""
        with patch.dict(os.environ, {"TEST_INT": "25"}):
            result = _get_env_int("TEST_INT", 10)
            assert result == 25

    def test_boundary_values(self):
        """Test boundary validation."""
        with patch.dict(os.environ, {"TEST_INT": "5"}):
            result = _get_env_int("TEST_INT", 10, min_val=0, max_val=10)
            assert result == 5

    def test_invalid_integer(self):
        """Test error on invalid integer."""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            with pytest.raises(ConfigurationError, match="must be an integer"):
                _get_env_int("TEST_INT", 10)

    def test_below_minimum(self):
        """Test error when value below minimum."""
        with patch.dict(os.environ, {"TEST_INT": "-5"}):
            with pytest.raises(ConfigurationError, match="must be between 0 and 100"):
                _get_env_int("TEST_INT", 10, min_val=0, max_val=100)

    def test_above_maximum(self):
        """Test error when value above maximum."""
        with patch.dict(os.environ, {"TEST_INT": "150"}):
            with pytest.raises(ConfigurationError, match="must be between 0 and 100"):
                _get_env_int("TEST_INT", 10, min_val=0, max_val=100)

    def test_empty_string(self):
        """Test empty string returns default."""
        with patch.dict(os.environ, {"TEST_INT": ""}):
            result = _get_env_int("TEST_INT", 42)
            assert result == 42

    def test_whitespace_only(self):
        """Test whitespace-only string raises error."""
        with patch.dict(os.environ, {"TEST_INT": "   "}):
            with pytest.raises(ConfigurationError, match="must be an integer"):
                _get_env_int("TEST_INT", 42)


class TestGetEnvStr:
    """Test the _get_env_str helper function."""

    def test_default_value(self):
        """Test returning default when env var not set."""
        result = _get_env_str("NONEXISTENT_VAR", "default")
        assert result == "default"

    def test_valid_string(self):
        """Test getting valid string from environment."""
        with patch.dict(os.environ, {"TEST_STR": "test_value"}):
            result = _get_env_str("TEST_STR", "default")
            assert result == "test_value"

    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed."""
        with patch.dict(os.environ, {"TEST_STR": "  test_value  "}):
            result = _get_env_str("TEST_STR", "default")
            assert result == "test_value"

    def test_allowed_values_valid(self):
        """Test validation with allowed values - valid case."""
        with patch.dict(os.environ, {"TEST_STR": "debug"}):
            result = _get_env_str("TEST_STR", "info", allowed=["debug", "info", "warn"])
            assert result == "debug"

    def test_allowed_values_case_insensitive(self):
        """Test validation with allowed values is case insensitive."""
        with patch.dict(os.environ, {"TEST_STR": "DEBUG"}):
            result = _get_env_str("TEST_STR", "info", allowed=["debug", "info", "warn"])
            assert result == "DEBUG"

    def test_allowed_values_invalid(self):
        """Test error when value not in allowed list."""
        with patch.dict(os.environ, {"TEST_STR": "invalid"}):
            with pytest.raises(ConfigurationError, match="must be one of"):
                _get_env_str("TEST_STR", "info", allowed=["debug", "info", "warn"])

    def test_empty_string_with_allowed(self):
        """Test empty string with allowed values raises error."""
        with patch.dict(os.environ, {"TEST_STR": ""}):
            with pytest.raises(ConfigurationError, match="must be one of"):
                _get_env_str("TEST_STR", "info", allowed=["debug", "info", "warn"])


class TestParseSupervisorCustomSections:
    """Test the _parse_supervisor_custom_sections helper function."""

    def test_empty_environment(self):
        """Test with no SUPERVISOR_ environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            result = _parse_supervisor_custom_sections()
            assert result == {}

    def test_skip_config_path(self):
        """Test that SUPERVISOR_CONFIG_PATH is skipped."""
        test_env = {"SUPERVISOR_CONFIG_PATH": "/tmp/test.conf"}
        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()
            assert result == {}

    def test_basic_sections(self):
        """Test parsing basic section configurations."""
        test_env = {
            "SUPERVISOR_PROGRAM_STARTSECS": "10",
            "SUPERVISOR_SUPERVISORD_LOGLEVEL": "debug",
        }
        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()

            expected = {
                "program": {"startsecs": "10"},
                "supervisord": {"loglevel": "debug"},
            }
            assert result == expected

    def test_colon_sections(self):
        """Test parsing sections with colons (double underscore conversion)."""
        test_env = {
            "SUPERVISOR_PROGRAM__WEB_COMMAND": "gunicorn app:app",
            "SUPERVISOR_RPCINTERFACE__SUPERVISOR_FACTORY": "supervisor.rpcinterface:make_main_rpcinterface",
        }
        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()

            expected = {
                "program:web": {"command": "gunicorn app:app"},
                "rpcinterface:supervisor": {
                    "factory": "supervisor.rpcinterface:make_main_rpcinterface"
                },
            }
            assert result == expected

    def test_mixed_sections(self):
        """Test parsing mix of basic and colon sections."""
        test_env = {
            "SUPERVISOR_PROGRAM_AUTORESTART": "true",
            "SUPERVISOR_PROGRAM__API_DIRECTORY": "/app/api",
            "SUPERVISOR_SUPERVISORD_NODAEMON": "true",
        }
        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()

            expected = {
                "program": {"autorestart": "true"},
                "program:api": {"directory": "/app/api"},
                "supervisord": {"nodaemon": "true"},
            }
            assert result == expected

    def test_case_conversion(self):
        """Test that section names and keys are converted to lowercase."""
        test_env = {
            "SUPERVISOR_PROGRAM_STARTSECS": "10",
            "SUPERVISOR_PROGRAM__WEB_COMMAND": "gunicorn app:app",
        }
        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()

            # Verify all keys are lowercase
            assert "program" in result
            assert "program:web" in result
            assert "startsecs" in result["program"]
            assert "command" in result["program:web"]

    def test_whitespace_trimming(self):
        """Test that values are trimmed of whitespace."""
        test_env = {
            "SUPERVISOR_PROGRAM_COMMAND": "  python app.py  ",
        }
        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()

            assert result["program"]["command"] == "python app.py"

    def test_valid_format_parsing(self):
        """Test that valid format environment variables are parsed correctly."""
        test_env = {
            "SUPERVISOR_PROGRAM_COMMAND": "python app.py",
            "SUPERVISOR_PROGRAM__WEB_DIRECTORY": "/app",
            "SUPERVISOR_SUPERVISORD_LOGLEVEL": "info",
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()

            # Should parse correctly
            expected = {
                "program": {"command": "python app.py"},
                "program:web": {"directory": "/app"},
                "supervisord": {"loglevel": "info"},
            }
            assert result == expected

    def test_invalid_format_ignored(self):
        """Test that invalid format environment variables are ignored."""
        test_env = {
            "SUPERVISOR_": "invalid",  # No section or key
            "SUPERVISOR_PROGRAM": "invalid",  # No key (no underscore)
            "SUPERVISOR_PROGRAM_": "invalid",  # Empty key name
            "SUPERVISOR__WEB_COMMAND": "gunicorn app:app",  # Empty section name
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()

            # All invalid formats should be ignored, result should be empty
            assert result == {}

    def test_leading_underscore_in_section_rejected(self):
        """Test that section names with leading underscores are rejected."""
        test_env = {
            "SUPERVISOR__PROGRAM_COMMAND": "python app.py",  # Leading underscore in section
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()
            assert result == {}

    def test_trailing_underscore_in_section_rejected(self):
        """Test that section names with trailing underscores are rejected."""
        test_env = {
            "SUPERVISOR_PROGRAM__COMMAND": "python app.py",  # Trailing underscore in section (before key)
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()
            assert result == {}

    def test_multiple_consecutive_underscores_rejected(self):
        """Test that three or more consecutive underscores are rejected."""
        test_env = {
            "SUPERVISOR_PROGRAM___WEB_COMMAND": "gunicorn app:app",  # Three underscores
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()
            assert result == {}

    def test_leading_underscore_in_key_rejected(self):
        """Test that key names with leading underscores are rejected."""
        test_env = {
            "SUPERVISOR_PROGRAM__COMMAND": "python app.py",  # Leading underscore in key
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()
            assert result == {}

    def test_trailing_underscore_in_key_rejected(self):
        """Test that key names with trailing underscores are rejected."""
        test_env = {
            "SUPERVISOR_PROGRAM_COMMAND_": "python app.py",  # Trailing underscore in key
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()
            assert result == {}

    def test_numeric_only_sections_and_keys_accepted(self):
        """Test that purely numeric section and key names are accepted."""
        test_env = {
            "SUPERVISOR_123_456": "value",  # Numeric section and key
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()

            expected = {
                "123": {"456": "value"},
            }
            assert result == expected

    def test_mixed_alphanumeric_accepted(self):
        """Test that mixed alphanumeric section and key names are accepted."""
        test_env = {
            "SUPERVISOR_PROGRAM2_COMMAND3": "python app.py",
            "SUPERVISOR_WEB1__API2_PORT8080": "8080",
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = _parse_supervisor_custom_sections()

            expected = {
                "program2": {"command3": "python app.py"},
                "web1:api2": {"port8080": "8080"},
            }
            assert result == expected


class TestParseEnvironmentVariables:
    """Test the main parse_environment_variables function."""

    def test_defaults(self):
        """Test parsing with default values."""
        # Clear supervisor-related env vars
        supervisor_vars = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                (
                    "ENABLE_SUPERVISOR",
                    "PROCESS_AUTO_RECOVERY",
                    "PROCESS_MAX_START_RETRIES",
                    "LOG_LEVEL",
                    "SUPERVISOR_",
                )
            )
        }

        with patch.dict(os.environ, {}, clear=False):
            # Remove supervisor vars
            for key in supervisor_vars:
                os.environ.pop(key, None)

            try:
                config = parse_environment_variables()

                assert config.enable_supervisor is False
                assert config.auto_recovery is True
                assert config.max_start_retries == 3
                assert config.config_path == "/tmp/supervisord.conf"
                assert config.log_level == "info"
                assert config.custom_sections == {}
            finally:
                # Restore original env vars
                os.environ.update(supervisor_vars)

    def test_all_custom_values(self):
        """Test parsing with all custom values."""
        test_env = {
            "ENABLE_SUPERVISOR": "true",
            "PROCESS_AUTO_RECOVERY": "false",
            "PROCESS_MAX_START_RETRIES": "5",
            "SUPERVISOR_CONFIG_PATH": "/custom/supervisord.conf",
            "LOG_LEVEL": "debug",
            "SUPERVISOR_PROGRAM_STARTSECS": "10",
            "SUPERVISOR_PROGRAM__WEB_COMMAND": "gunicorn app:app",
        }

        with patch.dict(os.environ, test_env):
            config = parse_environment_variables()

            assert config.enable_supervisor is True
            assert config.auto_recovery is False
            assert config.max_start_retries == 5
            assert config.config_path == "/custom/supervisord.conf"
            assert config.log_level == "debug"

            expected_custom = {
                "program": {"startsecs": "10"},
                "program:web": {"command": "gunicorn app:app"},
            }
            assert config.custom_sections == expected_custom

    def test_invalid_max_start_retries(self):
        """Test error handling for invalid PROCESS_MAX_START_RETRIES."""
        with patch.dict(os.environ, {"PROCESS_MAX_START_RETRIES": "invalid"}):
            with pytest.raises(ConfigurationError, match="must be an integer"):
                parse_environment_variables()

    def test_invalid_log_level(self):
        """Test error handling for invalid LOG_LEVEL."""
        with patch.dict(os.environ, {"LOG_LEVEL": "invalid"}):
            with pytest.raises(ConfigurationError, match="must be one of"):
                parse_environment_variables()

    def test_max_start_retries_out_of_range(self):
        """Test error handling for PROCESS_MAX_START_RETRIES out of range."""
        with patch.dict(os.environ, {"PROCESS_MAX_START_RETRIES": "150"}):
            with pytest.raises(ConfigurationError, match="must be between 0 and 100"):
                parse_environment_variables()

    def test_configuration_error_logging(self):
        """Test that configuration errors are logged."""
        with patch.dict(os.environ, {"PROCESS_MAX_START_RETRIES": "invalid"}):
            with patch(
                "model_hosting_container_standards.supervisor.models.logger"
            ) as mock_logger:
                with pytest.raises(ConfigurationError):
                    parse_environment_variables()

                mock_logger.error.assert_called_once()
                assert (
                    "Configuration validation failed"
                    in mock_logger.error.call_args[0][0]
                )

    def test_boolean_variations(self):
        """Test various boolean value formats for PROCESS_AUTO_RECOVERY."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"PROCESS_AUTO_RECOVERY": env_value}):
                config = parse_environment_variables()
                assert config.auto_recovery is expected

    def test_log_level_case_insensitive(self):
        """Test that LOG_LEVEL validation is case insensitive."""
        test_cases = ["debug", "DEBUG", "Debug", "INFO", "info", "WARN", "warn"]

        for log_level in test_cases:
            with patch.dict(os.environ, {"LOG_LEVEL": log_level}):
                config = parse_environment_variables()
                assert config.log_level == log_level


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
