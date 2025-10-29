"""Unit tests for supervisor configuration module."""

import os
from unittest.mock import patch

import pytest

from model_hosting_container_standards.supervisor.config import (
    FrameworkName,
    SupervisorConfig,
    get_framework_name,
    parse_environment_variables,
    validate_config_directory,
    validate_environment_variable,
)


class TestFrameworkName:
    """Test FrameworkName enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert FrameworkName.VLLM.value == "vllm"
        assert FrameworkName.TENSORRT_LLM.value == "tensorrt-llm"

    def test_enum_count(self):
        """Test that enum has exactly 2 values."""
        assert len(FrameworkName) == 2


class TestSupervisorConfig:
    """Test SupervisorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SupervisorConfig()

        assert config.auto_recovery is True
        assert config.max_recovery_attempts == 3
        assert config.recovery_backoff_seconds == 10
        assert config.framework_command is None
        assert config.config_path == "/opt/aws/supervisor/conf.d/supervisord.conf"
        assert config.log_level == "info"
        assert config.framework_name is None


class TestValidateEnvironmentVariable:
    """Test validate_environment_variable helper function."""

    @pytest.mark.parametrize(
        "value,var_type,expected",
        [
            ("5", int, True),
            ("0", int, True),
            ("100", int, True),
            ("true", bool, True),
            ("false", bool, True),
            ("1", bool, True),
            ("0", bool, True),
            ("yes", bool, True),
            ("no", bool, True),
            ("on", bool, True),
            ("off", bool, True),
            ("valid_string", str, True),
        ],
    )
    def test_valid_values(self, value, var_type, expected):
        """Test validation of valid values."""
        is_valid, error_msg = validate_environment_variable("TEST_VAR", value, var_type)
        assert is_valid == expected
        assert error_msg is None

    @pytest.mark.parametrize(
        "value,var_type",
        [
            ("not_a_number", int),
            ("1.5", int),
            ("invalid_bool", bool),
            ("", str),
            ("   ", str),
        ],
    )
    def test_invalid_values(self, value, var_type):
        """Test validation of invalid values."""
        is_valid, error_msg = validate_environment_variable("TEST_VAR", value, var_type)
        assert is_valid is False
        assert error_msg is not None
        assert "TEST_VAR" in error_msg

    def test_integer_range_validation(self):
        """Test integer range validation."""
        # Valid range
        is_valid, error_msg = validate_environment_variable(
            "TEST_VAR", "5", int, min_value=0, max_value=10
        )
        assert is_valid is True
        assert error_msg is None

        # Below minimum
        is_valid, error_msg = validate_environment_variable(
            "TEST_VAR", "-1", int, min_value=0
        )
        assert is_valid is False
        assert "must be >= 0" in error_msg

        # Above maximum
        is_valid, error_msg = validate_environment_variable(
            "TEST_VAR", "15", int, max_value=10
        )
        assert is_valid is False
        assert "must be <= 10" in error_msg

    def test_string_allowed_values_validation(self):
        """Test string allowed values validation."""
        allowed_values = ["debug", "info", "warn", "error"]

        # Valid value
        is_valid, error_msg = validate_environment_variable(
            "LOG_LEVEL", "debug", str, allowed_values=allowed_values
        )
        assert is_valid is True
        assert error_msg is None

        # Invalid value
        is_valid, error_msg = validate_environment_variable(
            "LOG_LEVEL", "invalid", str, allowed_values=allowed_values
        )
        assert is_valid is False
        assert "must be one of" in error_msg


class TestValidateConfigDirectory:
    """Test validate_config_directory function."""

    def test_valid_directory(self):
        """Test validation of valid directory."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")
            is_valid, error_msg = validate_config_directory(config_path)
            assert is_valid is True
            assert error_msg is None

    def test_creates_missing_directory(self):
        """Test that missing directories are created."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "dir", "supervisord.conf")
            is_valid, error_msg = validate_config_directory(nested_path)
            assert is_valid is True
            assert error_msg is None
            assert os.path.exists(os.path.dirname(nested_path))


class TestParseEnvironmentVariables:
    """Test parse_environment_variables function."""

    def test_default_configuration(self):
        """Test parsing with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = parse_environment_variables()

            assert config.auto_recovery is True
            assert config.max_recovery_attempts == 3
            assert config.recovery_backoff_seconds == 10
            assert config.framework_command is None
            assert config.config_path == "/opt/aws/supervisor/conf.d/supervisord.conf"
            assert config.log_level == "info"
            assert config.framework_name is None

    def test_all_environment_variables_set(self):
        """Test parsing with all environment variables set."""
        env_vars = {
            "ENGINE_AUTO_RECOVERY": "false",
            "ENGINE_MAX_RECOVERY_ATTEMPTS": "5",
            "ENGINE_RECOVERY_BACKOFF_SECONDS": "30",
            "FRAMEWORK_COMMAND": "python -m vllm.entrypoints.api_server",
            "SUPERVISOR_CONFIG_PATH": "/custom/path/supervisord.conf",
            "SUPERVISOR_LOG_LEVEL": "debug",
            "FRAMEWORK_NAME": "vllm",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = parse_environment_variables()

            assert config.auto_recovery is False
            assert config.max_recovery_attempts == 5
            assert config.recovery_backoff_seconds == 30
            assert config.framework_command == "python -m vllm.entrypoints.api_server"
            assert config.config_path == "/custom/path/supervisord.conf"
            assert config.log_level == "debug"
            assert config.framework_name == FrameworkName.VLLM

    def test_partial_environment_variables(self):
        """Test parsing with only some environment variables set."""
        env_vars = {
            "ENGINE_AUTO_RECOVERY": "false",
            "FRAMEWORK_NAME": "tensorrt-llm",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = parse_environment_variables()

            # Changed values
            assert config.auto_recovery is False
            assert config.framework_name == FrameworkName.TENSORRT_LLM

            # Default values
            assert config.max_recovery_attempts == 3
            assert config.recovery_backoff_seconds == 10
            assert config.framework_command is None
            assert config.config_path == "/opt/aws/supervisor/conf.d/supervisord.conf"
            assert config.log_level == "info"

    def test_string_trimming(self):
        """Test that string values are properly trimmed."""
        env_vars = {
            "FRAMEWORK_COMMAND": "  python -m vllm  ",
            "SUPERVISOR_CONFIG_PATH": "  /path/to/config  ",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = parse_environment_variables()

            assert config.framework_command == "python -m vllm"
            assert config.config_path == "/path/to/config"

    def test_invalid_values_use_defaults_with_warnings(self):
        """Test that invalid values use defaults and log warnings."""
        env_vars = {
            "ENGINE_AUTO_RECOVERY": "invalid_bool",
            "ENGINE_MAX_RECOVERY_ATTEMPTS": "invalid_int",
            "SUPERVISOR_LOG_LEVEL": "invalid_level",
            "FRAMEWORK_NAME": "invalid_framework",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Should not raise exception, but use defaults
            config = parse_environment_variables()

            # Check that defaults are used
            assert config.auto_recovery is True  # default
            assert config.max_recovery_attempts == 3  # default
            assert config.log_level == "info"  # default
            assert config.framework_name is None  # default


class TestGetFrameworkName:
    """Test get_framework_name function."""

    def test_default_framework_name(self):
        """Test default framework name when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_framework_name()
            assert result is None

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("vllm", FrameworkName.VLLM),
            ("tensorrt-llm", FrameworkName.TENSORRT_LLM),
        ],
    )
    def test_valid_framework_names(self, value, expected):
        """Test parsing of valid framework names."""
        with patch.dict(os.environ, {"FRAMEWORK_NAME": value}):
            result = get_framework_name()
            assert result == expected

    def test_invalid_framework_name_returns_none(self):
        """Test that invalid framework names return None."""
        with patch.dict(os.environ, {"FRAMEWORK_NAME": "invalid"}):
            result = get_framework_name()
            assert result is None


class TestSupervisorConfigGeneration:
    """Test supervisor_config module functions."""

    def test_generate_supervisord_config_basic(self):
        """Test basic supervisord configuration generation."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        config = generate_supervisord_config("python app.py")

        assert "[supervisord]" in config
        assert "[program:framework]" in config
        assert "command=python app.py" in config
        assert "autostart=true" in config
        assert "autorestart=true" in config
        assert "startretries=3" in config

    def test_generate_supervisord_config_with_custom_program_name(self):
        """Test configuration generation with custom program name."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        config = generate_supervisord_config("python app.py", program_name="my-service")

        assert "[program:my-service]" in config
        assert "command=python app.py" in config

    def test_generate_supervisord_config_with_custom_config(self):
        """Test configuration generation with custom SupervisorConfig."""
        from model_hosting_container_standards.supervisor.config import SupervisorConfig
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        custom_config = SupervisorConfig(
            auto_recovery=False, max_recovery_attempts=5, log_level="debug"
        )

        config = generate_supervisord_config("python app.py", custom_config)

        assert "autorestart=false" in config
        assert "startretries=5" in config
        assert "loglevel=debug" in config

    def test_write_supervisord_config(self):
        """Test writing configuration to file."""
        import os
        import tempfile

        from model_hosting_container_standards.supervisor.supervisor_config import (
            write_supervisord_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")

            write_supervisord_config(config_path, "python app.py")

            assert os.path.exists(config_path)

            with open(config_path, "r") as f:
                content = f.read()
                assert "[supervisord]" in content
                assert "command=python app.py" in content

    def test_write_supervisord_config_creates_directories(self):
        """Test that write_supervisord_config creates parent directories."""
        import os
        import tempfile

        from model_hosting_container_standards.supervisor.supervisor_config import (
            write_supervisord_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "nested", "dir", "supervisord.conf")

            write_supervisord_config(config_path, "python app.py")

            assert os.path.exists(config_path)


class TestFrameworkConfig:
    """Test framework_config module functions."""

    def test_get_framework_command_with_explicit_command(self):
        """Test getting framework command from FRAMEWORK_COMMAND env var."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        with patch.dict(os.environ, {"FRAMEWORK_COMMAND": "custom command"}):
            result = get_framework_command()
            assert result == "custom command"

    def test_get_framework_command_without_command_returns_none(self):
        """Test getting framework command when no FRAMEWORK_COMMAND is set."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        with patch.dict(os.environ, {"FRAMEWORK_NAME": "vllm"}, clear=True):
            result = get_framework_command()
            assert result is None

    def test_get_framework_command_no_framework(self):
        """Test getting framework command when no framework is specified."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        with patch.dict(os.environ, {}, clear=True):
            result = get_framework_command()
            assert result is None

    def test_get_framework_command_explicit_overrides_framework(self):
        """Test that explicit FRAMEWORK_COMMAND overrides framework defaults."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        env_vars = {"FRAMEWORK_COMMAND": "explicit command", "FRAMEWORK_NAME": "vllm"}

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_framework_command()
            assert result == "explicit command"

    def test_get_framework_command_strips_whitespace(self):
        """Test that framework command is stripped of whitespace."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        with patch.dict(os.environ, {"FRAMEWORK_COMMAND": "  python app.py  "}):
            result = get_framework_command()
            assert result == "python app.py"

    @pytest.mark.parametrize(
        "command,expected",
        [
            ("python app.py", True),
            ("python -m vllm.entrypoints.api_server", True),
            ("/usr/bin/python3 script.py", True),
            ("", False),
            ("   ", False),
        ],
    )
    def test_validate_framework_command(self, command, expected):
        """Test framework command validation."""
        from model_hosting_container_standards.supervisor.framework_config import (
            validate_framework_command,
        )

        result = validate_framework_command(command)
        assert result == expected


class TestSupervisorConfigModule:
    """Test supervisor_config module functions."""

    def test_generate_supervisord_config_basic(self):
        """Test basic supervisord configuration generation."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        config = generate_supervisord_config("python app.py")

        assert "[supervisord]" in config
        assert "[program:framework]" in config
        assert "command=python app.py" in config
        assert "autostart=true" in config
        assert "autorestart=true" in config
        assert "startretries=3" in config

    def test_generate_supervisord_config_with_custom_program_name(self):
        """Test configuration generation with custom program name."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        config = generate_supervisord_config("python app.py", program_name="my-service")

        assert "[program:my-service]" in config
        assert "command=python app.py" in config

    def test_generate_supervisord_config_with_custom_config(self):
        """Test configuration generation with custom SupervisorConfig."""
        from model_hosting_container_standards.supervisor.config import SupervisorConfig
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        custom_config = SupervisorConfig(
            auto_recovery=False, max_recovery_attempts=5, log_level="debug"
        )

        config = generate_supervisord_config("python app.py", custom_config)

        assert "autorestart=false" in config
        assert "startretries=5" in config
        assert "loglevel=debug" in config

    def test_generate_supervisord_config_empty_command_raises_error(self):
        """Test that empty framework command raises ValueError."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        with pytest.raises(ValueError, match="Framework command cannot be empty"):
            generate_supervisord_config("")

        with pytest.raises(ValueError, match="Framework command cannot be empty"):
            generate_supervisord_config("   ")

    def test_generate_supervisord_config_empty_program_name_raises_error(self):
        """Test that empty program name raises ValueError."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        with pytest.raises(ValueError, match="Program name cannot be empty"):
            generate_supervisord_config("python app.py", program_name="")

        with pytest.raises(ValueError, match="Program name cannot be empty"):
            generate_supervisord_config("python app.py", program_name="   ")

    def test_write_supervisord_config(self):
        """Test writing configuration to file."""
        import os
        import tempfile

        from model_hosting_container_standards.supervisor.supervisor_config import (
            write_supervisord_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")

            write_supervisord_config(config_path, "python app.py")

            assert os.path.exists(config_path)

            with open(config_path, "r") as f:
                content = f.read()
                assert "[supervisord]" in content
                assert "command=python app.py" in content

    def test_write_supervisord_config_creates_directories(self):
        """Test that write_supervisord_config creates parent directories."""
        import os
        import tempfile

        from model_hosting_container_standards.supervisor.supervisor_config import (
            write_supervisord_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "nested", "dir", "supervisord.conf")

            write_supervisord_config(config_path, "python app.py")

            assert os.path.exists(config_path)

    def test_write_supervisord_config_empty_path_raises_error(self):
        """Test that empty config path raises ValueError."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            write_supervisord_config,
        )

        with pytest.raises(ValueError, match="Configuration path cannot be empty"):
            write_supervisord_config("", "python app.py")

        with pytest.raises(ValueError, match="Configuration path cannot be empty"):
            write_supervisord_config("   ", "python app.py")


class TestFrameworkConfigModule:
    """Test framework_config module functions."""

    def test_get_framework_command_with_explicit_command(self):
        """Test getting framework command from FRAMEWORK_COMMAND env var."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        with patch.dict(os.environ, {"FRAMEWORK_COMMAND": "custom command"}):
            result = get_framework_command()
            assert result == "custom command"

    def test_get_framework_command_without_command_returns_none(self):
        """Test getting framework command when no FRAMEWORK_COMMAND is set."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        with patch.dict(os.environ, {"FRAMEWORK_NAME": "vllm"}, clear=True):
            result = get_framework_command()
            assert result is None

    def test_get_framework_command_no_framework(self):
        """Test getting framework command when no framework is specified."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        with patch.dict(os.environ, {}, clear=True):
            result = get_framework_command()
            assert result is None

    def test_get_framework_command_explicit_overrides_framework(self):
        """Test that explicit FRAMEWORK_COMMAND overrides framework defaults."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        env_vars = {"FRAMEWORK_COMMAND": "explicit command", "FRAMEWORK_NAME": "vllm"}

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_framework_command()
            assert result == "explicit command"

    def test_get_framework_command_strips_whitespace(self):
        """Test that framework command is stripped of whitespace."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        with patch.dict(os.environ, {"FRAMEWORK_COMMAND": "  python app.py  "}):
            result = get_framework_command()
            assert result == "python app.py"

    def test_get_framework_command_empty_explicit_command(self):
        """Test that empty FRAMEWORK_COMMAND falls back to framework detection."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        env_vars = {"FRAMEWORK_COMMAND": "   ", "FRAMEWORK_NAME": "vllm"}

        with patch.dict(os.environ, env_vars, clear=True):
            result = get_framework_command()
            assert result is None

    @pytest.mark.parametrize(
        "command,expected",
        [
            ("python app.py", True),
            ("python -m vllm.entrypoints.api_server", True),
            ("/usr/bin/python3 script.py", True),
            ("./run_server.sh", True),
            ("java -jar app.jar", True),
            ("node server.js", True),
            ("bash start.sh", True),
            ("", False),
            ("   ", False),
        ],
    )
    def test_validate_framework_command(self, command, expected):
        """Test framework command validation."""
        from model_hosting_container_standards.supervisor.framework_config import (
            validate_framework_command,
        )

        result = validate_framework_command(command)
        assert result == expected

    def test_get_supported_frameworks(self):
        """Test getting supported frameworks mapping."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_supported_frameworks,
        )

        frameworks = get_supported_frameworks()

        assert isinstance(frameworks, set)
        assert "vllm" in frameworks
        assert "tensorrt-llm" in frameworks


class TestIntegration:
    """Test integration between supervisor modules."""

    def test_end_to_end_config_generation(self):
        """Test complete configuration generation workflow."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        env_vars = {
            "FRAMEWORK_COMMAND": "python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8080",
            "FRAMEWORK_NAME": "vllm",
            "ENGINE_AUTO_RECOVERY": "false",
            "ENGINE_MAX_RECOVERY_ATTEMPTS": "5",
            "SUPERVISOR_LOG_LEVEL": "debug",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            framework_command = get_framework_command()
            assert framework_command is not None

            config = generate_supervisord_config(framework_command)

            # Check framework command is included
            assert "python -m vllm.entrypoints.api_server" in config

            # Check custom settings are applied
            assert "autorestart=false" in config
            assert "startretries=5" in config
            assert "loglevel=debug" in config

    def test_config_generation_with_explicit_command(self):
        """Test configuration generation with explicit framework command."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        env_vars = {
            "FRAMEWORK_COMMAND": "python my_custom_server.py --port 9000",
            "ENGINE_AUTO_RECOVERY": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            framework_command = get_framework_command()
            config = generate_supervisord_config(
                framework_command, program_name="custom-server"
            )

            assert "[program:custom-server]" in config
            assert "command=python my_custom_server.py --port 9000" in config
            assert "autorestart=true" in config
