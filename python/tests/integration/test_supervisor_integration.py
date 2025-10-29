"""Integration tests for supervisor functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSupervisorIntegration:
    """Integration tests for supervisor process management."""

    @property
    def script_path(self):
        """Get path to the generate_supervisor_config.py script."""
        return (
            Path(__file__).parent.parent.parent
            / "model_hosting_container_standards"
            / "supervisor"
            / "scripts"
            / "generate_supervisor_config.py"
        )

    @property
    def entrypoint_script_path(self):
        """Get path to the supervisor-entrypoint.sh script."""
        return (
            Path(__file__).parent.parent.parent
            / "model_hosting_container_standards"
            / "supervisor"
            / "scripts"
            / "supervisor-entrypoint.sh"
        )

    def test_end_to_end_config_generation_and_validation(self):
        """Test complete configuration generation and validation workflow."""
        from model_hosting_container_standards.supervisor.config import (
            parse_environment_variables,
        )
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
            write_supervisord_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")

            # Set up environment for vLLM
            env_vars = {
                "FRAMEWORK_NAME": "vllm",
                "ENGINE_AUTO_RECOVERY": "true",
                "ENGINE_MAX_RECOVERY_ATTEMPTS": "3",
                "ENGINE_RECOVERY_BACKOFF_SECONDS": "5",
                "SUPERVISOR_LOG_LEVEL": "info",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                # Parse configuration
                config = parse_environment_variables()
                assert config.auto_recovery is True
                assert config.max_recovery_attempts == 3
                assert config.recovery_backoff_seconds == 5
                assert config.log_level == "info"

                # Get framework command
                framework_command = get_framework_command()
                assert framework_command is not None
                assert "vllm" in framework_command

                # Generate configuration
                config_content = generate_supervisord_config(framework_command, config)
                assert "[supervisord]" in config_content
                assert "[program:framework]" in config_content
                assert "autorestart=true" in config_content

                # Write configuration to file
                write_supervisord_config(config_path, framework_command, config)
                assert os.path.exists(config_path)

                # Verify file contents
                with open(config_path, "r") as f:
                    file_content = f.read()
                    assert file_content == config_content

    def test_framework_integration_with_environment_variables(self):
        """Test framework integration with various environment variable combinations."""
        from model_hosting_container_standards.supervisor.config import (
            parse_environment_variables,
        )
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        # Test with TensorRT-LLM framework
        env_vars = {
            "FRAMEWORK_NAME": "tensorrt-llm",
            "ENGINE_AUTO_RECOVERY": "false",
            "ENGINE_MAX_RECOVERY_ATTEMPTS": "1",
            "SUPERVISOR_LOG_LEVEL": "debug",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = parse_environment_variables()
            framework_command = get_framework_command()

            assert framework_command is not None
            assert "tensorrt_llm_server" in framework_command

            generated_config = generate_supervisord_config(
                framework_command, config, "tensorrt-server"
            )

            assert "[program:tensorrt-server]" in generated_config
            assert "tensorrt_llm_server" in generated_config
            assert "autorestart=false" in generated_config
            assert "startretries=1" in generated_config
            assert "loglevel=debug" in generated_config

    def test_configuration_error_handling(self):
        """Test error handling in configuration generation."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        # Test with invalid configuration values
        with pytest.raises(ValueError, match="Framework command cannot be empty"):
            generate_supervisord_config("")

        with pytest.raises(ValueError, match="Program name cannot be empty"):
            generate_supervisord_config("python app.py", program_name="")

    def test_framework_command_resolution_priority(self):
        """Test that framework command resolution follows correct priority."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
        )

        # Test priority: FRAMEWORK_COMMAND > FRAMEWORK_NAME
        env_vars = {"FRAMEWORK_COMMAND": "explicit command", "FRAMEWORK_NAME": "vllm"}

        with patch.dict(os.environ, env_vars, clear=True):
            command = get_framework_command()
            assert command == "explicit command"

        # Test fallback to framework name when FRAMEWORK_COMMAND is empty
        env_vars = {"FRAMEWORK_COMMAND": "   ", "FRAMEWORK_NAME": "vllm"}

        with patch.dict(os.environ, env_vars, clear=True):
            command = get_framework_command()
            assert "vllm" in command

    def test_configuration_file_permissions_and_structure(self):
        """Test that generated configuration files have correct permissions and structure."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            write_supervisord_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")

            write_supervisord_config(config_path, "python app.py")

            # Check file exists and is readable
            assert os.path.exists(config_path)
            assert os.access(config_path, os.R_OK)

            # Check file structure
            with open(config_path, "r") as f:
                content = f.read()

                # Must have supervisord section
                assert "[supervisord]" in content
                assert "nodaemon=true" in content

                # Must have program section
                assert "[program:framework]" in content
                assert "command=python app.py" in content

                # Must have logging configuration
                assert "stdout_logfile=/dev/stdout" in content
                assert "stderr_logfile=/dev/stderr" in content

    def test_multiple_framework_support(self):
        """Test configuration generation for multiple supported frameworks."""
        from model_hosting_container_standards.supervisor.framework_config import (
            get_framework_command,
            get_supported_frameworks,
        )
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        supported_frameworks = get_supported_frameworks()

        for framework_name, expected_command in supported_frameworks.items():
            with patch.dict(os.environ, {"FRAMEWORK_NAME": framework_name}, clear=True):
                # Test framework command resolution
                command = get_framework_command()
                assert command == expected_command

                # Test configuration generation
                config = generate_supervisord_config(
                    command, program_name=framework_name
                )
                assert f"[program:{framework_name}]" in config
                assert f"command={expected_command}" in config

    def test_environment_variable_validation_integration(self):
        """Test integration of environment variable validation across modules."""
        from model_hosting_container_standards.supervisor.config import (
            parse_environment_variables,
        )

        # Test with valid environment variables
        valid_env = {
            "ENGINE_AUTO_RECOVERY": "true",
            "ENGINE_MAX_RECOVERY_ATTEMPTS": "5",
            "ENGINE_RECOVERY_BACKOFF_SECONDS": "15",
            "SUPERVISOR_LOG_LEVEL": "warn",
            "FRAMEWORK_NAME": "vllm",
        }

        with patch.dict(os.environ, valid_env, clear=True):
            config = parse_environment_variables()
            assert config.auto_recovery is True
            assert config.max_recovery_attempts == 5
            assert config.recovery_backoff_seconds == 15
            assert config.log_level == "warn"

        # Test with invalid environment variables - these should use defaults with warnings, not raise errors
        invalid_env_cases = [
            {"ENGINE_AUTO_RECOVERY": "invalid"},
            {"ENGINE_MAX_RECOVERY_ATTEMPTS": "-1"},
            {"SUPERVISOR_LOG_LEVEL": "invalid"},
            {"FRAMEWORK_NAME": "unsupported"},
        ]

        for invalid_env in invalid_env_cases:
            with patch.dict(os.environ, invalid_env, clear=True):
                # Should not raise exception, but use defaults
                config = parse_environment_variables()
                assert config is not None

    def test_module_consistency_across_functions(self):
        """Test that different module functions produce consistent results."""
        from model_hosting_container_standards.supervisor.config import (
            parse_environment_variables,
        )
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
            write_supervisord_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "module_config.conf")

            env_vars = {
                "FRAMEWORK_COMMAND": "python test_server.py",
                "ENGINE_AUTO_RECOVERY": "false",
                "ENGINE_MAX_RECOVERY_ATTEMPTS": "2",
                "SUPERVISOR_LOG_LEVEL": "error",
            }

            with patch.dict(os.environ, env_vars, clear=True):
                # Generate config using generate function
                config = parse_environment_variables()
                generated_content = generate_supervisord_config(
                    "python test_server.py", config, "test-program"
                )

                # Generate config using write function
                write_supervisord_config(
                    config_path, "python test_server.py", config, "test-program"
                )

                # Compare generated configurations
                with open(config_path, "r") as f:
                    written_content = f.read()

                assert generated_content == written_content

    def test_entrypoint_script_exists_and_executable(self):
        """Test that the entrypoint script exists and has proper structure."""
        assert self.entrypoint_script_path.exists()
        assert self.entrypoint_script_path.is_file()

        # Check that script has bash shebang
        with open(self.entrypoint_script_path, "r") as f:
            first_line = f.readline().strip()
            assert first_line.startswith("#!/")
            assert "bash" in first_line or "sh" in first_line

    def test_directory_creation_integration(self):
        """Test that configuration directory creation works across modules."""
        from model_hosting_container_standards.supervisor.supervisor_config import (
            write_supervisord_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test deeply nested directory creation
            nested_path = os.path.join(temp_dir, "a", "b", "c", "d", "supervisord.conf")

            write_supervisord_config(nested_path, "python app.py")

            assert os.path.exists(nested_path)
            assert os.path.isfile(nested_path)

            # Verify all parent directories were created
            parent_dir = os.path.dirname(nested_path)
            assert os.path.exists(parent_dir)
            assert os.path.isdir(parent_dir)

    def test_configuration_template_completeness(self):
        """Test that generated configuration includes all required supervisord sections."""
        from model_hosting_container_standards.supervisor.config import SupervisorConfig
        from model_hosting_container_standards.supervisor.supervisor_config import (
            generate_supervisord_config,
        )

        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            recovery_backoff_seconds=10,
            log_level="info",
        )

        generated_config = generate_supervisord_config("python app.py", config)

        # Check required supervisord sections
        required_supervisord_settings = [
            "nodaemon=true",
            "loglevel=info",
            "logfile=/dev/stdout",
            "pidfile=/tmp/supervisord.pid",
        ]

        for setting in required_supervisord_settings:
            assert setting in generated_config

        # Check required program sections
        required_program_settings = [
            "command=python app.py",
            "autostart=true",
            "autorestart=true",
            "startretries=3",
            "stdout_logfile=/dev/stdout",
            "stderr_logfile=/dev/stderr",
        ]

        for setting in required_program_settings:
            assert setting in generated_config
