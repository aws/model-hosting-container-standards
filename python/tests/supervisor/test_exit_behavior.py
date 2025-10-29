"""
Unit tests specifically for the exit behavior and monitoring logic.

These tests focus on the core logic that makes LLM services restart on any exit
and exit the container when max retries are exceeded.
"""

import pytest

from model_hosting_container_standards.supervisor.generator import (
    generate_supervisord_config,
)
from model_hosting_container_standards.supervisor.models import SupervisorConfig


class TestExitBehaviorLogic:
    """Test the core exit behavior logic."""

    def test_exit_codes_configuration(self):
        """Test that exitcodes=255 is set to treat all normal exits as unexpected."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command="python -m llm_service",
            log_level="info",
        )

        config_content = generate_supervisord_config(config)

        # Critical: Only exit code 255 should be "expected"
        # This means exit codes 0, 1, 2, etc. will all trigger restarts
        assert "exitcodes=255" in config_content

    def test_start_seconds_configuration(self):
        """Test that startsecs=1 is set to require minimum runtime."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=5,
            launch_command="python -m my_service",
            log_level="debug",
        )

        config_content = generate_supervisord_config(config)

        # Process must run at least 1 second to be considered successfully started
        # This prevents rapid restart loops for immediately failing services
        assert "startsecs=1" in config_content

    def test_autorestart_behavior_with_recovery_enabled(self):
        """Test autorestart=true when auto_recovery is enabled."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=2,
            launch_command="service --port 8080",
            log_level="warn",
        )

        config_content = generate_supervisord_config(config)

        # Should automatically restart failed processes
        assert "autorestart=true" in config_content

    def test_autorestart_behavior_with_recovery_disabled(self):
        """Test autorestart=false when auto_recovery is disabled."""
        config = SupervisorConfig(
            auto_recovery=False,
            max_recovery_attempts=1,
            launch_command="service --port 8080",
            log_level="error",
        )

        config_content = generate_supervisord_config(config)

        # Should not automatically restart when recovery is disabled
        assert "autorestart=false" in config_content

    def test_retry_limit_configuration(self):
        """Test that startretries matches max_recovery_attempts."""
        test_cases = [0, 1, 3, 5, 10, 100]

        for max_attempts in test_cases:
            config = SupervisorConfig(
                auto_recovery=True,
                max_recovery_attempts=max_attempts,
                launch_command="echo test",
                log_level="info",
            )

            config_content = generate_supervisord_config(config)

            # Should match exactly
            assert f"startretries={max_attempts}" in config_content

    def test_program_name_in_configuration(self):
        """Test that program name is correctly set in configuration."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command="python -m vllm.entrypoints.api_server",
            log_level="info",
        )

        # Test default program name
        config_content = generate_supervisord_config(config)
        assert "[program:llm-engine]" in config_content

        # Test custom program name
        config_content = generate_supervisord_config(config, "custom-service")
        assert "[program:custom-service]" in config_content

    def test_logging_configuration_for_containers(self):
        """Test that logging is configured for container environments."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command="python -m service",
            log_level="info",
        )

        config_content = generate_supervisord_config(config)

        # Should log to stdout/stderr for container compatibility
        assert "stdout_logfile=/dev/stdout" in config_content
        assert "stderr_logfile=/dev/stderr" in config_content
        assert "logfile=/dev/stdout" in config_content

        # Should not rotate logs (maxbytes=0)
        assert "stdout_logfile_maxbytes=0" in config_content
        assert "stderr_logfile_maxbytes=0" in config_content
        assert "logfile_maxbytes=0" in config_content

    def test_supervisord_daemon_configuration(self):
        """Test supervisord daemon configuration for containers."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=3,
            launch_command="python -m service",
            log_level="debug",
        )

        config_content = generate_supervisord_config(config)

        # Should run in foreground for containers
        assert "nodaemon=true" in config_content

        # Should use specified log level
        assert "loglevel=debug" in config_content

    def test_complete_exit_behavior_configuration(self):
        """Test that all exit behavior settings work together correctly."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=4,
            launch_command="python -m llm_engine --config /app/config.yaml",
            log_level="warn",
        )

        config_content = generate_supervisord_config(config, "my-llm-service")

        # Verify all critical exit behavior settings are present
        lines = config_content.split("\n")

        # Program section should exist
        assert any("[program:my-llm-service]" in line for line in lines)

        # Command should be correct
        assert any(
            "python -m llm_engine --config /app/config.yaml" in line for line in lines
        )

        # Exit behavior settings
        assert any("exitcodes=255" in line for line in lines)  # Only 255 is expected
        assert any("startsecs=1" in line for line in lines)  # Must run 1 sec minimum
        assert any("autorestart=true" in line for line in lines)  # Auto restart enabled
        assert any("startretries=4" in line for line in lines)  # Max 4 restart attempts

        # Logging settings
        assert any("loglevel=warn" in line for line in lines)
        assert any("stdout_logfile=/dev/stdout" in line for line in lines)

    def test_edge_case_zero_retries(self):
        """Test behavior with zero retry attempts."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_recovery_attempts=0,
            launch_command="python -m service",
            log_level="info",
        )

        config_content = generate_supervisord_config(config)

        # Should still have exit behavior settings even with 0 retries
        assert "startretries=0" in config_content
        assert "exitcodes=255" in config_content
        assert "startsecs=1" in config_content

    def test_configuration_consistency_across_settings(self):
        """Test that configuration is consistent across different auto_recovery settings."""
        base_config = {
            "max_recovery_attempts": 3,
            "launch_command": "python -m test_service",
            "log_level": "info",
        }

        # Test with auto_recovery=True
        config_enabled = SupervisorConfig(auto_recovery=True, **base_config)
        content_enabled = generate_supervisord_config(config_enabled)

        # Test with auto_recovery=False
        config_disabled = SupervisorConfig(auto_recovery=False, **base_config)
        content_disabled = generate_supervisord_config(config_disabled)

        # Both should have the same exit behavior settings
        for content in [content_enabled, content_disabled]:
            assert "exitcodes=255" in content
            assert "startsecs=1" in content
            assert "startretries=3" in content

        # Only autorestart should differ
        assert "autorestart=true" in content_enabled
        assert "autorestart=false" in content_disabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
