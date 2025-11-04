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

    def test_core_exit_behavior_settings(self):
        """Test that all critical exit behavior settings are configured correctly."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            launch_command="python -m llm_service",
            log_level="info",
        )

        config_content = generate_supervisord_config(config, "test-service")

        # Core exit behavior settings
        assert "exitcodes=255" in config_content  # Only 255 is "expected"
        assert "startsecs=1" in config_content  # Must run 1 sec minimum
        assert "autorestart=true" in config_content
        assert "startretries=3" in config_content
        assert "[program:test-service]" in config_content

    @pytest.mark.parametrize(
        "auto_recovery,expected",
        [
            (True, "autorestart=true"),
            (False, "autorestart=false"),
        ],
    )
    def test_autorestart_behavior(self, auto_recovery, expected):
        """Test autorestart setting based on auto_recovery flag."""
        config = SupervisorConfig(
            auto_recovery=auto_recovery,
            max_start_retries=2,
            launch_command="python -m service",
            log_level="info",
        )

        config_content = generate_supervisord_config(config)
        assert expected in config_content
        # Exit behavior should be consistent regardless of auto_recovery
        assert "exitcodes=255" in config_content

    @pytest.mark.parametrize("retries", [0, 1, 5, 100])
    def test_retry_limits(self, retries):
        """Test that startretries matches max_start_retries."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=retries,
            launch_command="echo test",
            log_level="info",
        )

        config_content = generate_supervisord_config(config)
        assert f"startretries={retries}" in config_content

    def test_container_logging_configuration(self):
        """Test logging configuration for container environments."""
        config = SupervisorConfig(
            auto_recovery=True,
            max_start_retries=3,
            launch_command="python -m service",
            log_level="debug",
        )

        config_content = generate_supervisord_config(config)

        # Container-friendly logging
        assert "stdout_logfile=/dev/stdout" in config_content
        assert "stderr_logfile=/dev/stderr" in config_content
        assert "stdout_logfile_maxbytes=0" in config_content
        assert "nodaemon=true" in config_content
        assert "loglevel=debug" in config_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
