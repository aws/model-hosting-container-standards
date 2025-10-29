"""
Supervisor process management module for ML frameworks.

This module provides supervisord-based process management capabilities
for containerized ML frameworks, enabling automatic process recovery
and self-contained resilience.
"""

from .generator import generate_supervisord_config, write_supervisord_config
from .models import ConfigurationError, SupervisorConfig, get_launch_command

__all__ = [
    "SupervisorConfig",
    "ConfigurationError",
    "generate_supervisord_config",
    "write_supervisord_config",
    "get_launch_command",
]
