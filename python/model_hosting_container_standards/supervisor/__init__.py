"""
Supervisor process management module for ML frameworks.

This module provides supervisord-based process management capabilities
for containerized ML frameworks, enabling automatic process recovery
and self-contained resilience.
"""

from .config import ConfigurationError, SupervisorConfig
from .framework_config import get_framework_command, validate_framework_command
from .supervisor_config import generate_supervisord_config, write_supervisord_config

__all__ = [
    "SupervisorConfig",
    "ConfigurationError",
    "generate_supervisord_config",
    "write_supervisord_config",
    "get_framework_command",
    "validate_framework_command",
]
