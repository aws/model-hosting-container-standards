"""Unit tests for route configuration and registry functions.

NOTE: This test file has been kept for backward compatibility, but the actual
route configuration has been moved to ../routes.py. The comprehensive tests
for the new location are in ../test_routes.py.

These tests verify that the new route configuration still works correctly.
"""

import pytest

from model_hosting_container_standards.common.fastapi.routing import RouteConfig
from model_hosting_container_standards.sagemaker.lora.constants import LoRAHandlerType
from model_hosting_container_standards.sagemaker.lora.routes import (
    get_lora_route_config,
)


class TestRouteConfig:
    """Test RouteConfig dataclass."""

    def test_route_config_is_frozen(self):
        """Test that RouteConfig is immutable (frozen)."""
        # Arrange
        config = RouteConfig(path="/test", method="POST")

        # Act & Assert
        with pytest.raises(AttributeError):
            config.path = "/modified"

    def test_route_config_equality(self):
        """Test RouteConfig equality comparison."""
        # Arrange
        config1 = RouteConfig(path="/test", method="POST", tags=["tag1"])
        config2 = RouteConfig(path="/test", method="POST", tags=["tag1"])
        config3 = RouteConfig(path="/other", method="POST", tags=["tag1"])

        # Assert
        assert config1 == config2
        assert config1 != config3


class TestGetHandlerRouteConfig:
    """Test get_lora_route_config function."""

    def test_get_register_adapter_config(self):
        """Test getting config for REGISTER_ADAPTER handler."""
        # Act
        config = get_lora_route_config(LoRAHandlerType.REGISTER_ADAPTER)

        # Assert
        assert config is not None
        assert config.path == "/adapters"
        assert config.method == "POST"
        assert "adapters" in config.tags
        assert "lora" in config.tags
        assert config.summary == "Register a new LoRA adapter"

    def test_get_unregister_adapter_config(self):
        """Test getting config for UNREGISTER_ADAPTER handler."""
        # Act
        config = get_lora_route_config(LoRAHandlerType.UNREGISTER_ADAPTER)

        # Assert
        assert config is not None
        assert config.path == "/adapters/{adapter_name}"
        assert config.method == "DELETE"
        assert "adapters" in config.tags
        assert "lora" in config.tags
        assert config.summary == "Unregister a LoRA adapter"

    def test_get_inject_adapter_id_config(self):
        """Test getting config for INJECT_ADAPTER_ID handler (no route)."""
        # Act
        config = get_lora_route_config(LoRAHandlerType.INJECT_ADAPTER_ID)

        # Assert
        assert config is None

    def test_get_invalid_handler_type_returns_none(self):
        """Test getting config for unregistered handler type returns None."""
        # Act
        config = get_lora_route_config("completely_invalid")

        # Assert
        assert config is None
