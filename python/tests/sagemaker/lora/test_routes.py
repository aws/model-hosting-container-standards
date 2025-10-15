"""Unit tests for LoRA route configuration."""

from model_hosting_container_standards.sagemaker.lora.constants import LoRAHandlerType
from model_hosting_container_standards.sagemaker.lora.routes import LORA_ROUTE_REGISTRY


class TestLoRARouteConfigs:
    """Test individual LoRA route configurations."""

    def test_register_adapter_route_path(self):
        """Test REGISTER_ADAPTER has correct path."""
        # Arrange
        config = LORA_ROUTE_REGISTRY[LoRAHandlerType.REGISTER_ADAPTER]

        # Assert
        assert config.path == "/adapters"

    def test_register_adapter_route_method(self):
        """Test REGISTER_ADAPTER uses POST method."""
        # Arrange
        config = LORA_ROUTE_REGISTRY[LoRAHandlerType.REGISTER_ADAPTER]

        # Assert
        assert config.method == "POST"

    def test_unregister_adapter_route_path(self):
        """Test UNREGISTER_ADAPTER has correct path with parameter."""
        # Arrange
        config = LORA_ROUTE_REGISTRY[LoRAHandlerType.UNREGISTER_ADAPTER]

        # Assert
        assert config.path == "/adapters/{adapter_name}"
        assert "{adapter_name}" in config.path  # Verify path param exists

    def test_unregister_adapter_route_method(self):
        """Test UNREGISTER_ADAPTER uses DELETE method."""
        # Arrange
        config = LORA_ROUTE_REGISTRY[LoRAHandlerType.UNREGISTER_ADAPTER]

        # Assert
        assert config.method == "DELETE"
