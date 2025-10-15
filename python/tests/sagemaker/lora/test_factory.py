"""Unit tests for LoRA factory module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request

from model_hosting_container_standards.sagemaker.lora.constants import LoRAHandlerType
from model_hosting_container_standards.sagemaker.lora.factory import (
    _resolve_transforms,
    create_transform_decorator,
)


class TestResolveTransforms:
    """Test _resolve_transforms function."""

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.get_transform_cls_from_handler_type"
    )
    def test_resolve_transforms_register_adapter(self, mock_get_transform_cls):
        """Test _resolve_transforms with register_adapter handler type."""
        # Arrange
        handler_type = LoRAHandlerType.REGISTER_ADAPTER
        request_shape = {"model": "body.name", "source": "body.src"}
        response_shape = {"status": "success"}

        mock_transform_cls = Mock()
        mock_transformer_instance = Mock()
        mock_transform_cls.return_value = mock_transformer_instance
        mock_get_transform_cls.return_value = mock_transform_cls

        # Act
        result = _resolve_transforms(handler_type, request_shape, response_shape)

        # Assert
        mock_get_transform_cls.assert_called_once_with(handler_type)
        mock_transform_cls.assert_called_once_with(request_shape, response_shape)
        assert result == mock_transformer_instance

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.get_transform_cls_from_handler_type"
    )
    def test_resolve_transforms_unregister_adapter(self, mock_get_transform_cls):
        """Test _resolve_transforms with unregister_adapter handler type."""
        # Arrange
        handler_type = LoRAHandlerType.UNREGISTER_ADAPTER
        request_shape = {"model": "path_params.adapter_name"}
        response_shape = {}

        mock_transform_cls = Mock()
        mock_transformer_instance = Mock()
        mock_transform_cls.return_value = mock_transformer_instance
        mock_get_transform_cls.return_value = mock_transform_cls

        # Act
        result = _resolve_transforms(handler_type, request_shape, response_shape)

        # Assert
        mock_get_transform_cls.assert_called_once_with(handler_type)
        mock_transform_cls.assert_called_once_with(request_shape, response_shape)
        assert result == mock_transformer_instance

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.get_transform_cls_from_handler_type"
    )
    def test_resolve_transforms_adapter_header_to_body(self, mock_get_transform_cls):
        """Test _resolve_transforms with adapter_header_to_body handler type."""
        # Arrange
        handler_type = LoRAHandlerType.ADAPTER_ID
        request_shape = {"model": "headers.X-Amzn-SageMaker-Adapter-Identifier"}
        response_shape = {}

        mock_transform_cls = Mock()
        mock_transformer_instance = Mock()
        mock_transform_cls.return_value = mock_transformer_instance
        mock_get_transform_cls.return_value = mock_transform_cls

        # Act
        result = _resolve_transforms(handler_type, request_shape, response_shape)

        # Assert
        mock_get_transform_cls.assert_called_once_with(handler_type)
        mock_transform_cls.assert_called_once_with(request_shape, response_shape)
        assert result == mock_transformer_instance

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.get_transform_cls_from_handler_type"
    )
    def test_resolve_transforms_empty_shapes(self, mock_get_transform_cls):
        """Test _resolve_transforms with empty request and response shapes."""
        # Arrange
        handler_type = LoRAHandlerType.REGISTER_ADAPTER
        request_shape = {}
        response_shape = {}

        mock_transform_cls = Mock()
        mock_transformer_instance = Mock()
        mock_transform_cls.return_value = mock_transformer_instance
        mock_get_transform_cls.return_value = mock_transform_cls

        # Act
        result = _resolve_transforms(handler_type, request_shape, response_shape)

        # Assert
        mock_get_transform_cls.assert_called_once_with(handler_type)
        mock_transform_cls.assert_called_once_with(request_shape, response_shape)
        assert result == mock_transformer_instance

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.get_transform_cls_from_handler_type"
    )
    def test_resolve_transforms_raises_value_error_for_invalid_handler_type(
        self, mock_get_transform_cls
    ):
        """Test _resolve_transforms raises ValueError for invalid handler type."""
        # Arrange
        handler_type = "invalid_handler_type"
        request_shape = {"model": "body.name"}
        response_shape = {}

        mock_get_transform_cls.side_effect = ValueError("Unsupported handler type")

        # Act & Assert
        with pytest.raises(ValueError):
            _resolve_transforms(handler_type, request_shape, response_shape)


class TestCreateTransformDecorator:
    """Test create_transform_decorator function."""

    def test_create_transform_decorator_returns_decorator_factory(self):
        """Test that create_transform_decorator returns a decorator factory function."""
        # Act
        decorator_factory = create_transform_decorator(LoRAHandlerType.REGISTER_ADAPTER)

        # Assert
        assert callable(decorator_factory)

    @patch("model_hosting_container_standards.sagemaker.lora.factory.handler_registry")
    @patch("model_hosting_container_standards.sagemaker.lora.factory.logger")
    def test_decorator_with_no_shapes_registers_passthrough_handler(
        self, mock_logger, mock_registry
    ):
        """Test decorator with no request/response shapes registers as passthrough and returns original function."""
        # Arrange
        handler_type = LoRAHandlerType.REGISTER_ADAPTER
        mock_func = AsyncMock()
        mock_func.__name__ = "test_handler"

        # Act
        decorator_factory = create_transform_decorator(handler_type)
        decorator = decorator_factory()
        result = decorator(mock_func)

        # Assert
        mock_logger.info.assert_any_call(
            "No transform shapes defined, using passthrough"
        )
        mock_registry.set_handler.assert_called_once_with(str(handler_type), mock_func)
        mock_logger.info.assert_any_call(
            f"[{handler_type.upper()}] Registered transform handler for {mock_func.__name__}"
        )
        # The decorator should return the original function when no shapes are provided
        assert result == mock_func

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory._resolve_transforms"
    )
    @patch("model_hosting_container_standards.sagemaker.lora.factory.handler_registry")
    @patch("model_hosting_container_standards.sagemaker.lora.factory.logger")
    def test_decorator_with_shapes_creates_wrapper_function(
        self, mock_logger, mock_registry, mock_resolve_transforms
    ):
        """Test decorator with shapes creates a wrapped function with transformations."""
        # Arrange
        handler_type = LoRAHandlerType.REGISTER_ADAPTER
        request_shape = {"model": "body.name"}
        response_shape = {"status": "success"}

        mock_func = AsyncMock()
        mock_func.__name__ = "test_handler"
        mock_func.return_value = {"result": "success"}

        mock_transformer = Mock()
        mock_transform_output = Mock()
        mock_transform_output.request = {"model": "test-adapter"}
        mock_transform_output.raw_request = Mock(spec=Request)
        mock_transformer.transform_request = AsyncMock(
            return_value=mock_transform_output
        )
        mock_transformer.transform_response = Mock(
            return_value={"transformed": "response"}
        )
        mock_resolve_transforms.return_value = mock_transformer

        # Act
        decorator_factory = create_transform_decorator(handler_type)
        decorator = decorator_factory(request_shape, response_shape)
        wrapped_func = decorator(mock_func)

        # Assert
        mock_resolve_transforms.assert_called_once_with(
            handler_type, request_shape, response_shape
        )
        mock_logger.info.assert_any_call(
            f"[{handler_type.upper()}] Transform decorator applied to: {mock_func.__name__}"
        )
        mock_registry.set_handler.assert_called_once_with(handler_type, wrapped_func)
        assert callable(wrapped_func)

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory._resolve_transforms"
    )
    @patch("model_hosting_container_standards.sagemaker.lora.factory.handler_registry")
    @patch("model_hosting_container_standards.sagemaker.lora.factory.logger")
    @pytest.mark.asyncio
    async def test_wrapped_function_with_transformed_request(
        self, mock_logger, mock_registry, mock_resolve_transforms
    ):
        """Test wrapped function executes with transformed request data."""
        # Arrange
        handler_type = LoRAHandlerType.REGISTER_ADAPTER
        request_shape = {"model": "body.name"}
        response_shape = {}

        mock_func = AsyncMock()
        mock_func.__name__ = "test_handler"
        mock_func.return_value = {"result": "success"}

        mock_transformer = Mock()
        mock_raw_request = Mock(spec=Request)
        mock_transform_output = Mock()
        mock_transform_output.request = {"model": "test-adapter"}
        mock_transform_output.raw_request = mock_raw_request
        mock_transformer.transform_request = AsyncMock(
            return_value=mock_transform_output
        )
        mock_transformer.transform_response = Mock(
            return_value={"transformed": "response"}
        )
        mock_resolve_transforms.return_value = mock_transformer

        # Act
        decorator_factory = create_transform_decorator(handler_type)
        decorator = decorator_factory(request_shape, response_shape)
        wrapped_func = decorator(mock_func)

        result = await wrapped_func(mock_raw_request)

        # Assert
        mock_transformer.transform_request.assert_called_once_with(mock_raw_request)
        # Verify the function was called with SimpleNamespace for structured access
        call_args = mock_func.call_args
        assert len(call_args[0]) == 2  # Two positional arguments
        assert isinstance(call_args[0][0], SimpleNamespace)
        assert call_args[0][0].model == "test-adapter"
        assert call_args[0][1] == mock_raw_request
        mock_transformer.transform_response.assert_called_once_with(
            {"result": "success"}, mock_transform_output
        )
        assert result == {"transformed": "response"}

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory._resolve_transforms"
    )
    @patch("model_hosting_container_standards.sagemaker.lora.factory.handler_registry")
    @patch("model_hosting_container_standards.sagemaker.lora.factory.logger")
    @pytest.mark.asyncio
    async def test_wrapped_function_with_none_transformed_request(
        self, mock_logger, mock_registry, mock_resolve_transforms
    ):
        """Test wrapped function executes when transformed request is None."""
        # Arrange
        handler_type = LoRAHandlerType.REGISTER_ADAPTER
        request_shape = {"model": "body.name"}
        response_shape = {}

        mock_func = AsyncMock()
        mock_func.__name__ = "test_handler"
        mock_func.return_value = {"result": "success"}

        mock_transformer = Mock()
        mock_raw_request = Mock(spec=Request)
        mock_transform_output = Mock()
        mock_transform_output.request = None  # No transformed request data
        mock_transform_output.raw_request = mock_raw_request
        mock_transformer.transform_request = AsyncMock(
            return_value=mock_transform_output
        )
        mock_transformer.transform_response = Mock(
            return_value={"transformed": "response"}
        )
        mock_resolve_transforms.return_value = mock_transformer

        # Act
        decorator_factory = create_transform_decorator(handler_type)
        decorator = decorator_factory(request_shape, response_shape)
        wrapped_func = decorator(mock_func)

        result = await wrapped_func(mock_raw_request)

        # Assert
        mock_transformer.transform_request.assert_called_once_with(mock_raw_request)
        # Verify the function was called with only the raw request
        mock_func.assert_called_once_with(mock_raw_request)
        mock_transformer.transform_response.assert_called_once_with(
            {"result": "success"}, mock_transform_output
        )
        assert result == {"transformed": "response"}

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory._resolve_transforms"
    )
    @patch("model_hosting_container_standards.sagemaker.lora.factory.handler_registry")
    @patch("model_hosting_container_standards.sagemaker.lora.factory.logger")
    @pytest.mark.asyncio
    async def test_wrapped_function_handles_exceptions(
        self, mock_logger, mock_registry, mock_resolve_transforms
    ):
        """Test wrapped function properly propagates exceptions from the handler."""
        # Arrange
        handler_type = LoRAHandlerType.REGISTER_ADAPTER
        request_shape = {"model": "body.name"}
        response_shape = {}

        mock_func = AsyncMock()
        mock_func.__name__ = "test_handler"
        mock_func.side_effect = ValueError("Handler error")

        mock_transformer = Mock()
        mock_raw_request = Mock(spec=Request)
        mock_transform_output = Mock()
        mock_transform_output.request = {"model": "test-adapter"}
        mock_transform_output.raw_request = mock_raw_request
        mock_transformer.transform_request = AsyncMock(
            return_value=mock_transform_output
        )
        mock_resolve_transforms.return_value = mock_transformer

        # Act
        decorator_factory = create_transform_decorator(handler_type)
        decorator = decorator_factory(request_shape, response_shape)
        wrapped_func = decorator(mock_func)

        # Assert
        with pytest.raises(ValueError, match="Handler error"):
            await wrapped_func(mock_raw_request)

    def test_multiple_handler_types_create_different_decorators(self):
        """Test that different handler types create different decorator factories."""
        # Act
        register_decorator_factory = create_transform_decorator(
            LoRAHandlerType.REGISTER_ADAPTER
        )
        unregister_decorator_factory = create_transform_decorator(
            LoRAHandlerType.UNREGISTER_ADAPTER
        )
        header_to_body_decorator_factory = create_transform_decorator(
            LoRAHandlerType.ADAPTER_ID
        )

        # Assert
        assert callable(register_decorator_factory)
        assert callable(unregister_decorator_factory)
        assert callable(header_to_body_decorator_factory)
        # Verify they are different functions
        assert register_decorator_factory != unregister_decorator_factory
        assert register_decorator_factory != header_to_body_decorator_factory
        assert unregister_decorator_factory != header_to_body_decorator_factory

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory._resolve_transforms"
    )
    @patch("model_hosting_container_standards.sagemaker.lora.factory.handler_registry")
    @patch("model_hosting_container_standards.sagemaker.lora.factory.logger")
    def test_decorator_called_multiple_times_with_same_shapes(
        self, mock_logger, mock_registry, mock_resolve_transforms
    ):
        """Test decorator can be applied to multiple functions with same configuration."""
        # Arrange
        handler_type = LoRAHandlerType.REGISTER_ADAPTER
        request_shape = {"model": "body.name"}
        response_shape = {}

        mock_func1 = AsyncMock()
        mock_func1.__name__ = "handler1"
        mock_func2 = AsyncMock()
        mock_func2.__name__ = "handler2"

        mock_transformer = Mock()
        mock_resolve_transforms.return_value = mock_transformer

        # Act
        decorator_factory = create_transform_decorator(handler_type)
        decorator = decorator_factory(request_shape, response_shape)

        wrapped_func1 = decorator(mock_func1)
        wrapped_func2 = decorator(mock_func2)

        # Assert
        # _resolve_transforms should be called twice (once for each function)
        assert mock_resolve_transforms.call_count == 2
        # Both functions should be registered
        assert mock_registry.set_handler.call_count == 2
        assert callable(wrapped_func1)
        assert callable(wrapped_func2)
        assert wrapped_func1 != wrapped_func2


class TestIntegration:
    """Integration tests for the factory module."""

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.get_transform_cls_from_handler_type"
    )
    @patch("model_hosting_container_standards.sagemaker.lora.factory.handler_registry")
    @patch("model_hosting_container_standards.sagemaker.lora.factory.logger")
    @pytest.mark.asyncio
    async def test_end_to_end_decorator_flow(
        self, mock_logger, mock_registry, mock_get_transform_cls
    ):
        """Test the complete end-to-end flow of creating and using a transform decorator."""
        # Arrange
        handler_type = LoRAHandlerType.REGISTER_ADAPTER
        request_shape = {"model": "body.name", "source": "body.src"}
        response_shape = {"status": "success"}

        # Mock the transformer class and instance
        mock_transform_cls = Mock()
        mock_transformer = Mock()
        mock_transform_cls.return_value = mock_transformer
        mock_get_transform_cls.return_value = mock_transform_cls

        # Mock transformer behavior
        mock_raw_request = Mock(spec=Request)
        mock_transform_output = Mock()
        mock_transform_output.request = {"model": "test-adapter", "source": "s3://test"}
        mock_transform_output.raw_request = mock_raw_request
        mock_transformer.transform_request = AsyncMock(
            return_value=mock_transform_output
        )
        mock_transformer.transform_response = Mock(
            return_value={"status": "registered"}
        )

        # Define a test handler function
        @create_transform_decorator(handler_type)(request_shape, response_shape)
        async def test_handler(request: SimpleNamespace, raw_request: Request):
            return {"original": f"Processing {request.model} from {request.source}"}

        # Act
        result = await test_handler(mock_raw_request)

        # Assert
        # Verify transformer was created correctly
        mock_get_transform_cls.assert_called_once_with(handler_type)
        mock_transform_cls.assert_called_once_with(request_shape, response_shape)

        # Verify transformation flow
        mock_transformer.transform_request.assert_called_once_with(mock_raw_request)
        mock_transformer.transform_response.assert_called_once()

        # Verify handler registration
        mock_registry.set_handler.assert_called_once()
        call_args = mock_registry.set_handler.call_args
        assert call_args[0][0] == handler_type
        assert callable(call_args[0][1])  # The wrapped function

        # Verify final result
        assert result == {"status": "registered"}
