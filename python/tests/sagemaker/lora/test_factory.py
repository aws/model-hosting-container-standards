"""Unit tests for LoRA factory module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request

from model_hosting_container_standards.common.transforms.base_factory import (
    _resolve_transforms,
)
from model_hosting_container_standards.sagemaker.lora.constants import LoRAHandlerType
from model_hosting_container_standards.sagemaker.lora.factory import (
    create_lora_transform_decorator,
)


class TestResolveTransforms:
    """Test _resolve_transforms function."""

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.resolve_lora_transform"
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
        result = _resolve_transforms(
            handler_type, mock_get_transform_cls, request_shape, response_shape
        )

        # Assert
        mock_get_transform_cls.assert_called_once_with(handler_type)
        mock_transform_cls.assert_called_once_with(request_shape, response_shape)
        assert result == mock_transformer_instance

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.resolve_lora_transform"
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
        result = _resolve_transforms(
            handler_type, mock_get_transform_cls, request_shape, response_shape
        )

        # Assert
        mock_get_transform_cls.assert_called_once_with(handler_type)
        mock_transform_cls.assert_called_once_with(request_shape, response_shape)
        assert result == mock_transformer_instance

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.resolve_lora_transform"
    )
    def test_resolve_transforms_inject_adapter_id(self, mock_get_transform_cls):
        """Test _resolve_transforms with inject_adapter_id handler type."""
        # Arrange
        handler_type = LoRAHandlerType.INJECT_ADAPTER_ID
        request_shape = {"model": 'headers."X-Amzn-SageMaker-Adapter-Identifier"'}
        response_shape = {}

        mock_transform_cls = Mock()
        mock_transformer_instance = Mock()
        mock_transform_cls.return_value = mock_transformer_instance
        mock_get_transform_cls.return_value = mock_transform_cls

        # Act
        result = _resolve_transforms(
            handler_type, mock_get_transform_cls, request_shape, response_shape
        )

        # Assert
        mock_get_transform_cls.assert_called_once_with(handler_type)
        mock_transform_cls.assert_called_once_with(request_shape, response_shape)
        assert result == mock_transformer_instance

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.resolve_lora_transform"
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
        result = _resolve_transforms(
            handler_type, mock_get_transform_cls, request_shape, response_shape
        )

        # Assert
        mock_get_transform_cls.assert_called_once_with(handler_type)
        mock_transform_cls.assert_called_once_with(request_shape, response_shape)
        assert result == mock_transformer_instance

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.resolve_lora_transform"
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
            _resolve_transforms(
                handler_type, mock_get_transform_cls, request_shape, response_shape
            )


class TestCreateTransformDecorator:
    """Test create_lora_transform_decorator function."""

    def test_create_lora_transform_decorator_returns_decorator_factory(self):
        """Test that create_lora_transform_decorator returns a decorator factory function."""
        # Act
        decorator_factory = create_lora_transform_decorator(
            LoRAHandlerType.REGISTER_ADAPTER
        )

        # Assert
        assert callable(decorator_factory)

    @patch(
        "model_hosting_container_standards.common.transforms.base_factory.handler_registry"
    )
    @patch("model_hosting_container_standards.common.transforms.base_factory.logger")
    def test_decorator_with_no_shapes_registers_passthrough_handler(
        self, mock_logger, mock_registry
    ):
        """Test decorator with no request/response shapes registers as passthrough and returns original function."""
        # Arrange
        handler_type = str(LoRAHandlerType.REGISTER_ADAPTER)
        mock_func = AsyncMock()
        mock_func.__name__ = "test_handler"

        # Act
        decorator_factory = create_lora_transform_decorator(handler_type)
        decorator = decorator_factory()
        result = decorator(mock_func)

        # Assert
        mock_logger.info.assert_any_call(
            "No transform shapes defined, using passthrough"
        )
        # Verify set_handler was called with HandlerInfo
        mock_registry.set_handler.assert_called_once()
        call_args = mock_registry.set_handler.call_args
        assert call_args[0][0] == handler_type
        handler_info = call_args[0][1]
        assert handler_info.func == mock_func
        assert handler_info.route_kwargs == {}
        mock_logger.info.assert_any_call(
            f"[{handler_type.upper()}] Registered transform handler for {mock_func.__name__}"
        )
        # The decorator should return the original function when no shapes are provided
        assert result == mock_func

    @patch(
        "model_hosting_container_standards.sagemaker.lora.factory.resolve_lora_transform"
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_factory._resolve_transforms"
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_factory.handler_registry"
    )
    @patch("model_hosting_container_standards.common.transforms.base_factory.logger")
    def test_decorator_with_shapes_creates_wrapper_function(
        self, mock_logger, mock_registry, mock_resolve_transforms, mock_lora_transform
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
        decorator_factory = create_lora_transform_decorator(handler_type)
        decorator = decorator_factory(request_shape, response_shape)
        wrapped_func = decorator(mock_func)

        # Assert
        mock_resolve_transforms.assert_called_once_with(
            handler_type, mock_lora_transform, request_shape, response_shape
        )
        mock_logger.info.assert_any_call(
            f"[{handler_type.upper()}] Transform decorator applied to: {mock_func.__name__}"
        )
        # Verify set_handler was called with HandlerInfo
        mock_registry.set_handler.assert_called_once()
        call_args = mock_registry.set_handler.call_args
        assert call_args[0][0] == handler_type
        handler_info = call_args[0][1]
        assert callable(handler_info.func)
        assert handler_info.route_kwargs == {}
        assert callable(wrapped_func)

    @patch(
        "model_hosting_container_standards.common.transforms.base_factory._resolve_transforms"
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_factory.handler_registry"
    )
    @patch("model_hosting_container_standards.common.transforms.base_factory.logger")
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
        mock_transformer.intercept = AsyncMock(return_value={"result": "success"})
        mock_transformer.transform_response = Mock(
            return_value={"transformed": "response"}
        )
        mock_resolve_transforms.return_value = mock_transformer

        # Act
        decorator_factory = create_lora_transform_decorator(handler_type)
        decorator = decorator_factory(request_shape, response_shape)
        wrapped_func = decorator(mock_func)

        result = await wrapped_func(mock_raw_request)

        # Assert
        mock_transformer.transform_request.assert_called_once_with(mock_raw_request)
        # Verify intercept was called with the function and transform output
        mock_transformer.intercept.assert_called_once_with(
            mock_func, mock_transform_output
        )
        mock_transformer.transform_response.assert_called_once_with(
            {"result": "success"}, mock_transform_output
        )
        assert result == {"transformed": "response"}

    @patch(
        "model_hosting_container_standards.common.transforms.base_factory._resolve_transforms"
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_factory.handler_registry"
    )
    @patch("model_hosting_container_standards.common.transforms.base_factory.logger")
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
        mock_transformer.intercept = AsyncMock(return_value={"result": "success"})
        mock_transformer.transform_response = Mock(
            return_value={"transformed": "response"}
        )
        mock_resolve_transforms.return_value = mock_transformer

        # Act
        decorator_factory = create_lora_transform_decorator(handler_type)
        decorator = decorator_factory(request_shape, response_shape)
        wrapped_func = decorator(mock_func)

        result = await wrapped_func(mock_raw_request)

        # Assert
        mock_transformer.transform_request.assert_called_once_with(mock_raw_request)
        # Verify intercept was called with the function and transform output
        mock_transformer.intercept.assert_called_once_with(
            mock_func, mock_transform_output
        )
        mock_transformer.transform_response.assert_called_once_with(
            {"result": "success"}, mock_transform_output
        )
        assert result == {"transformed": "response"}

    @patch(
        "model_hosting_container_standards.common.transforms.base_factory._resolve_transforms"
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_factory.handler_registry"
    )
    @patch("model_hosting_container_standards.common.transforms.base_factory.logger")
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
        mock_transformer.intercept = AsyncMock(side_effect=ValueError("Handler error"))
        mock_resolve_transforms.return_value = mock_transformer

        # Act
        decorator_factory = create_lora_transform_decorator(handler_type)
        decorator = decorator_factory(request_shape, response_shape)
        wrapped_func = decorator(mock_func)

        # Assert
        with pytest.raises(ValueError, match="Handler error"):
            await wrapped_func(mock_raw_request)

    def test_multiple_handler_types_create_different_decorators(self):
        """Test that different handler types create different decorator factories."""
        # Act
        register_decorator_factory = create_lora_transform_decorator(
            LoRAHandlerType.REGISTER_ADAPTER
        )
        unregister_decorator_factory = create_lora_transform_decorator(
            LoRAHandlerType.UNREGISTER_ADAPTER
        )
        inject_adapter_id_decorator_factory = create_lora_transform_decorator(
            LoRAHandlerType.INJECT_ADAPTER_ID
        )

        # Assert
        assert callable(register_decorator_factory)
        assert callable(unregister_decorator_factory)
        assert callable(inject_adapter_id_decorator_factory)
        # Verify they are different functions
        assert register_decorator_factory != unregister_decorator_factory
        assert register_decorator_factory != inject_adapter_id_decorator_factory
        assert unregister_decorator_factory != inject_adapter_id_decorator_factory

    @patch(
        "model_hosting_container_standards.common.transforms.base_factory._resolve_transforms"
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_factory.handler_registry"
    )
    @patch("model_hosting_container_standards.common.transforms.base_factory.logger")
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
        decorator_factory = create_lora_transform_decorator(handler_type)
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
        "model_hosting_container_standards.sagemaker.lora.factory.resolve_lora_transform"
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_factory.handler_registry"
    )
    @patch("model_hosting_container_standards.common.transforms.base_factory.logger")
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
        mock_transformer.intercept = AsyncMock(
            return_value={"original": "Processing test-adapter from s3://test"}
        )
        mock_transformer.transform_response = Mock(
            return_value={"status": "registered"}
        )

        # Define a test handler function
        @create_lora_transform_decorator(handler_type)(request_shape, response_shape)
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
        mock_transformer.intercept.assert_called_once()
        mock_transformer.transform_response.assert_called_once()

        # Verify handler registration
        mock_registry.set_handler.assert_called_once()
        call_args = mock_registry.set_handler.call_args
        assert call_args[0][0] == handler_type
        # Verify HandlerInfo was registered
        handler_info = call_args[0][1]
        assert callable(handler_info.func)
        assert handler_info.route_kwargs == {}

        # Verify final result
        assert result == {"status": "registered"}
