"""Test scaffolding for unload adapter transform functionality."""

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from model_hosting_container_standards.sagemaker.lora.transforms2.unload_adapter import (
    SageMakerUnloadAdapterRequest,
    UnloadLoraApiTransform,
)


class TestSageMakerUnloadAdapterRequest:
    """Test SageMaker unload adapter request model."""

    def test_model_validates_with_name(self):
        """Test model validation with name parameter."""
        request = SageMakerUnloadAdapterRequest(name="adapter-1")
        assert request.name == "adapter-1"

    def test_model_validation_fails_without_name(self):
        """Test that validation fails when name is missing."""
        with pytest.raises(ValidationError):
            SageMakerUnloadAdapterRequest()


class TestUnloadLoraApiTransformInitialization:
    """Test UnloadLoraApiTransform initialization.

    Note: Generic initialization tests are in test_base_lora_api_transform2.py.
    """

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters specific to unload adapter."""
        from unittest.mock import Mock

        mock_func = Mock()
        engine_paths = {"name": "body.name"}
        model_cls = Mock(spec=BaseModel)

        transform = UnloadLoraApiTransform(mock_func, engine_paths, model_cls, None)

        assert transform.original_function == mock_func
        assert transform.engine_request_paths == engine_paths

    def test_init_with_engine_request_defaults(self):
        """Test initialization with engine_request_defaults parameter."""
        from unittest.mock import Mock

        mock_func = Mock()
        defaults = {"body.timeout": "30"}

        transform = UnloadLoraApiTransform(
            mock_func, {"name": "body.name"}, Mock(spec=BaseModel), defaults
        )

        assert transform.engine_request_defaults == defaults


class TestUnloadLoraApiTransformValidateSageMakerParams:
    """Test validation of SageMaker parameters."""

    def test_validate_with_valid_param_name(self):
        """Test validation with valid parameter name (name)."""
        from unittest.mock import Mock

        mock_func = Mock()
        transform = UnloadLoraApiTransform(
            mock_func, {"name": "body.name"}, Mock(spec=BaseModel), None
        )

        # Should not raise
        transform._init_validate_sagemaker_params("name")

    def test_validate_with_invalid_param_raises_error(self):
        """Test that invalid parameter name raises ValueError."""
        from unittest.mock import Mock

        mock_func = Mock()
        transform = UnloadLoraApiTransform(
            mock_func, {"name": "body.name"}, Mock(spec=BaseModel), None
        )

        with pytest.raises(ValueError, match="Invalid sagemaker_param"):
            transform._init_validate_sagemaker_params("invalid_field")

    def test_error_message_includes_allowed_values(self):
        """Test that error message includes list of allowed parameter names."""
        from unittest.mock import Mock

        mock_func = Mock()
        transform = UnloadLoraApiTransform(
            mock_func, {"name": "body.name"}, Mock(spec=BaseModel), None
        )

        with pytest.raises(ValueError) as exc_info:
            transform._init_validate_sagemaker_params("bad_param")

        assert "Allowed value(s):" in str(exc_info.value)


class TestUnloadLoraApiTransformValidateRequest:
    """Test request validation logic for unload adapter."""

    async def test_validate_request_extracts_name_from_path(self):
        """Test that adapter name is correctly extracted from request path."""
        # TODO: Implement test

    async def test_validate_request_with_valid_path_param(self):
        """Test validation with valid path parameter."""
        # TODO: Implement test

    async def test_validate_request_with_missing_path_param_raises_error(self):
        """Test that missing path parameter raises HTTPException."""
        # TODO: Implement test

    async def test_validate_request_handles_key_error(self):
        """Test handling of KeyError during path extraction."""
        # TODO: Implement test

    async def test_validate_request_handles_validation_error(self):
        """Test handling of ValidationError during model validation."""
        # TODO: Implement test


class TestUnloadLoraApiTransformGenerateResponse:
    """Test response generation logic for unload adapter."""

    def test_generate_successful_response_with_adapter_alias(self):
        """Test response generation when adapter alias is present."""
        from unittest.mock import Mock

        from model_hosting_container_standards.common.transforms.base_api_transform2 import (
            BaseTransformRequestOutput,
        )

        mock_func = Mock()
        transform = UnloadLoraApiTransform(
            mock_func, {"name": "body.name"}, Mock(spec=BaseModel), None
        )

        mock_response = Mock(spec=Response)
        transform_output = BaseTransformRequestOutput(
            raw_request=Mock(),
            request={"name": "adapter-1"},
            additional_fields={
                "adapter_name": "adapter-1",
                "adapter_alias": "my-alias",
            },
        )

        result = transform._generate_successful_response_content(
            mock_response, transform_output
        )

        assert "my-alias" in result
        assert "unregistered" in result.lower()

    def test_generate_successful_response_without_adapter_alias(self):
        """Test response generation when adapter alias is absent (uses name)."""
        from unittest.mock import Mock

        from model_hosting_container_standards.common.transforms.base_api_transform2 import (
            BaseTransformRequestOutput,
        )

        mock_func = Mock()
        transform = UnloadLoraApiTransform(
            mock_func, {"name": "body.name"}, Mock(spec=BaseModel), None
        )

        mock_response = Mock(spec=Response)
        transform_output = BaseTransformRequestOutput(
            raw_request=Mock(),
            request={"name": "adapter-1"},
            additional_fields={"adapter_name": "adapter-1", "adapter_alias": None},
        )

        result = transform._generate_successful_response_content(
            mock_response, transform_output
        )

        assert "adapter-1" in result
        assert "unregistered" in result.lower()


class TestUnloadLoraApiTransformIntegration:
    """Test end-to-end unload adapter transformation."""

    async def test_full_transform_flow_with_valid_request(self):
        """Test complete transformation flow with valid request."""
        # TODO: Implement test

    async def test_transform_with_engine_defaults(self):
        """Test transformation with engine request defaults applied."""
        # TODO: Implement test

    async def test_transform_handles_validation_errors(self):
        """Test that transformation properly handles validation errors."""
        # TODO: Implement test
