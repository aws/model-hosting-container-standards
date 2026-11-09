"""Test scaffolding for load adapter transform functionality."""

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from model_hosting_container_standards.sagemaker.lora.transforms2.load_adapter import (
    LoadLoraApiTransform,
    SageMakerLoadAdapterRequest,
)


class TestSageMakerLoadAdapterRequest:
    """Test SageMaker load adapter request model."""

    def test_model_validates_with_minimal_params(self):
        """Test model validation with only required parameters."""
        request = SageMakerLoadAdapterRequest(name="adapter-1", src="s3://bucket/path")

        assert request.name == "adapter-1"
        assert request.src == "s3://bucket/path"
        assert request.preload is True  # default
        assert request.pinned is False  # default

    def test_model_validates_with_all_params(self):
        """Test model validation with all parameters."""
        request = SageMakerLoadAdapterRequest(
            name="adapter-1", src="s3://bucket/path", preload=False, pinned=True
        )

        assert request.name == "adapter-1"
        assert request.src == "s3://bucket/path"
        assert request.preload is False
        assert request.pinned is True

    def test_preload_defaults_to_true(self):
        """Test that preload field defaults to True."""
        request = SageMakerLoadAdapterRequest(name="adapter-1", src="s3://bucket/path")
        assert request.preload is True

    def test_pinned_defaults_to_false(self):
        """Test that pinned field defaults to False."""
        request = SageMakerLoadAdapterRequest(name="adapter-1", src="s3://bucket/path")
        assert request.pinned is False


class TestLoadLoraApiTransformInitialization:
    """Test LoadLoraApiTransform initialization.

    Note: Generic initialization tests are in test_base_lora_api_transform2.py.
    """

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters specific to load adapter."""
        from unittest.mock import Mock

        mock_func = Mock()
        engine_paths = {"name": "body.name", "src": "body.src"}
        model_cls = Mock(spec=BaseModel)

        transform = LoadLoraApiTransform(mock_func, engine_paths, model_cls, None)

        assert transform.original_function == mock_func
        assert transform.engine_request_paths == engine_paths
        assert transform.engine_request_model_cls == model_cls

    def test_init_with_engine_request_defaults(self):
        """Test initialization with engine_request_defaults parameter."""
        from unittest.mock import Mock

        mock_func = Mock()
        engine_paths = {"name": "body.name", "src": "body.src"}
        defaults = {"body.timeout": "30"}

        transform = LoadLoraApiTransform(
            mock_func, engine_paths, Mock(spec=BaseModel), defaults
        )

        assert transform.engine_request_defaults == defaults


class TestLoadLoraApiTransformValidateSageMakerParams:
    """Test validation of SageMaker parameters."""

    def test_validate_with_valid_params(self):
        """Test validation with all valid parameter names."""
        from unittest.mock import Mock

        mock_func = Mock()
        transform = LoadLoraApiTransform(
            mock_func,
            {"name": "body.name", "src": "body.src"},
            Mock(spec=BaseModel),
            None,
        )

        # These should not raise
        for param in ["name", "src", "preload", "pinned"]:
            transform._init_validate_sagemaker_params(param)

    def test_validate_with_invalid_param_raises_error(self):
        """Test that invalid parameter name raises ValueError."""
        from unittest.mock import Mock

        mock_func = Mock()
        transform = LoadLoraApiTransform(
            mock_func, {"name": "body.name"}, Mock(spec=BaseModel), None
        )

        with pytest.raises(ValueError, match="Invalid sagemaker_param: invalid_field"):
            transform._init_validate_sagemaker_params("invalid_field")

    def test_error_message_includes_allowed_values(self):
        """Test that error message includes list of allowed parameter names."""
        from unittest.mock import Mock

        mock_func = Mock()
        transform = LoadLoraApiTransform(
            mock_func, {"name": "body.name"}, Mock(spec=BaseModel), None
        )

        with pytest.raises(ValueError) as exc_info:
            transform._init_validate_sagemaker_params("bad_param")

        error_msg = str(exc_info.value)
        assert "Allowed value(s):" in error_msg


class TestLoadLoraApiTransformValidateRequest:
    """Test request validation logic for load adapter."""

    async def test_validate_request_with_valid_query_params(self):
        """Test validation with valid query parameters."""
        # TODO: Implement test

    async def test_validate_request_extracts_all_fields(self):
        """Test that all fields (name, src, preload, pinned) are correctly extracted."""
        # TODO: Implement test

    async def test_validate_request_without_query_params_raises_error(self):
        """Test that missing query parameters raises HTTPException."""
        # TODO: Implement test

    async def test_validate_request_with_missing_required_param_raises_error(self):
        """Test that missing required parameter (name or src) raises HTTPException."""
        # TODO: Implement test


class TestLoadLoraApiTransformGenerateResponse:
    """Test response generation logic for load adapter."""

    def test_generate_successful_response_with_adapter_alias(self):
        """Test response generation when adapter alias is present."""
        from unittest.mock import Mock

        from model_hosting_container_standards.common.transforms.base_api_transform2 import (
            BaseTransformRequestOutput,
        )

        mock_func = Mock()
        transform = LoadLoraApiTransform(
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
        assert "registered" in result.lower()

    def test_generate_successful_response_without_adapter_alias(self):
        """Test response generation when adapter alias is absent (uses name)."""
        from unittest.mock import Mock

        from model_hosting_container_standards.common.transforms.base_api_transform2 import (
            BaseTransformRequestOutput,
        )

        mock_func = Mock()
        transform = LoadLoraApiTransform(
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
        assert "registered" in result.lower()


class TestLoadLoraApiTransformIntegration:
    """Test end-to-end load adapter transformation."""

    async def test_full_transform_flow_with_valid_request(self):
        """Test complete transformation flow with valid request."""
        # TODO: Implement test

    async def test_transform_with_engine_defaults(self):
        """Test transformation with engine request defaults applied."""
        # TODO: Implement test

    async def test_transform_handles_validation_errors(self):
        """Test that transformation properly handles validation errors."""
        # TODO: Implement test
