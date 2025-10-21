"""Unit tests for LoRA API transform classes."""

from http import HTTPStatus
from unittest.mock import Mock, patch

import jmespath
from fastapi import Request, Response
from pydantic import BaseModel

from model_hosting_container_standards.sagemaker.lora.base_lora_api_transform import (
    BaseLoRAApiTransform,
)
from model_hosting_container_standards.sagemaker.lora.models.transform import (
    BaseLoRATransformRequestOutput,
)


class MockRequestModel(BaseModel):
    """Mock request model for testing."""

    name: str
    src: str
    preload: bool = True


class ConcreteLoRAApiTransform(BaseLoRAApiTransform):
    """Concrete implementation for testing abstract base class."""

    async def transform_request(self, request, raw_request):
        """Mock implementation of abstract method."""
        transformed_request = self._transform_request(request, raw_request)
        return BaseLoRATransformRequestOutput(
            request=transformed_request,
            raw_request=raw_request,
            adapter_name=getattr(request, "name", None) if request else None,
        )

    def _transform_ok_response(self, response: Response, **kwargs):
        """Mock implementation of abstract method."""
        adapter_name = kwargs.get("adapter_name")
        adapter_alias = kwargs.get("adapter_alias")
        return Response(
            status_code=HTTPStatus.OK,
            content=f"Success for adapter: {adapter_name} (alias: {adapter_alias})",
        )

    def _transform_error_response(self, response: Response, **kwargs):
        """Mock implementation of abstract method."""
        adapter_name = kwargs.get("adapter_name")
        adapter_alias = kwargs.get("adapter_alias")
        return Response(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content=f"Error for adapter: {adapter_name} (alias: {adapter_alias})",
        )


class TestBaseLoRATransformRequestOutput:
    """Test BaseLoRATransformRequestOutput model."""

    def test_create_with_all_fields(self):
        """Test creating output with all fields."""
        mock_request = {"key": "value"}
        mock_raw_request = Mock(spec=Request)
        adapter_name = "test-adapter"

        output = BaseLoRATransformRequestOutput(
            request=mock_request,
            raw_request=mock_raw_request,
            adapter_name=adapter_name,
        )

        assert output.request == mock_request
        assert output.raw_request == mock_raw_request
        assert output.adapter_name == adapter_name

    def test_create_with_optional_fields_none(self):
        """Test creating output with optional fields as None."""
        output = BaseLoRATransformRequestOutput()

        assert output.request is None
        assert output.raw_request is None
        assert output.adapter_name is None

    def test_create_with_partial_fields(self):
        """Test creating output with some fields set."""
        mock_raw_request = Mock(spec=Request)

        output = BaseLoRATransformRequestOutput(
            raw_request=mock_raw_request, adapter_name="partial-adapter"
        )

        assert output.request is None
        assert output.raw_request == mock_raw_request
        assert output.adapter_name == "partial-adapter"


class TestBaseLoRAApiTransform:
    """Test BaseLoRAApiTransform class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.request_shape = {
            "model": "body.name",
            "source": "body.src",
            "preload_flag": "body.preload",
        }
        self.response_shape = {"status": "status_code", "message": "content"}
        self.transformer = ConcreteLoRAApiTransform(
            self.request_shape, self.response_shape
        )

    def test_init_compiles_jmespath_expressions(self):
        """Test that __init__ compiles JMESPath expressions."""
        # Verify request shape expressions are compiled JMESPath objects
        assert isinstance(
            self.transformer._request_shape["model"], jmespath.parser.ParsedResult
        )
        assert isinstance(
            self.transformer._request_shape["source"], jmespath.parser.ParsedResult
        )
        assert isinstance(
            self.transformer._request_shape["preload_flag"],
            jmespath.parser.ParsedResult,
        )

        # Verify response shape expressions are compiled JMESPath objects
        assert isinstance(
            self.transformer._response_shape["status"], jmespath.parser.ParsedResult
        )
        assert isinstance(
            self.transformer._response_shape["message"], jmespath.parser.ParsedResult
        )

    def test_init_with_empty_response_shape(self):
        """Test initialization with empty response shape."""
        transformer = ConcreteLoRAApiTransform(self.request_shape)

        assert len(transformer._request_shape) == 3
        assert len(transformer._response_shape) == 0

    def test_transform_with_valid_data(self):
        """Test _transform method with valid source data."""
        request_source_data = {
            "body": {"name": "test-model", "src": "s3://bucket/path", "preload": True},
        }

        # Test request transformation
        result = self.transformer._transform(
            request_source_data, self.transformer._request_shape
        )

        expected = {
            "model": "test-model",
            "source": "s3://bucket/path",
            "preload_flag": True,
        }
        assert result == expected

        # Test response transformation

        response_source_data = {"status_code": 200, "content": "success"}
        result = self.transformer._transform(
            response_source_data, self.transformer._response_shape
        )

        expected = {"status": 200, "message": "success"}
        assert result == expected

    def test_transform_with_missing_fields(self):
        """Test _transform method when JMESPath expressions don't match."""
        source_data = {"body": {"other_field": "value"}}

        result = self.transformer._transform(
            source_data, self.transformer._request_shape
        )

        # Missing fields should result in None values
        expected = {"model": None, "source": None, "preload_flag": None}
        assert result == expected

    def test_transform_with_nested_data(self):
        """Test _transform method with nested data structures."""
        request_shape = {
            "nested_value": "body.config.setting",
            "array_value": "body.items[0].name",
        }
        transformer = ConcreteLoRAApiTransform(request_shape)

        source_data = {
            "body": {
                "config": {"setting": "production"},
                "items": [{"name": "first-item"}, {"name": "second-item"}],
            }
        }

        result = transformer._transform(source_data, transformer._request_shape)

        expected = {"nested_value": "production", "array_value": "first-item"}
        assert result == expected

    def test_transform_with_nested_shape_dict(self):
        """Test _transform method with nested shape dictionaries."""
        request_shape = {
            "basic_field": "body.name",
            "nested_object": {
                "inner_field": "body.config.setting",
                "nested_nested_object": {
                    "field_within": "body.without",
                },
                "another_field": "body.config.value",
            },
        }
        transformer = ConcreteLoRAApiTransform(request_shape)

        source_data = {
            "body": {
                "without": False,
                "name": "test-name",
                "config": {"setting": "production", "value": 42},
            }
        }

        result = transformer._transform(source_data, transformer._request_shape)

        expected = {
            "basic_field": "test-name",
            "nested_object": {
                "inner_field": "production",
                "nested_nested_object": {
                    "field_within": False,
                },
                "another_field": 42,
            },
        }
        assert result == expected

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform.serialize_request"
    )
    def test_transform_request_calls_utils(self, mock_get_data):
        """Test that _transform_request calls utility function."""
        mock_request = MockRequestModel(name="test", src="s3://test")
        mock_raw_request = Mock(spec=Request)

        mock_get_data.return_value = {
            "body": {"name": "test", "src": "s3://test", "preload": True}
        }

        result = self.transformer._transform_request(mock_request, mock_raw_request)

        # Verify utility function was called
        mock_get_data.assert_called_once_with(mock_request, mock_raw_request)

        # Verify transformation result
        expected = {"model": "test", "source": "s3://test", "preload_flag": True}
        assert result == expected

    @patch(
        "model_hosting_container_standards.sagemaker.lora.base_lora_api_transform.get_adapter_name_from_request"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.lora.base_lora_api_transform.get_adapter_alias_from_request_header"
    )
    def test_transform_response_ok_status(self, mock_get_alias, mock_get_adapter):
        """Test transform_response with OK status."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = HTTPStatus.OK

        mock_transform_output = Mock(spec=BaseLoRATransformRequestOutput)
        mock_transform_output.raw_request = Mock(spec=Request)
        mock_get_adapter.return_value = "test-adapter"
        mock_get_alias.return_value = "test-alias"

        with patch.object(
            self.transformer, "_transform_ok_response"
        ) as mock_ok_response:
            mock_ok_response.return_value = "ok_result"

            result = self.transformer.transform_response(
                mock_response, mock_transform_output
            )

            mock_get_adapter.assert_called_once_with(mock_transform_output)
            mock_get_alias.assert_called_once_with(mock_transform_output.raw_request)
            mock_ok_response.assert_called_once_with(
                mock_response, adapter_name="test-adapter", adapter_alias="test-alias"
            )
            assert result == "ok_result"

    @patch(
        "model_hosting_container_standards.sagemaker.lora.base_lora_api_transform.get_adapter_name_from_request"
    )
    @patch(
        "model_hosting_container_standards.sagemaker.lora.base_lora_api_transform.get_adapter_alias_from_request_header"
    )
    def test_transform_response_error_status(self, mock_get_alias, mock_get_adapter):
        """Test transform_response with error status."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = HTTPStatus.BAD_REQUEST

        mock_transform_output = Mock(spec=BaseLoRATransformRequestOutput)
        mock_transform_output.raw_request = Mock(spec=Request)
        mock_get_adapter.return_value = "test-adapter"
        mock_get_alias.return_value = "test-alias"

        with patch.object(
            self.transformer, "_transform_error_response"
        ) as mock_error_response:
            mock_error_response.return_value = "error_result"

            result = self.transformer.transform_response(
                mock_response, mock_transform_output
            )

            mock_get_adapter.assert_called_once_with(mock_transform_output)
            mock_get_alias.assert_called_once_with(mock_transform_output.raw_request)
            mock_error_response.assert_called_once_with(
                mock_response, adapter_name="test-adapter", adapter_alias="test-alias"
            )
            assert result == "error_result"

    def test_complex_jmespath_expressions(self):
        """Test with complex JMESPath expressions."""
        complex_shape = {
            "filtered_items": "body.items[?active==`true`].name",
            "first_active": "body.items[?active==`true`] | [0].name",
            "count": "length(body.items)",
        }
        transformer = ConcreteLoRAApiTransform(complex_shape)

        source_data = {
            "body": {
                "items": [
                    {"name": "item1", "active": True},
                    {"name": "item2", "active": False},
                    {"name": "item3", "active": True},
                ]
            }
        }

        result = transformer._transform(source_data, transformer._request_shape)

        expected = {
            "filtered_items": ["item1", "item3"],
            "first_active": "item1",
            "count": 3,
        }
        assert result == expected
