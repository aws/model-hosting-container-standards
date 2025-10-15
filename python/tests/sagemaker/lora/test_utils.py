"""Unit tests for LoRA utils module."""

from unittest.mock import Mock

from fastapi import Request
from pydantic import BaseModel

from model_hosting_container_standards.fastapi.utils import serialize_request
from model_hosting_container_standards.sagemaker.lora.constants import (
    RequestField,
    SageMakerLoRAApiHeader,
)
from model_hosting_container_standards.sagemaker.lora.models.transform import (
    BaseLoRATransformRequestOutput,
)
from model_hosting_container_standards.sagemaker.lora.utils import (
    get_adapter_alias_from_request_header,
    get_adapter_name_from_request,
    get_adapter_name_from_request_path,
)


class MockRequestModel(BaseModel):
    """Mock request model for testing."""

    name: str
    src: str
    preload: bool = True


class TestGetRequestDataForJmespath:
    """Test serialize_request function."""

    def test_with_request_model_and_raw_request(self):
        """Test function with both request model and raw request."""
        # Setup mock request model
        request_model = MockRequestModel(name="test-adapter", src="s3://bucket/path")

        # Setup mock raw request
        raw_request = Mock(spec=Request)
        raw_request.headers = {SageMakerLoRAApiHeader.ADAPTER_ALIAS: "test-alias"}
        raw_request.query_params = {"test": "test_query_param"}
        raw_request.path_params = {"adapter_name": "path-adapter"}

        # Call function
        result = serialize_request(request_model, raw_request)

        # Verify result structure
        assert "body" in result
        assert "headers" in result
        assert "query_params" in result
        assert "path_params" in result

        # Verify body content
        expected_body = {
            "name": "test-adapter",
            "src": "s3://bucket/path",
            "preload": True,
        }
        assert result["body"] == expected_body

        # Verify other components
        assert result["headers"] == raw_request.headers
        assert result["query_params"] == raw_request.query_params
        assert result["path_params"] == raw_request.path_params

    def test_with_none_request_model(self):
        """Test function when request model is None."""
        # Setup mock raw request
        raw_request = Mock(spec=Request)
        raw_request.headers = {"X-Test": "value"}
        raw_request.query_params = {"param": "value"}
        raw_request.path_params = {"id": "123"}

        # Call function with None request
        result = serialize_request(None, raw_request)

        # Verify body is None but other fields are preserved
        assert result["body"] is None
        assert result["headers"] == raw_request.headers
        assert result["query_params"] == raw_request.query_params
        assert result["path_params"] == raw_request.path_params

    def test_with_empty_raw_request_data(self):
        """Test function with empty raw request data."""
        request_model = MockRequestModel(name="test", src="s3://test")

        raw_request = Mock(spec=Request)
        raw_request.headers = {}
        raw_request.query_params = {}
        raw_request.path_params = {}

        result = serialize_request(request_model, raw_request)

        assert result["body"] == {"name": "test", "src": "s3://test", "preload": True}
        assert result["headers"] == {}
        assert result["query_params"] == {}
        assert result["path_params"] == {}

    def test_with_dict_request(self):
        """Test function when request is a dictionary."""
        request_dict = {
            "name": "dict-adapter",
            "src": "s3://bucket/dict-path",
            "preload": False,
        }

        raw_request = Mock(spec=Request)
        raw_request.headers = {"X-Custom": "header-value"}
        raw_request.query_params = {"param": "query-value"}
        raw_request.path_params = {"id": "path-value"}

        result = serialize_request(request_dict, raw_request)

        # Verify body is the dictionary as-is
        assert result["body"] == request_dict
        assert result["headers"] == raw_request.headers
        assert result["query_params"] == raw_request.query_params
        assert result["path_params"] == raw_request.path_params

    def test_with_unsupported_request_type(self):
        """Test function with unsupported request type."""
        unsupported_request = "string-request"  # Not BaseModel or dict

        raw_request = Mock(spec=Request)
        raw_request.headers = {}
        raw_request.query_params = {}
        raw_request.path_params = {}

        result = serialize_request(unsupported_request, raw_request)

        # Should set body to None for unsupported types
        assert result["body"] is None
        assert result["headers"] == raw_request.headers
        assert result["query_params"] == raw_request.query_params
        assert result["path_params"] == raw_request.path_params


class TestGetAdapterNameFromRequest:
    """Test get_adapter_name_from_request function."""

    def create_mock_transform_output(self, raw_request=None, adapter_name=None):
        """Helper to create mock transform request output."""
        output = Mock(spec=BaseLoRATransformRequestOutput)
        output.raw_request = raw_request or Mock(spec=Request)
        output.adapter_name = adapter_name
        return output

    def test_path_param_adapter_name_priority(self):
        """Test that path parameter adapter_name has second priority."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER: "header-id"}
        raw_request.path_params = {RequestField.ADAPTER_NAME: "path-adapter"}

        transform_output = self.create_mock_transform_output(
            raw_request=raw_request, adapter_name="output-adapter"
        )

        result = get_adapter_name_from_request(transform_output)
        assert result == "path-adapter"

    def test_transform_output_adapter_name_second_priority(self):
        """Test that transform output adapter_name has third priority."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER: "header-id"}
        raw_request.path_params = {}

        transform_output = self.create_mock_transform_output(
            raw_request=raw_request, adapter_name="output-adapter"
        )

        result = get_adapter_name_from_request(transform_output)
        assert result == "output-adapter"

    def test_adapter_identifier_header_fallback(self):
        """Test that adapter identifier header is used as fallback."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {
            SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER: "identifier-fallback"
        }
        raw_request.path_params = {}

        transform_output = self.create_mock_transform_output(
            raw_request=raw_request, adapter_name=None
        )

        result = get_adapter_name_from_request(transform_output)
        assert result == "identifier-fallback"

    def test_returns_none_when_no_adapter_found(self):
        """Test that None is returned when no adapter name is found."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {}
        raw_request.path_params = {}

        transform_output = self.create_mock_transform_output(
            raw_request=raw_request, adapter_name=None
        )

        result = get_adapter_name_from_request(transform_output)
        assert result is None


class TestGetAdapterAliasFromRequestHeader:
    """Test get_adapter_alias_from_request_header function."""

    def test_gets_adapter_alias_from_headers(self):
        """Test that adapter alias is extracted from headers."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {
            SageMakerLoRAApiHeader.ADAPTER_ALIAS: "test-alias",
            "other-header": "other-value",
        }

        result = get_adapter_alias_from_request_header(raw_request)
        assert result == "test-alias"

    def test_returns_none_when_no_alias_header(self):
        """Test that None is returned when no alias header is present."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {
            "other-header": "other-value",
            SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER: "identifier",
        }

        result = get_adapter_alias_from_request_header(raw_request)
        assert result is None

    def test_returns_none_when_alias_header_empty(self):
        """Test that None is returned when alias header is empty."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {
            SageMakerLoRAApiHeader.ADAPTER_ALIAS: "",
            "other-header": "other-value",
        }

        result = get_adapter_alias_from_request_header(raw_request)
        assert result is None

    def test_returns_none_when_no_headers(self):
        """Test that None is returned when headers is None."""
        raw_request = Mock(spec=Request)
        raw_request.headers = None

        result = get_adapter_alias_from_request_header(raw_request)
        assert result is None

    def test_returns_none_when_headers_empty(self):
        """Test that None is returned when headers dict is empty."""
        raw_request = Mock(spec=Request)
        raw_request.headers = {}

        result = get_adapter_alias_from_request_header(raw_request)
        assert result is None


class TestGetAdapterNameFromRequestPath:
    """Test get_adapter_name_from_request_path function."""

    def test_gets_adapter_name_from_path_params(self):
        """Test that adapter name is extracted from path parameters."""
        raw_request = Mock(spec=Request)
        raw_request.path_params = {
            RequestField.ADAPTER_NAME: "path-adapter",
            "other-param": "other-value",
        }

        result = get_adapter_name_from_request_path(raw_request)
        assert result == "path-adapter"

    def test_returns_none_when_no_adapter_name_param(self):
        """Test that None is returned when no adapter_name path param is present."""
        raw_request = Mock(spec=Request)
        raw_request.path_params = {"other-param": "other-value", "id": "123"}

        result = get_adapter_name_from_request_path(raw_request)
        assert result is None

    def test_returns_none_when_adapter_name_param_empty(self):
        """Test that None is returned when adapter_name path param is empty."""
        raw_request = Mock(spec=Request)
        raw_request.path_params = {
            RequestField.ADAPTER_NAME: "",
            "other-param": "other-value",
        }

        result = get_adapter_name_from_request_path(raw_request)
        assert result is None

    def test_returns_none_when_no_path_params(self):
        """Test that None is returned when path_params is None."""
        raw_request = Mock(spec=Request)
        raw_request.path_params = None

        result = get_adapter_name_from_request_path(raw_request)
        assert result is None

    def test_returns_none_when_path_params_empty(self):
        """Test that None is returned when path_params dict is empty."""
        raw_request = Mock(spec=Request)
        raw_request.path_params = {}

        result = get_adapter_name_from_request_path(raw_request)
        assert result is None
