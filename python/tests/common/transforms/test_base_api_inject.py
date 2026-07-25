import json
from http import HTTPStatus
from typing import Any, Dict, Union
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from model_hosting_container_standards.common.transforms.base_api_inject import (
    BaseApiInject,
    BaseInjectRequestOutput,
    BaseInjectValidateOutput,
    InjectDefinition,
    SageMakerInjectMode,
)


class MockEngineRequest(BaseModel):
    prompt: str
    model: str


class ConcreteRequest(BaseModel):
    model_name: str


async def dummy_original_function_with_model(
    request: MockEngineRequest, raw_request: Request
):
    return Response(status_code=200, content=json.dumps(request.model_dump()))


class ConcreteApiInject(BaseApiInject):
    def _init_validate_sagemaker_params(self, sagemaker_param: str):
        if sagemaker_param not in ConcreteRequest.model_fields.keys():
            raise ValueError(f"Invalid sagemaker param: {sagemaker_param}")

    async def validate_request_should_inject(
        self, raw_request: Request
    ) -> BaseInjectValidateOutput:
        try:
            body = await raw_request.json()
            return BaseInjectValidateOutput(
                should_inject=True,
                request_body=body,
                sagemaker_values=ConcreteRequest.model_validate(body),
            )
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Invalid JSON in request body",
            )
        except ValidationError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))

    def _extract_additional_fields(
        self,
        sagemaker_values: Union[BaseModel, Dict[str, Any]],
        request_body: Dict[str, Any],
        raw_request: Request,
    ) -> Dict[str, Any]:
        return {}


class TestBaseInjectRequestOutput:
    """Test BaseInjectRequestOutput model."""

    def test_model_validates_with_minimal_fields(self):
        """Test model validation with only raw_request."""
        mock_request = Mock(spec=Request)
        output = BaseInjectRequestOutput(raw_request=mock_request)

        assert output.raw_request == mock_request
        assert output.request_body is None
        assert output.additional_fields == {}

    def test_model_validates_with_all_fields(self):
        """Test model validation with all fields."""
        mock_request = Mock(spec=Request)
        body = {"key": "value"}
        fields = {"field1": "value1"}

        output = BaseInjectRequestOutput(
            raw_request=mock_request, request_body=body, additional_fields=fields
        )

        assert output.raw_request == mock_request
        assert output.request_body == body
        assert output.additional_fields == fields


class TestBaseInjectValidateOutput:
    """Test BaseInjectValidateOutput model."""

    def test_model_validates_with_should_inject_false(self):
        """Test model validation when should_inject is False."""
        output = BaseInjectValidateOutput(
            should_inject=False, request_body={"key": "value"}, sagemaker_values=None
        )

        assert output.should_inject is False
        assert output.request_body == {"key": "value"}
        assert output.sagemaker_values is None

    def test_model_validates_with_should_inject_true(self):
        """Test model validation when should_inject is True."""
        # Use ConcreteRequest which is a proper BaseModel subclass
        values = ConcreteRequest(model_name="test-model")
        output = BaseInjectValidateOutput(
            should_inject=True, request_body={"key": "value"}, sagemaker_values=values
        )

        assert output.should_inject is True
        assert output.request_body == {"key": "value"}
        assert output.sagemaker_values == values


class TestInjectDefinition:
    """Test InjectDefinition model."""

    def test_model_validates_with_path_only(self):
        """Test model validation with only path and default mode."""
        definition = InjectDefinition(path="body.field")

        assert definition.path == "body.field"
        assert definition.mode == "replace"
        assert definition.separator is None

    def test_model_validates_with_all_fields(self):
        """Test model validation with all fields."""
        definition = InjectDefinition(path="body.field", mode="append", separator=":")

        assert definition.path == "body.field"
        assert definition.mode == "append"
        assert definition.separator == ":"

    def test_mode_accepts_valid_values(self):
        """Test that mode accepts all valid literal values."""
        for mode in ["append", "prepend", "replace"]:
            definition = InjectDefinition(path="test", mode=mode)
            assert definition.mode == mode


class TestBaseApiInjectInitialization:
    """Test BaseApiInject initialization and validation."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        mock_func = Mock()
        # Use 'model_name' which is a valid field in ConcreteRequest
        inject_defs = {"model_name": InjectDefinition(path="body.field")}

        inject = ConcreteApiInject(mock_func, inject_defs, MockEngineRequest, None)

        assert inject.original_function == mock_func
        assert inject.engine_request_inject_definitions == inject_defs
        assert inject.engine_request_model_cls == MockEngineRequest
        assert inject.engine_request_defaults == {}

    def test_init_with_empty_inject_definitions(self):
        """Test initialization with empty inject definitions."""
        mock_func = Mock()

        inject = ConcreteApiInject(mock_func, {}, None, None)

        assert inject.engine_request_inject_definitions == {}

    def test_init_validates_empty_path_raises_error(self):
        """Test that empty path in inject definition raises ValueError."""
        mock_func = Mock()
        inject_defs = {"param1": InjectDefinition(path="")}

        with pytest.raises(ValueError, match="empty string"):
            ConcreteApiInject(mock_func, inject_defs, None, None)

    def test_init_validates_invalid_jmespath_raises_error(self):
        """Test that invalid JMESPath expression raises ValueError."""
        mock_func = Mock()
        inject_defs = {"param1": InjectDefinition(path="body.[invalid")}

        with pytest.raises(ValueError, match="not a valid JMESPath"):
            ConcreteApiInject(mock_func, inject_defs, None, None)


class TestBaseApiInjectRequestInjection:
    """Test request injection functionality."""

    @pytest.mark.parametrize(
        "values,description",
        [
            ({"single": "value"}, "single value"),
            ({"key1": "val1", "key2": "val2"}, "multiple values"),
            ({"key": None}, "None values (should skip)"),
        ],
    )
    def test_inject_sagemaker_to_engine_with_values(self, values, description):
        """Test injecting SageMaker values to engine request."""
        # TODO: Implement test

    def test_inject_sagemaker_to_engine_with_nested_path(self):
        """Test injection to nested JMESPath location."""
        # TODO: Implement test

    def test_inject_sagemaker_to_engine_with_replace_mode(self):
        """Test injection with replace mode."""
        # TODO: Implement test

    def test_inject_sagemaker_to_engine_with_append_mode(self):
        """Test injection with append mode."""
        # TODO: Implement test

    def test_inject_sagemaker_to_engine_with_prepend_mode(self):
        """Test injection with prepend mode."""
        # TODO: Implement test

    def test_apply_to_raw_request_updates_headers(self):
        """Test that headers are updated in raw request."""
        # TODO: Implement test

    def test_apply_to_raw_request_updates_query_params(self):
        """Test that query params are updated in raw request."""
        # TODO: Implement test

    def test_apply_to_raw_request_updates_body(self):
        """Test that body is updated in raw request."""
        # TODO: Implement test

    def test_apply_to_raw_request_updates_path_params(self):
        """Test that path params are updated in raw request."""
        # TODO: Implement test

    def test_inject_engine_defaults_applies_defaults(self):
        """Test that engine defaults are applied."""
        # TODO: Implement test

    def test_inject_engine_defaults_with_no_defaults(self):
        """Test behavior when no defaults are configured."""
        # TODO: Implement test

    @pytest.mark.parametrize("has_values", [True, False])
    def test_inject_request_with_or_without_sagemaker_values(self, has_values):
        """Test inject_request with and without sagemaker_values."""
        # TODO: Implement test

    def test_inject_request_extracts_additional_fields(self):
        """Test that additional fields are extracted during injection."""
        # TODO: Implement test

    def test_inject_request_converts_basemodel_to_dict(self):
        """Test that BaseModel sagemaker_values are converted to dict."""
        # TODO: Implement test


class TestBaseApiInjectFunctionCalling:
    """Test function calling with injected requests."""

    async def test_call_with_request_model_validation(self):
        """Test calling function with request model validation."""
        # TODO: Implement test

    async def test_call_without_request_model_validation(self):
        """Test calling function without request model validation."""
        # TODO: Implement test

    async def test_call_with_validation_error_raises_http_exception(self):
        """Test that validation errors raise HTTPException."""
        # TODO: Implement test

    async def test_call_uses_provided_function(self):
        """Test that provided function is called."""
        # TODO: Implement test

    async def test_call_uses_original_function_when_none_provided(self):
        """Test that original function is used when func is None."""
        # TODO: Implement test

    async def test_call_uses_provided_request_model_cls(self):
        """Test that provided request_model_cls is used."""
        # TODO: Implement test

    async def test_call_uses_engine_request_model_cls_when_none_provided(self):
        """Test that engine_request_model_cls is used when request_model_cls is None."""
        # TODO: Implement test

    async def test_call_passes_requests_to_function(self):
        """Test that both validated and raw requests are passed to function."""
        # TODO: Implement test


class TestBaseApiInjectIntegrationFlow:
    """Test end-to-end injection flow."""

    async def test_inject_full_flow_with_injection(self):
        """Test complete injection flow when injection is needed."""
        # TODO: Implement test

    async def test_inject_full_flow_without_injection(self):
        """Test complete injection flow when injection is not needed."""
        # TODO: Implement test

    async def test_inject_calls_full_pipeline(self):
        """Test that inject calls validate_request_should_inject → inject_request → call in sequence."""
        # TODO: Implement test

    async def test_inject_handles_http_exceptions(self):
        """Test that HTTPExceptions are properly handled and re-raised."""
        # TODO: Implement test

    async def test_inject_handles_unexpected_exceptions(self):
        """Test that unexpected exceptions are caught and wrapped."""
        # TODO: Implement test

    async def test_inject_logs_errors(self):
        """Test that errors are logged during injection."""
        # TODO: Implement test


class TestBaseApiInjectAbstractMethods:
    """Test abstract method requirements."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "_init_validate_sagemaker_params",
            "validate_request_should_inject",
            "_extract_additional_fields",
        ],
    )
    def test_abstract_methods_must_be_implemented(self, method_name):
        """Test that abstract methods must be implemented by subclasses."""
        # TODO: Implement test

    def test_concrete_implementation_works(self):
        """Test that concrete implementation with all abstract methods works."""
        # TODO: Implement test
