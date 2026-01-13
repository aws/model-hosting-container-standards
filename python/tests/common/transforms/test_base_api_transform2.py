"""Unit tests for common.transforms.base_api_transform2 module."""

import json
from http import HTTPStatus
from typing import Any, Dict
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from model_hosting_container_standards.common.transforms.base_api_transform2 import (
    BaseApiTransform2,
    BaseTransformRequestOutput,
    set_value,
)


class MockValidatedRequest(BaseModel):
    """Mock validated request model for testing."""

    param1: str


class MockEngineRequest(BaseModel):
    """Mock engine request model for testing."""

    engine_param1: str
    engine_param2: int = 5


async def dummy_original_function_with_model(
    engine_request: MockEngineRequest, raw_request: Request
):
    """Mock original function for testing."""
    return Response(
        status_code=HTTPStatus.OK.value,
        content=json.dumps({"message": "Success"}),
    )


async def original_function_with_model_exception(
    engine_request: MockEngineRequest, raw_request: Request
):
    """Mock original function that raises an error for testing."""
    raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="Bad Request")


async def dummy_original_function_without_model(raw_request: Request):
    """Mock original function for testing."""
    return Response(
        status_code=HTTPStatus.OK.value,
        content=json.dumps({"message": "Success"}),
    )


async def original_function_with_exception(raw_request: Request):
    """Mock original function that raises an error for testing."""
    raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="Bad Request")


class ConcreteApiTransform(BaseApiTransform2):
    """Concrete implementation of BaseApiTransform2 for testing."""

    async def validate_request(self, raw_request: Request) -> BaseModel:
        """Mock implementation of validate_request."""
        body = json.loads(raw_request._body.decode())
        return MockValidatedRequest.model_validate(body)

    def _extract_additional_fields(
        self, validated_request: BaseModel, raw_request: Request
    ) -> Dict[str, Any]:
        """Mock implementation of _extract_additional_fields."""
        return {"additional_field": "value"}

    def _init_validate_sagemaker_params(self, sagemaker_param: str) -> None:
        if sagemaker_param not in MockValidatedRequest.model_fields.keys():
            raise ValueError(f"Invalid sagemaker_param: {sagemaker_param}")


class TestBaseTransformRequestOutput:
    """Test suite for BaseTransformRequestOutput Pydantic model."""

    def test_creation_with_all_fields(self):
        """
        Test creating BaseTransformRequestOutput with all fields populated.
        """
        test_dict = {
            "raw_request": Request(scope={"type": "http"}),
            "transformed_request": {"param1": "value1"},
            "additional_fields": {"field1": "value1", "field2": 42},
        }
        transform_request_output = BaseTransformRequestOutput.model_validate(test_dict)
        assert transform_request_output.model_dump() == test_dict

    def test_creation_with_minimal_fields(self):
        """
        Test creating BaseTransformRequestOutput with only required fields.
        """
        test_dict = {"raw_request": Request(scope={"type": "http"})}
        transform_request_output = BaseTransformRequestOutput.model_validate(test_dict)
        assert transform_request_output.model_dump() == {
            "raw_request": test_dict.get("raw_request"),
            "transformed_request": None,
            "additional_fields": {},
        }

    def test_field_validation(self):
        """
        Test field validation for BaseTransformRequestOutput.
        """
        bad_test_dict_0 = {}
        with pytest.raises(ValidationError) as e_0:
            # Missing required field `raw_request`
            BaseTransformRequestOutput.model_validate(bad_test_dict_0)
        assert e_0.value.errors()[0]["loc"] == ("raw_request",)

        bad_test_dict_1 = {
            "raw_request": Request(scope={"type": "http"}),
            "transformed_request": "not_a_dict",
        }
        with pytest.raises(ValidationError) as e_1:
            # `transformed_request` must be a dictionary or None
            BaseTransformRequestOutput.model_validate(bad_test_dict_1)
        assert e_1.value.errors()[0]["loc"] == ("transformed_request",)

        bad_test_dict_2 = {
            "raw_request": Request(scope={"type": "http"}),
            "additional_fields": "not_a_dict",
        }
        with pytest.raises(ValidationError) as e_2:
            # `additional_fields` must be a dictionary or None
            BaseTransformRequestOutput.model_validate(bad_test_dict_2)
        assert e_2.value.errors()[0]["loc"] == ("additional_fields",)


class TestBaseApiTransform2Initialization:
    """Test suite for BaseApiTransform2 initialization and configuration."""

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_init_with_all_parameters(self, mock_logger):
        """
        Test initialization with all constructor parameters.

        Should verify:
        - All parameters are properly stored as instance attributes
        - Logger debug messages are generated for paths and defaults
        - No exceptions are raised during initialization
        """
        test_concrete_api_transform = ConcreteApiTransform(
            original_function=dummy_original_function_with_model,
            engine_request_paths={"param1": "body.engine_param1"},
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.engine_param2": 10},
        )

        assert (
            test_concrete_api_transform.original_function
            == dummy_original_function_with_model
        )
        assert test_concrete_api_transform.engine_request_paths == {
            "param1": "body.engine_param1"
        }
        assert test_concrete_api_transform.engine_request_model_cls == MockEngineRequest
        assert test_concrete_api_transform.engine_request_defaults == {
            "body.engine_param2": 10
        }

        # Since engine_request_defaults is defined, logger should log DEBUG level twice
        assert mock_logger.debug.call_count == 2
        mock_logger.debug.assert_any_call(
            "Initialized ConcreteApiTransform with paths: {'param1': 'body.engine_param1'}"
        )
        mock_logger.debug.assert_any_call(
            "Using request defaults: {'body.engine_param2': 10}"
        )

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_init_with_minimal_parameters(self, mock_logger):
        """
        Test initialization with only required parameters.
        """
        test_concrete_api_transform = ConcreteApiTransform(
            original_function=dummy_original_function_with_model,
            engine_request_paths={"param1": "body.engine_param1"},
        )

        assert (
            test_concrete_api_transform.original_function
            == dummy_original_function_with_model
        )
        assert test_concrete_api_transform.engine_request_paths == {
            "param1": "body.engine_param1"
        }
        assert test_concrete_api_transform.engine_request_model_cls is None
        assert test_concrete_api_transform.engine_request_defaults == {}

        assert mock_logger.debug.call_count == 1
        mock_logger.debug.assert_any_call(
            "Initialized ConcreteApiTransform with paths: {'param1': 'body.engine_param1'}"
        )

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_init_with_empty_engine_request_paths(self, mock_logger):
        """
        Test initialization with empty engine_request_paths dictionary.
        """
        test_concrete_api_transform = ConcreteApiTransform(
            original_function=dummy_original_function_with_model,
            engine_request_paths={},
        )

        assert (
            test_concrete_api_transform.original_function
            == dummy_original_function_with_model
        )
        assert test_concrete_api_transform.engine_request_paths == {}
        assert test_concrete_api_transform.engine_request_model_cls is None
        assert test_concrete_api_transform.engine_request_defaults == {}

        assert mock_logger.debug.call_count == 1
        mock_logger.debug.assert_any_call(
            "Initialized ConcreteApiTransform with paths: {}"
        )

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_init_with_none_values(self, mock_logger):
        """
        Test initialization with None values for optional parameters.
        """
        test_concrete_api_transform = ConcreteApiTransform(
            original_function=dummy_original_function_with_model,
            engine_request_paths={},
            engine_request_model_cls=None,
            engine_request_defaults=None,
        )

        assert (
            test_concrete_api_transform.original_function
            == dummy_original_function_with_model
        )
        assert test_concrete_api_transform.engine_request_paths == {}
        assert test_concrete_api_transform.engine_request_model_cls is None
        assert test_concrete_api_transform.engine_request_defaults == {}

        assert mock_logger.debug.call_count == 1
        mock_logger.debug.assert_any_call(
            "Initialized ConcreteApiTransform with paths: {}"
        )

    @pytest.mark.parametrize(
        "invalid_engine_request_paths,error_msg",
        [
            ({"param1": 123}, "Engine path for param1 is not a string: 123"),
            (
                {"param1": ""},
                "Engine path for param1 is an empty string. This is not allowed.",
            ),
            ({"param1": None}, "Engine path for param1 is None. This is not allowed."),
            (
                {"param1": "body..engine_param1"},
                "Engine path for param1 is not a valid JMESPath expression: body..engine_param1",
            ),
        ],
        ids=[
            "non-string-value",
            "empty-string-value",
            "none-value",
            "invalid-jmespath-expression",
        ],
    )
    def test_init_with_invalid_engine_request_paths(
        self, invalid_engine_request_paths, error_msg
    ):
        """
        Test initialization with invalid engine_request_paths values.
        """
        with pytest.raises(ValueError) as e:
            ConcreteApiTransform(
                original_function=dummy_original_function_with_model,
                engine_request_paths=invalid_engine_request_paths,
            )
        assert error_msg in str(e)


class TestBaseApiTransform2RequestTransformation:
    """Test suite for request transformation methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_transform = ConcreteApiTransform(
            original_function=dummy_original_function_with_model,
            engine_request_paths={"param1": "body.engine_param1"},
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.engine_param2": 10},
        )

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.set_value",
        wraps=set_value,
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_transform_sagemaker_request_to_engine_basic(
        self, mock_logger, mock_set_value
    ):
        """
        Test basic SageMaker to engine request transformation.
        """
        test_sagemaker_request_dict = {"param1": "value1"}
        expected = {"body": {"engine_param1": "value1"}}
        actual = self.api_transform._transform_sagemaker_request_to_engine(
            transformed_request={"body": {}},
            sagemaker_request_dict=test_sagemaker_request_dict,
        )
        assert actual == expected
        mock_logger.debug.assert_any_call(
            f"Transforming SageMaker request to engine format. Input: {test_sagemaker_request_dict}"
        )
        for (
            test_sagemaker_param,
            test_engine_path,
        ) in self.api_transform.engine_request_paths.items():
            if test_engine_path is not None:
                value = test_sagemaker_request_dict.get(test_sagemaker_param)
                mock_logger.debug.assert_any_call(
                    f"Mapping {test_sagemaker_param}={value} to engine path: {test_engine_path}"
                )
                mock_set_value.assert_any_call(
                    ANY,
                    test_engine_path,
                    value,
                    create_parent=True,
                    max_create_depth=None,
                )
        mock_logger.debug.assert_any_call(f"Transformed request: {expected}")

    def test_transform_sagemaker_request_to_engine_nested_paths(self):
        """
        Test transformation with deeply nested engine paths.

        Should verify:
        - Deep nested paths like "body.config.model.parameters" are created
        - Parent structures are created automatically
        - Multiple nested mappings don't interfere with each other
        """

        # Test complex nested path transformations
        def _dummy_init_validate(*args, **kwargs):
            pass

        ConcreteApiTransform._init_validate_sagemaker_params = MagicMock(
            wraps=_dummy_init_validate
        )
        api_transform = ConcreteApiTransform(
            original_function=dummy_original_function_with_model,
            engine_request_paths={
                "param1": "body.config.model.parameters.param1",
                "param2": "body.config.inference.param2",
                "param3": "body.metadata.param3",
            },
        )

        test_sagemaker_request_dict = {
            "param1": "value1",
            "param2": "value2",
            "param3": "value3",
        }

        test_transformed_request = {"body": {}}

        expected = {
            "body": {
                "config": {
                    "model": {"parameters": {"param1": "value1"}},
                    "inference": {"param2": "value2"},
                },
                "metadata": {"param3": "value3"},
            }
        }

        actual = api_transform._transform_sagemaker_request_to_engine(
            transformed_request=test_transformed_request,
            sagemaker_request_dict=test_sagemaker_request_dict,
        )

        assert actual == expected

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_transform_sagemaker_request_to_engine_none_paths(self, mock_logger):
        """
        Test transformation when engine_path is None (should be skipped).
        (This should not be allowed to reach this method.)
        """
        original_engine_request_paths = self.api_transform.engine_request_paths
        self.api_transform.engine_request_paths = {"param1": None}

        test_transformed_request = {"body": {}}
        test_sagemaker_request_dict = {"param1": "value1"}
        expected = {"body": {}}

        actual = self.api_transform._transform_sagemaker_request_to_engine(
            transformed_request=test_transformed_request,
            sagemaker_request_dict=test_sagemaker_request_dict,
        )

        assert actual == expected

        mock_logger.debug.assert_any_call(
            f"Transforming SageMaker request to engine format. Input: {test_sagemaker_request_dict}"
        )
        mock_logger.debug.assert_any_call(
            f"Transformed request: {test_transformed_request}"
        )

        # Restore original engine_request_paths
        self.api_transform.engine_request_paths = original_engine_request_paths

    def test_transform_sagemaker_request_to_engine_missing_params(self):
        """
        Test transformation when SageMaker request is missing expected parameters.
        """
        # Test with incomplete SageMaker request data
        test_sagemaker_request_dict = {}  # Missing param1
        test_transformed_request = {"body": {}}
        expected = {"body": {"engine_param1": None}}

        actual = self.api_transform._transform_sagemaker_request_to_engine(
            transformed_request=test_transformed_request,
            sagemaker_request_dict=test_sagemaker_request_dict,
        )

        assert actual == expected

        # Test with partially missing parameters
        test_sagemaker_request_dict_partial = {
            "param1": "value1"
        }  # param2 missing but has default
        test_transformed_request_partial = {"body": {}}
        expected_partial = {"body": {"engine_param1": "value1"}}

        actual_partial = self.api_transform._transform_sagemaker_request_to_engine(
            transformed_request=test_transformed_request_partial,
            sagemaker_request_dict=test_sagemaker_request_dict_partial,
        )

        assert actual_partial == expected_partial

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.set_value",
        wraps=set_value,
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_transform_request_defaults_application(self, mock_logger, mock_set_value):
        """
        Test application of default values to transformed request.
        """
        engine_request_defaults = self.api_transform.engine_request_defaults
        test_transformed_request = {"body": {}}
        expected = {"body": {"engine_param2": 10}}
        actual = self.api_transform._transform_request_defaults(
            test_transformed_request
        )
        assert actual == expected
        mock_logger.debug.assert_any_call(
            f"Applying request defaults: {engine_request_defaults}"
        )
        for test_engine_path, test_engine_value in engine_request_defaults.items():
            mock_logger.debug.assert_any_call(
                f"Setting default {test_engine_path}={test_engine_value}"
            )
            mock_set_value.assert_any_call(
                test_transformed_request,
                test_engine_path,
                test_engine_value,
                create_parent=True,
                max_create_depth=None,
            )

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_transform_request_defaults_none(self, mock_logger):
        """
        Test behavior when no defaults are configured.
        """
        original_engine_request_defaults = self.api_transform.engine_request_defaults

        self.api_transform.engine_request_defaults = None

        test_transformed_request = {"body": {}}
        expected = {"body": {}}

        actual = self.api_transform._transform_request_defaults(
            test_transformed_request
        )

        assert actual == expected
        assert mock_logger.debug.call_count == 1
        mock_logger.debug.assert_called_with("No request defaults to apply")

        # restore original defaults
        self.api_transform.engine_request_defaults = original_engine_request_defaults

    def test_apply_to_raw_request_all_components(self):
        """
        Test applying transformed request components to raw FastAPI Request.
        """
        mock_request = MagicMock(spec=Request)

        mock_request._headers = {}
        mock_request._body = b"{}"
        mock_request.query_params = {}
        mock_request.path_params = {}

        test_headers = {"header1": "value1"}
        test_query_params = {"param1": "value1"}
        test_body = {"key": "value"}
        test_path_params = {"path_param1": "value1"}

        test_transformed_request = {
            "headers": test_headers,
            "query_params": test_query_params,
            "body": test_body,
            "path_params": test_path_params,
        }

        actual = self.api_transform._apply_to_raw_request(
            mock_request, test_transformed_request
        )

        assert actual._headers == test_headers
        assert actual.query_params == test_query_params
        assert actual._body == json.dumps(test_body).encode()
        assert actual.path_params == test_path_params

    @pytest.mark.parametrize(
        "body",
        [
            {},
            None,
        ],
        ids=[
            "empty-dict-body",
            "none-body",
        ],
    )
    def test_apply_to_raw_request_empty_body(self, body):
        """
        Test applying empty or None body to raw request.
        """
        mock_request = MagicMock(spec=Request)
        original_body = b'{"test": "body"}'
        mock_request._body = original_body

        test_transformed_request = {
            "body": body,
        }

        actual = self.api_transform._apply_to_raw_request(
            mock_request, test_transformed_request
        )
        assert actual._body == original_body

    @pytest.mark.parametrize(
        "validated_request_type",
        [
            "dict",
            "model",
        ],
        ids=[
            "validated_request_as_dict",
            "validated_request_as_model",
        ],
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_transform_request_complete_flow(self, mock_logger, validated_request_type):
        """
        Test the complete transform_request method flow.

        Should verify:
        - Defaults are applied first
        - SageMaker-to-engine transformation follows
        - Raw request is updated with transformed data
        - BaseTransformRequestOutput is returned with correct data
        - Additional fields are extracted and included
        - Logger debug messages for start and completion
        """
        validated_request = {"param1": "value1"}
        if validated_request_type == "model":
            validated_request = MockValidatedRequest.model_validate(validated_request)

        test_raw_request = MagicMock(spec=Request)
        test_raw_request._headers = {}
        test_raw_request._body = b"{}"
        test_raw_request.query_params = {}
        test_raw_request.path_params = {}

        expected_body = {"engine_param1": "value1", "engine_param2": 10}
        expected_transformed_request = {
            "body": expected_body,
            "headers": {},
            "query_params": {},
            "path_params": {},
        }

        actual_output: BaseTransformRequestOutput = (
            self.api_transform.transform_request(validated_request, test_raw_request)
        )
        assert actual_output.transformed_request == expected_transformed_request
        assert json.loads(actual_output.raw_request._body.decode()) == expected_body

        mock_logger.debug.assert_any_call(
            f"Starting request transformation for request: {validated_request}"
        )
        mock_logger.debug.assert_any_call(
            "Request transformation completed successfully"
        )


class TestBaseApiTransform2FunctionCalling:
    """Test suite for function calling and validation methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_transform_with_model_cls = ConcreteApiTransform(
            original_function=AsyncMock(
                wraps=dummy_original_function_with_model,
            ),
            engine_request_paths={"param1": "body.engine_param1"},
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.engine_param2": 10},
        )
        mock_body = {"engine_param1": "value1", "engine_param2": 10}
        self.transformed_request = {
            "body": mock_body,
            "headers": {},
            "query_params": {},
            "path_params": {},
        }
        mock_raw_request = MagicMock(spec=Request)
        mock_raw_request._headers = {}
        mock_raw_request.path_params = {}
        mock_raw_request.query_params = {}
        mock_raw_request._body = json.dumps(mock_body, sort_keys=True).encode()

        self.mock_transform_request_output: BaseTransformRequestOutput = (
            BaseTransformRequestOutput.model_validate(
                {
                    "transformed_request": self.transformed_request,
                    "raw_request": mock_raw_request,
                    "additional_fields": {},
                }
            )
        )

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    async def test_call_with_request_model_validation_success(self, mock_logger):
        """
        Test successful call with request model validation.
        """
        await self.api_transform_with_model_cls.call(self.mock_transform_request_output)
        mock_logger.debug.assert_any_call(
            f"Validating request body with model: {self.api_transform_with_model_cls.engine_request_model_cls.__name__}"
        )
        mock_logger.debug.assert_any_call("Request body validation successful")
        self.api_transform_with_model_cls.original_function.assert_called_with(
            self.api_transform_with_model_cls.engine_request_model_cls.model_validate(
                self.mock_transform_request_output.transformed_request.get("body")
            ),
            self.mock_transform_request_output.raw_request,
        )

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    async def test_call_with_request_model_validation_error(self, mock_logger):
        """
        Test call with request model validation failure.
        """

        class MockBadEngineRequest(MockEngineRequest):
            engine_param3: int

        with pytest.raises(HTTPException) as exc_info:
            await self.api_transform_with_model_cls.call(
                self.mock_transform_request_output,
                request_model_cls=MockBadEngineRequest,
            )
        assert exc_info.value.status_code == HTTPStatus.FAILED_DEPENDENCY.value

        mock_logger.debug.assert_any_call(
            f"Validating request body with model: {MockBadEngineRequest.__name__}"
        )
        mock_logger.error.assert_any_call(
            f"Request validation failed: {exc_info.value.detail}"
        )

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    async def test_call_with_request_model_validation_success_original_func_error(
        self, mock_logger
    ):
        """
        Test call with request model success and original function throws error.
        """
        mock_failure_method = AsyncMock(
            wraps=original_function_with_model_exception,
        )
        with pytest.raises(HTTPException):
            await self.api_transform_with_model_cls.call(
                self.mock_transform_request_output,
                func=mock_failure_method,
            )

        mock_logger.debug.assert_any_call(
            f"Validating request body with model: {self.api_transform_with_model_cls.engine_request_model_cls.__name__}"
        )
        mock_logger.debug.assert_any_call("Request body validation successful")
        mock_failure_method.assert_called_with(
            self.api_transform_with_model_cls.engine_request_model_cls.model_validate(
                self.mock_transform_request_output.transformed_request.get("body")
            ),
            self.mock_transform_request_output.raw_request,
        )

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    async def test_call_without_request_model(self, mock_logger):
        """
        Test call when no request model validation is configured.
        """
        original_engine_request_model_cls = (
            self.api_transform_with_model_cls.engine_request_model_cls
        )
        self.api_transform_with_model_cls.engine_request_model_cls = None

        mock_original_func = AsyncMock(
            wraps=dummy_original_function_without_model,
        )

        await self.api_transform_with_model_cls.call(
            self.mock_transform_request_output,
            func=mock_original_func,
        )
        mock_logger.debug.assert_any_call(
            "No request model validation required, calling function directly"
        )
        mock_original_func.assert_called_with(
            self.mock_transform_request_output.raw_request
        )

        self.api_transform_with_model_cls.engine_request_model_cls = (
            original_engine_request_model_cls
        )

    @pytest.mark.asyncio
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    async def test_call_with_none_transformed_request(self, mock_logger):
        """
        Test call when transformed_request is None.
        """
        mock_original_func = AsyncMock(
            wraps=dummy_original_function_without_model,
        )
        await self.api_transform_with_model_cls.call(
            BaseTransformRequestOutput(
                transformed_request=None,
                raw_request=self.mock_transform_request_output.raw_request,
            ),
            func=mock_original_func,
        )
        mock_logger.debug.assert_any_call(
            "No request model validation required, calling function directly"
        )
        mock_original_func.assert_called_with(
            self.mock_transform_request_output.raw_request
        )


class TestBaseApiTransform2ResponseHandling:
    """Test suite for response transformation and normalization methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_transform = ConcreteApiTransform(
            original_function=AsyncMock(
                wraps=dummy_original_function_with_model,
            ),
            engine_request_paths={"param1": "body.engine_param1"},
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.engine_param2": 10},
        )
        self.mock_transform_request_output = BaseTransformRequestOutput(
            raw_request=MagicMock(spec=Request),
            transformed_request={
                "body": {"engine_param1": "value1", "engine_param2": 10}
            },
            additional_fields={},
        )

    def test_generate_successful_response_content_default(self):
        """
        Test default implementation of _generate_successful_response_content.
        """
        mock_response = MagicMock(spec=Response)
        expected = "mock-response-body"
        mock_response.body = expected.encode()

        actual = self.api_transform._generate_successful_response_content(
            mock_response, transform_request_output=self.mock_transform_request_output
        )
        assert actual == expected

    def test_transform_ok_response(self):
        """
        Test default _transform_ok_response method.
        """
        self.api_transform._generate_successful_response_content = MagicMock(
            wraps=self.api_transform._generate_successful_response_content
        )
        mock_response = MagicMock(spec=Response)
        mock_response.body = "mock-response-body".encode()

        actual_response = self.api_transform._transform_ok_response(
            mock_response, self.mock_transform_request_output
        )

        self.api_transform._generate_successful_response_content.assert_any_call(
            mock_response, self.mock_transform_request_output
        )
        assert actual_response.status_code == HTTPStatus.OK.value
        assert actual_response.body == mock_response.body

    def test_transform_error_response(self):
        """
        Test default _transform_error_response method.
        """
        mock_response = MagicMock(spec=Response)

        actual = self.api_transform._transform_error_response(
            mock_response, self.mock_transform_request_output
        )

        assert actual == mock_response

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_normalize_response_with_response_object(self, mock_logger):
        """
        Test _normalize_response with existing Response object.
        """
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 200

        actual = self.api_transform._normalize_response(mock_response)
        mock_logger.debug.assert_not_called()
        assert actual == mock_response

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_normalize_response_with_basemodel(self, mock_logger):
        """
        Test _normalize_response with BaseModel object.
        """

        class MockResponseModel(BaseModel):
            response: str

        mock_response = MockResponseModel(response="test-response")

        expected = MagicMock(spec=Response)
        expected.status_code = HTTPStatus.OK.value
        expected.body = '{"response":"test-response"}'.encode()
        actual = self.api_transform._normalize_response(mock_response)

        mock_logger.debug.assert_any_call(
            "Response has no status_code attribute."
            "Assuming success if the handler returned data without explicit status."
        )

        assert actual.status_code == expected.status_code
        assert actual.body == expected.body

    @pytest.mark.parametrize(
        "mock_response",
        [
            {"response": "test-response"},
            "response",
        ],
        ids=[
            "dict",
            "str",
        ],
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_normalize_response_with_json_dumps(self, mock_logger, mock_response):
        """
        Test _normalize_response with dictionary object.
        """
        expected = MagicMock(spec=Response)
        expected.status_code = HTTPStatus.OK.value
        expected.body = json.dumps(mock_response).encode()

        actual = self.api_transform._normalize_response(mock_response)

        mock_logger.debug.assert_any_call(
            "Response has no status_code attribute."
            "Assuming success if the handler returned data without explicit status."
        )

        assert actual.status_code == expected.status_code
        assert actual.body == expected.body

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.json.dumps",
        wraps=json.dumps,
    )
    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_normalize_response_with_non_serializable(
        self, mock_logger, mock_json_dumps
    ):
        """
        Test _normalize_response with non-JSON-serializable object.

        Should verify:
        - TypeError is caught during json.dumps()
        - HTTPException is raised with INTERNAL_SERVER_ERROR status
        - Error message indicates serialization failure
        - Logger error message is generated
        """
        mock_json_dumps.side_effect = TypeError("Test serialization failure")
        mock_response = "response"

        with pytest.raises(HTTPException) as exc_info:
            self.api_transform._normalize_response(mock_response)

        mock_logger.debug.assert_any_call(
            "Response has no status_code attribute."
            "Assuming success if the handler returned data without explicit status."
        )
        mock_logger.error.assert_any_call(
            "Unable to serialize response to JSON: response"
        )

        assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value
        assert "Unable to serialize response to JSON" in exc_info.value.detail

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_transform_response_ok_status(self, mock_logger):
        """
        Test transform_response with OK (200) status code.
        """
        self.api_transform._normalize_response = MagicMock(
            wraps=self.api_transform._normalize_response
        )
        self.api_transform._transform_ok_response = MagicMock(
            wraps=self.api_transform._transform_ok_response
        )

        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 200
        mock_response.body = "mock-response-body".encode()

        actual = self.api_transform.transform_response(
            mock_response, self.mock_transform_request_output
        )

        assert actual.status_code == 200
        assert actual.body == mock_response.body
        self.api_transform._normalize_response.assert_called_with(mock_response)
        self.api_transform._transform_ok_response.assert_called_with(
            mock_response, self.mock_transform_request_output
        )
        mock_logger.debug.assert_any_call("Processing response with status code: 200")

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    def test_transform_response_error_status(self, mock_logger):
        """
        Test transform_response with error status codes (4xx, 5xx).

        Should verify:
        - _transform_error_response is called
        - Response is normalized first
        - Logger debug message for error status processing
        - Original error response is preserved
        """
        self.api_transform._normalize_response = MagicMock(
            wraps=self.api_transform._normalize_response
        )
        self.api_transform._transform_error_response = MagicMock(
            wraps=self.api_transform._transform_error_response
        )

        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 400
        mock_response.body = "mock-response-body".encode()

        actual = self.api_transform.transform_response(
            mock_response, self.mock_transform_request_output
        )

        self.api_transform._normalize_response.assert_called_with(mock_response)
        self.api_transform._transform_error_response.assert_called_with(
            mock_response, self.mock_transform_request_output
        )
        mock_logger.debug.assert_any_call("Processing response with status code: 400")
        assert actual == mock_response


class TestBaseApiTransform2IntegrationFlow:
    """Test suite for complete transformation flow integration."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.api_transform = ConcreteApiTransform(
            original_function=AsyncMock(
                wraps=dummy_original_function_with_model,
            ),
            engine_request_paths={"param1": "body.engine_param1"},
            engine_request_model_cls=MockEngineRequest,
            engine_request_defaults={"body.engine_param2": 10},
        )

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    @pytest.mark.asyncio
    async def test_transform_complete_success_flow(self, mock_logger):
        """
        Test complete successful transformation flow.
        """
        self.api_transform.validate_request = AsyncMock(
            wraps=self.api_transform.validate_request
        )
        self.api_transform.transform_request = MagicMock(
            wraps=self.api_transform.transform_request
        )
        self.api_transform.call = AsyncMock(wraps=self.api_transform.call)
        self.api_transform.transform_response = MagicMock(
            wraps=self.api_transform.transform_response
        )

        mock_raw_request = MagicMock(spec=Request)
        mock_raw_request._headers = {}
        mock_raw_request.query_params = {}
        mock_raw_request._body = b'{"param1": "value1"}'
        mock_raw_request.path_params = {}

        await self.api_transform.transform(mock_raw_request)

        mock_logger.debug.assert_any_call("Starting API transformation")
        self.api_transform.validate_request.assert_called_once_with(mock_raw_request)
        expected_validated_request = MockValidatedRequest(param1="value1")

        self.api_transform.transform_request.assert_called_once_with(
            expected_validated_request, mock_raw_request
        )
        mock_logger.debug.assert_any_call(
            f"Request validation successful for request: {expected_validated_request}"
        )
        expected_transform_request_output = BaseTransformRequestOutput(
            raw_request=mock_raw_request,
            transformed_request={
                "body": {
                    "engine_param1": "value1",
                    "engine_param2": 10,
                },
                "headers": {},
                "query_params": {},
                "path_params": {},
            },
            additional_fields={"additional_field": "value"},
        )

        self.api_transform.call.assert_called_once_with(
            expected_transform_request_output
        )
        mock_logger.debug.assert_any_call("Engine function call completed")
        self.api_transform.transform_response.assert_called_once_with(
            ANY, expected_transform_request_output
        )
        mock_logger.debug.assert_any_call("Response transformation completed")

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    @pytest.mark.asyncio
    async def test_transform_validation_failure(self, mock_logger):
        """
        Test transformation flow with request validation failure.
        """
        self.api_transform.validate_request = AsyncMock(
            side_effect=HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Test validation failure",
            )
        )

        with pytest.raises(HTTPException) as exc_info:
            await self.api_transform.transform(MagicMock(spec=Request))

        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert exc_info.value.detail == "Test validation failure"
        mock_logger.error.assert_any_call(
            "HTTP exception during transformation: Test validation failure"
        )

    @patch(
        "model_hosting_container_standards.common.transforms.base_api_transform2.logger"
    )
    @pytest.mark.asyncio
    async def test_transform_function_call_failure(self, mock_logger):
        """
        Test transformation flow with function execution failure.

        Should verify:
        - Request processing succeeds
        - Function call raises exception
        - Exception handling depends on exception type
        - Logger error messages are appropriate
        """
        self.api_transform.validate_request = AsyncMock(
            wraps=self.api_transform.validate_request
        )
        self.api_transform.transform_request = MagicMock(
            wraps=self.api_transform.transform_request
        )
        self.api_transform.call = AsyncMock(side_effect=Exception(""))
        mock_raw_request = MagicMock(spec=Request)
        mock_raw_request._headers = {}
        mock_raw_request.query_params = {}
        mock_raw_request._body = b'{"param1": "value1"}'
        mock_raw_request.path_params = {}
        with pytest.raises(Exception) as exc_info:
            await self.api_transform.transform(mock_raw_request)

        assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value
        assert exc_info.value.detail == "Unexpected error during transformation"
        mock_logger.error.assert_any_call(
            f"Unexpected error during transformation: {str(Exception(''))}"
        )


class TestBaseApiTransform2AbstractMethods:
    """Test suite for abstract method implementations and contracts."""

    def test_cannot_instantiate_base_class_directly(self):
        """
        Test that BaseApiTransform2 cannot be instantiated directly.

        Should verify:
        - TypeError is raised when trying to instantiate BaseApiTransform2
        - Error message mentions abstract methods
        - All three abstract methods are listed in the error
        """
        # Test direct instantiation failure
        with pytest.raises(TypeError) as exc_info:
            BaseApiTransform2(
                original_function=dummy_original_function_with_model,
                engine_request_paths={"param1": "body.engine_param1"},
            )

        error_message = str(exc_info.value)
        assert "Can't instantiate abstract class BaseApiTransform2" in error_message

        # Check that all three abstract methods are mentioned
        assert "validate_request" in error_message
        assert "_extract_additional_fields" in error_message
        assert "_init_validate_sagemaker_params" in error_message
