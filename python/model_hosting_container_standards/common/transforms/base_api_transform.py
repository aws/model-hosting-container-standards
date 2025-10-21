import abc
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import jmespath
from fastapi import Request, Response
from pydantic import BaseModel, Field

from ...logging_config import logger
from ..fastapi.utils import serialize_request
from .utils import _compile_jmespath_expressions


class BaseTransformRequestOutput(BaseModel):
    """Output model for generic request transformation."""

    request: Optional[Any] = None
    raw_request: Optional[Any] = None
    intercept_func: Optional[Callable[..., Any]] = Field(default=None)


class BaseApiTransform(abc.ABC):
    """Base abstract class for LoRA API request/response transformations.

    This class provides the foundation for transforming HTTP requests and responses
    using JMESPath expressions defined in request_shape and response_shape dictionaries.
    Subclasses must implement the abstract methods to handle specific transformation logic.
    """

    def __init__(
        self, request_shape: Dict[str, Any], response_shape: Dict[str, Any] = {}
    ):
        """Initialize the transformer with request and response mapping shapes.

        :param Dict[str, Any] request_shape: Dictionary containing JMESPath expressions
            that define how to extract and map data from incoming requests
        :param Dict[str, Any] response_shape: Dictionary containing JMESPath expressions
            that define how to transform response data (defaults to empty dict)
        """
        self._request_shape = _compile_jmespath_expressions(request_shape)
        self._response_shape = _compile_jmespath_expressions(response_shape)

    def _transform(
        self, source_data: Dict[str, Any], target_shape: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply JMESPath transformations to source data using the target shape.

        :param Dict[str, Any] source_data: The source data to transform
        :param Dict[str, Any] target_shape: Dictionary of compiled JMESPath expressions
        :return Dict[str, Any]: Transformed data structure
        """
        transformed_request = {}
        for target_key, nested_or_compiled in target_shape.items():
            if isinstance(nested_or_compiled, jmespath.parser.ParsedResult):
                # Apply compiled JMESPath expression to extract value
                value = nested_or_compiled.search(source_data)
                transformed_request[target_key] = value
            elif isinstance(nested_or_compiled, dict):
                # Recursively transform nested structures
                transformed_request[target_key] = self._transform(
                    source_data, nested_or_compiled
                )
            else:
                logger.warning(
                    f"Request/response mapping must be a dictionary of strings (nested allowed), not {type(nested_or_compiled)}. This value will be ignored."
                )
        return transformed_request

    async def intercept(
        self,
        func: Callable[..., Any],
        transform_request_output: BaseTransformRequestOutput,
    ):
        transformed_request = transform_request_output.request
        transformed_raw_request = transform_request_output.raw_request
        func_to_call = transform_request_output.intercept_func or func

        if not transformed_request:
            logger.debug("No transformed request data, passing raw request only")
            # If transformed_request is None, only pass the modified raw request
            response = await func_to_call(transformed_raw_request)
        else:
            logger.debug("Passing transformed request data and raw request to handler")
            # Pass both transformed data and original request for context
            # Convert dict to SimpleNamespace for attribute access
            response = await func_to_call(
                SimpleNamespace(**transformed_request), transformed_raw_request
            )
        return response

    @abc.abstractmethod
    async def transform_request(self, raw_request: Request):
        """Transform an incoming HTTP request for operations.

        Subclasses must implement this method to handle request parsing, validation,
        and transformation according to their specific operation requirements.

        :param Request raw_request: The incoming FastAPI request object
        :raises NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    def _transform_request(
        self, request: Optional[BaseModel], raw_request: Request
    ) -> Dict[str, Any]:
        """Apply request shape transformations to extract structured data from the request.

        :param Optional[BaseModel] request: Parsed request body as Pydantic model (can be None)
        :param Request raw_request: The raw FastAPI request object
        :return Dict[str, Any]: Transformed request data based on request_shape
        """
        request_data = serialize_request(request, raw_request)
        return self._transform(request_data, self._request_shape)

    @abc.abstractmethod
    def transform_response(self, response: Response, transform_request_output):
        """Transform the response based on the request processing results.

        Subclasses must implement this method to handle request parsing, validation,
        and transformation according to their specific operation requirements.

        :param Response response: The response to transform
        :param transform_request_output: Output from the request transformation
        :raises NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    def _transform_ok_response(self, response: Response, **kwargs):
        """Transform successful (200 OK) responses.

        :param Response response: The successful response to transform
        :raises NotImplementedError: Must be implemented by subclasses if needed
        """
        raise NotImplementedError()

    def _transform_error_response(self, response: Response, **kwargs):
        """Transform error responses.

        :param Response response: The error response to transform
        :raises NotImplementedError: Must be implemented by subclasses if needed
        """
        raise NotImplementedError()
