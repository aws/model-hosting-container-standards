import abc
import json
from http import HTTPStatus

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from ....common import BaseApiTransform, BaseTransformRequestOutput
from ....logging_config import logger


class BaseEngineSessionApiTransform(BaseApiTransform):
    """Base abstract class for engine-specific session API transformations.

    This class provides the foundation for transforming HTTP requests and responses
    for engines that implement their own session management APIs. It handles common
    response normalization and routing logic, while subclasses implement specific
    transformation behavior for create/close session operations.
    """

    async def transform_request(
        self, raw_request: Request
    ) -> BaseTransformRequestOutput:
        """Transform an incoming HTTP request for engine session operations.

        Parses JSON request body, applies JMESPath transformations, and validates
        any session-specific requirements. Subclasses can override to add custom
        validation logic before or after the base transformation.

        :param Request raw_request: The incoming FastAPI request object
        :return BaseTransformRequestOutput: Transformed request data and metadata
        :raises HTTPException: If JSON parsing fails or validation errors occur
        """
        # Subclasses can override _validate_request_preconditions for early validation
        self._validate_request_preconditions(raw_request)

        try:
            request_data = await raw_request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"JSON decode error: {e}",
            ) from e

        transformed_request = self._transform_request(request_data, raw_request)
        raw_request._body = json.dumps(transformed_request).encode("utf-8")

        return BaseTransformRequestOutput(
            request=transformed_request,
            raw_request=raw_request,
            intercept_func=None,
        )

    def _validate_request_preconditions(self, raw_request: Request) -> None:
        """Validate request preconditions before transformation.

        Subclasses can override this method to perform early validation
        (e.g., checking for required headers). Default implementation does nothing.

        :param Request raw_request: The incoming request to validate
        :raises HTTPException: If validation fails
        """
        pass

    def transform_response(
        self, response: Response, transform_request_output: BaseTransformRequestOutput
    ) -> Response:
        """Transform the response based on the request processing results.

        Normalizes various response types to FastAPI Response objects and routes
        to appropriate transformation method based on HTTP status code.

        :param Response response: The response to transform (may be Response, BaseModel, dict, or str)
        :param BaseTransformRequestOutput transform_request_output: Output from the request transformation
        :return Response: Transformed response
        """
        # Normalize response to Response object
        response = self._normalize_response(response)

        # Route based on status code
        if response.status_code == HTTPStatus.OK.value:
            return self._transform_ok_response(
                response, transform_request_output=transform_request_output
            )
        else:
            return self._transform_error_response(response)

    def _normalize_response(self, response):
        """Convert various response types to FastAPI Response object.

        Handles responses that may be BaseModel instances, dictionaries, strings,
        or already Response objects. If the response doesn't have a status_code,
        it's assumed to be a successful response (200 OK) from the engine handler.

        Note: This method only normalizes the response format. Validation of required
        fields (like session IDs) should be done in _transform_ok_response() to provide
        appropriate error responses if the engine returns invalid data.

        :param response: Response in various formats
        :return Response: Normalized FastAPI Response object
        """
        if not hasattr(response, "status_code"):
            # Handle the case where the response is not a Response object
            # Assume success if the handler returned data without explicit status
            if isinstance(response, BaseModel):
                response = response.model_dump_json()
            elif not isinstance(response, str):
                response = json.dumps(response)
            response = Response(
                status_code=HTTPStatus.OK.value,
                content=response,
            )
        return response

    @abc.abstractmethod
    def _transform_ok_response(self, response: Response, **kwargs) -> Response:
        """Transform successful (200 OK) responses.

        Subclasses must implement this method to handle session-specific response
        formatting and header management.

        :param Response response: The successful response to transform
        :param BaseTransformRequestOutput transform_request_output: Output from the request transformation
        :return Response: Transformed response
        :raises NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    def _transform_error_response(self, response: Response, **kwargs) -> Response:
        """Transform error responses.

        Default implementation passes through error responses unchanged.
        Subclasses can override to add custom error handling.

        :param Response response: The error response to transform
        :param BaseTransformRequestOutput transform_request_output: Output from the request transformation
        :return Response: Transformed response (default: unchanged)
        """
        return response
