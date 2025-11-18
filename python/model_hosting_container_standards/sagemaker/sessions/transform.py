import json
from http import HTTPStatus
from typing import Optional

from fastapi import Request
from fastapi.exceptions import HTTPException
from pydantic import ValidationError

from ...common import BaseApiTransform, BaseTransformRequestOutput
from .handlers import get_handler_for_request_type
from .manager import SessionManager, session_manager
from .models import SessionRequest
from .utils import get_session, get_session_id_from_request


def _parse_session_request(request_data: dict) -> Optional[SessionRequest]:
    """Parse and validate if request is a session management request.

    Args:
        request_data: Parsed JSON request body

    Returns:
        SessionRequest if valid session management request, None if not a session request

    Raises:
        HTTPException: If requestType is present but validation fails
    """
    try:
        return SessionRequest.model_validate(request_data)
    except ValidationError as e:
        # If requestType is present but validation failed, it's a malformed session request
        if "requestType" in request_data:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=e.json(include_url=False),
            )
        # Not a session request
        return None


def _validate_session_if_present(raw_request: Request, session_manager: SessionManager):
    """Validate that the session ID in the request exists and is not expired.

    Args:
        raw_request: FastAPI Request object
        session_manager: SessionManager instance

    Raises:
        HTTPException: If session validation fails
    """
    session_id = get_session_id_from_request(raw_request)
    if session_id:
        try:
            get_session(session_manager, raw_request)
        except ValueError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Bad request: {str(e)}",
            )


def process_session_request(
    request_data: dict, raw_request: Request, session_manager: SessionManager
):
    """Process a potential session management request.

    Determines if the request is a session management operation (NEW_SESSION or CLOSE)
    and routes it to the appropriate handler, or passes through for normal processing.

    Args:
        request_data: Parsed JSON request body
        raw_request: FastAPI Request object
        session_manager: SessionManager instance

    Returns:
        BaseTransformRequestOutput with either:
        - intercept_func set if this is a session management request
        - None/passthrough if this is a regular request

    Raises:
        HTTPException: If request is malformed or session validation fails
    """
    session_request = _parse_session_request(request_data)

    # Validate session if session ID is present in headers
    # and raise error if session ID is invalid
    _validate_session_if_present(raw_request, session_manager)

    # Not a session request - pass through for normal processing
    if session_request is None:
        return BaseTransformRequestOutput(
            request=None,
            raw_request=raw_request,
            intercept_func=None,
        )

    # Route to appropriate session management handler
    intercept_func = get_handler_for_request_type(session_request.requestType)

    return BaseTransformRequestOutput(
        request=session_manager, raw_request=raw_request, intercept_func=intercept_func
    )


class SessionApiTransform(BaseApiTransform):
    """API transform that intercepts and processes stateful session management requests.

    This transform extends BaseApiTransform to add session management capabilities.
    It parses incoming requests to detect session management operations (NEW_SESSION, CLOSE)
    and routes them to appropriate handlers, while passing through regular API requests.
    """

    def __init__(self, request_shape, response_shape={}):
        """Initialize the SessionApiTransform.

        Args:
            request_shape: Passed to parent BaseApiTransform (unused in session logic)
            response_shape: Passed to parent BaseApiTransform (unused in session logic)

        Note:
            The request/response shapes are passed to the parent class but not used
            for validation in this transform, as session requests use their own validation.
        """
        self._session_manager = session_manager
        super().__init__(request_shape, response_shape)

    async def transform_request(self, raw_request):
        """Transform incoming request, intercepting session management operations.

        Parses the request JSON and determines if it's a session management request
        (NEW_SESSION or CLOSE) or a regular API request. Session requests are routed
        to handlers, while regular requests pass through for normal processing.

        Args:
            raw_request: FastAPI Request object

        Returns:
            BaseTransformRequestOutput with intercept_func set if session request

        Raises:
            HTTPException: If JSON parsing fails (400 BAD_REQUEST)
        """
        try:
            request_data = await raw_request.json()
            return process_session_request(
                request_data, raw_request, self._session_manager
            )
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"JSON decode error: {e}",
            )

    def transform_response(self, response, transform_request_output):
        """Transform outgoing response (currently pass-through).

        Args:
            response: The response object to transform
            transform_request_output: Output from transform_request

        Returns:
            The unmodified response object
        """
        return response
