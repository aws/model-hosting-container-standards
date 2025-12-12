import json
from http import HTTPStatus
from typing import Optional

from fastapi import Request
from fastapi.exceptions import HTTPException
from pydantic import ValidationError

from ...common import BaseApiTransform, BaseTransformRequestOutput
from ...common.handler import handler_registry
from ...common.transforms.utils import set_value
from ...logging_config import logger
from .handlers import get_handler_for_request_type
from .manager import get_session_manager
from .models import (
    SESSION_DISABLED_ERROR_DETAIL,
    SESSION_DISABLED_LOG_MESSAGE,
    SageMakerSessionHeader,
    SessionRequest,
    SessionRequestType,
)
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


class SessionApiTransform(BaseApiTransform):
    def __init__(self, request_shape, response_shape={}):
        """Initialize the SessionApiTransform.

        Args:
            request_shape: Passed to parent BaseApiTransform
            response_shape: Passed to parent BaseApiTransform

        Note:
            The request/response shapes are passed to the parent class but not used
            for validation in this transform, as session requests use their own validation.
        """
        self._session_manager = get_session_manager()
        self._use_default_manager = None

        # Extract session_id_target_key before compiling JMESPath expressions
        self._session_id_target_key = self._get_session_id_target_key(request_shape)
        super().__init__(request_shape, response_shape)

    def _check_use_default_manager(self):
        """Check if the default session manager should be used.

        Returns:
            bool: True if the default session manager should be used, False otherwise
        """
        if self._use_default_manager is None:
            # If unset, first call -> set cached value
            logger.info("Checking if default session manager should be used.")
            self._use_default_manager = not handler_registry.has_handler(
                "create_session"
            ) and not handler_registry.has_handler("close_session")
            logger.info(f"Using default session manager: {self._use_default_manager}")
        return self._use_default_manager

    def _get_session_id_target_key(self, request_shape: dict) -> Optional[str]:
        if not request_shape:
            return None
        for target_key, source_path in request_shape.items():
            if source_path == f'headers."{SageMakerSessionHeader.SESSION_ID}"':
                return target_key
        return None

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
            return self._process_request(request_data, raw_request)
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

    def _validate_session_id(self, session_id: Optional[str], raw_request: Request):
        """Validate that the session ID in the request exists and is not expired.

        Raises:
            HTTPException: If session validation fails
        """
        try:
            get_session(self._session_manager, raw_request)
            return session_id
        except ValueError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Bad request: {str(e)}",
            )

    def _process_invocations_request(
        self, session_id: Optional[str], request_data: dict, raw_request: Request
    ):
        # If not a session request
        if session_id and self._check_use_default_manager():
            # but it has a session id header and we are using the default session manager,
            # then we need to validate that the session id exists in the session manager
            self._validate_session_id(session_id, raw_request)

        # Inject session ID into request body if target key is specified
        if session_id and self._session_id_target_key:
            request_data = set_value(
                obj=request_data,
                path=self._session_id_target_key,
                value=session_id,
                create_parent=True,
            )
            logger.debug(f"Updated request body: {request_data}")
            raw_request._body = json.dumps(request_data).encode("utf-8")

        return BaseTransformRequestOutput(
            raw_request=raw_request,
            intercept_func=None,
        )

    def _process_session_request(self, session_request, session_id, raw_request):
        # Validation
        if self._check_use_default_manager() and not self._session_manager:
            # if no custom handlers are registered, but default session manager
            # does not exist -> then raise error that session management is disabled
            logger.error(SESSION_DISABLED_LOG_MESSAGE)
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=SESSION_DISABLED_ERROR_DETAIL,
            )
        elif self._check_use_default_manager() and self._session_manager:
            if session_request.requestType == SessionRequestType.NEW_SESSION:
                # Ignores any session id header in create session request
                session_id = SessionRequestType.NEW_SESSION
            session_id = self._validate_session_id(session_id, raw_request)

        # Route to appropriate session management handler
        intercept_func = get_handler_for_request_type(session_request.requestType)

        return BaseTransformRequestOutput(
            raw_request=raw_request, intercept_func=intercept_func
        )

    def _process_request(self, request_data, raw_request):
        session_request = _parse_session_request(request_data)
        session_id = get_session_id_from_request(raw_request)
        if not session_request:
            return self._process_invocations_request(
                session_id, request_data, raw_request
            )
        else:
            return self._process_session_request(
                session_request, session_id, raw_request
            )
