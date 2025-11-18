from http import HTTPStatus
from logging import getLogger
from typing import Any, Dict

from fastapi import Request, Response
from fastapi.exceptions import HTTPException

from ....common import BaseTransformRequestOutput
from ..models import SageMakerSessionHeader
from ..utils import get_session_id_from_request
from .base_engine_session_api_transform import BaseEngineSessionApiTransform
from .constants import RESPONSE_CONTENT_KEY

logger = getLogger(__name__)


class CloseSessionApiTransform(BaseEngineSessionApiTransform):
    def __init__(
        self, request_shape: Dict[str, Any], response_shape: Dict[str, Any] = {}
    ):
        try:
            assert RESPONSE_CONTENT_KEY in response_shape.keys()
        except AssertionError as e:
            raise ValueError(
                f"Response shape must contain {RESPONSE_CONTENT_KEY} key"
            ) from e

        super().__init__(request_shape, response_shape)

    def _validate_request_preconditions(self, raw_request: Request) -> None:
        """Validate that session ID exists in request headers before processing.

        :param Request raw_request: The incoming request to validate
        :raises HTTPException: If session ID is missing from headers
        """
        session_id = get_session_id_from_request(raw_request)
        if not session_id:
            logger.error("No session ID found in request headers for close session")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Session ID is required in request headers to close a session",
            )

    def _transform_ok_response(self, response: Response, **kwargs) -> Response:
        """Transform successful close session response.

        Extracts session ID from request headers and content from engine response,
        validates them, and returns formatted response with CLOSED_SESSION_ID header.

        :param Response response: The successful response to transform
        :param BaseTransformRequestOutput transform_request_output: Output from the request transformation
        :return Response: Transformed response with session headers
        """
        transform_request_output: BaseTransformRequestOutput = kwargs.get("transform_request_output")  # type: ignore
        # Session ID already validated in transform_request, safe to extract
        session_id = get_session_id_from_request(transform_request_output.raw_request)

        transformed_response_data = self._transform_response(response)
        content = transformed_response_data.get(RESPONSE_CONTENT_KEY)

        # Validate that content was extracted from the response
        if not content:
            logger.debug(
                f"No content extracted from close session response for session {session_id}"
            )

        logger.info(f"Session {session_id}: {content}")
        return Response(
            status_code=HTTPStatus.OK.value,
            content=f"Session {session_id}: {content}",
            headers={SageMakerSessionHeader.CLOSED_SESSION_ID: session_id},
        )
