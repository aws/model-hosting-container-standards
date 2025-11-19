from http import HTTPStatus
from logging import getLogger
from typing import Any, Dict

from fastapi import Response
from fastapi.exceptions import HTTPException

from ..models import SageMakerSessionHeader
from .base_engine_session_api_transform import BaseEngineSessionApiTransform
from .constants import RESPONSE_CONTENT_KEY

logger = getLogger(__name__)


class CreateSessionApiTransform(BaseEngineSessionApiTransform):
    def __init__(
        self, request_shape: Dict[str, Any], response_shape: Dict[str, Any] = {}
    ):
        try:
            assert SageMakerSessionHeader.NEW_SESSION_ID in response_shape.keys()
            assert RESPONSE_CONTENT_KEY in response_shape.keys()
        except AssertionError as e:
            raise ValueError(
                f"Response shape must contain {SageMakerSessionHeader.NEW_SESSION_ID} and {RESPONSE_CONTENT_KEY} keys"
            ) from e

        super().__init__(request_shape, response_shape)

    def _transform_ok_response(self, response: Response, **kwargs) -> Response:
        """Transform successful create session response.

        Extracts session ID and content from engine response, validates them,
        and returns formatted response with NEW_SESSION_ID header.

        :param Response response: The successful response to transform
        :return Response: Transformed response with session headers
        :raises HTTPException: If session ID cannot be extracted from response
        """
        transformed_response_data = self._transform_response(response)
        content = transformed_response_data.get(RESPONSE_CONTENT_KEY)
        session_id = transformed_response_data.get(
            SageMakerSessionHeader.NEW_SESSION_ID
        )

        # Validate that session_id was extracted from the response
        if not session_id:
            logger.error(
                f"Failed to extract session ID from engine response. "
                f"Response data: {transformed_response_data}"
            )
            raise HTTPException(
                status_code=HTTPStatus.BAD_GATEWAY.value,
                detail="Engine failed to return a valid session ID in the response",
            )

        # Validate that content was extracted from the response
        if not content:
            logger.debug(
                f"No content extracted from create session response for session {session_id}"
            )

        logger.info(f"Session {session_id}: {content}")
        return Response(
            status_code=HTTPStatus.OK.value,
            content=f"Session {session_id}: {content}",
            headers={SageMakerSessionHeader.NEW_SESSION_ID: session_id},
        )
