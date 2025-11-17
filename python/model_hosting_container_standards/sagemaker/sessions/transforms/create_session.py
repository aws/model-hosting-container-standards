import json
from http import HTTPStatus
from logging import getLogger
from typing import Any, Dict

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from ....common import BaseApiTransform, BaseTransformRequestOutput
from ..models import SageMakerSessionHeader
from .close_session import CloseSessionApiTransform

RESPONSE_CONTENT_KEY = "content"

logger = getLogger(__name__)


class CreateSessionApiTransform(BaseApiTransform):
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

    async def transform_request(self, raw_request: Request):
        try:
            _ = await raw_request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"JSON decode error: {e}",
            ) from e
        transformed_request = self._transform_request(None, raw_request)
        raw_request._body = json.dumps(transformed_request).encode("utf-8")
        return BaseTransformRequestOutput(
            request=transformed_request,
            raw_request=raw_request,
            intercept_func=None,
        )

    def transform_response(self, response: Response, transform_request_output):
        if not hasattr(response, "status_code"):
            # Handle the case where the response is not a Response object
            if isinstance(response, BaseModel):
                response = response.model_dump_json()
            elif not isinstance(response, str):
                response = json.dumps(response)
            response = Response(
                status_code=HTTPStatus.OK.value,
                content=response,
            )
        if response.status_code == HTTPStatus.OK.value:
            return self._transform_ok_response(response)
        else:
            return self._transform_error_response(response)

    def _transform_error_response(self, response: Response, **kwargs):
        return response

    def _transform_ok_response(self, response: Response, **kwargs):
        transformed_response_data = self._transform_response(response)
        content = transformed_response_data.get(RESPONSE_CONTENT_KEY)
        session_id = transformed_response_data.get(
            SageMakerSessionHeader.NEW_SESSION_ID
        )
        logger.info(f"Session {session_id}: {content}")
        return Response(
            status_code=HTTPStatus.OK.value,
            content=f"Session {session_id}: {content}",
            headers={SageMakerSessionHeader.NEW_SESSION_ID: session_id},
        )


def resolve_engine_session_transform(handler_type: str):
    if handler_type == "create_session":
        return CreateSessionApiTransform
    elif handler_type == "close_session":
        return CloseSessionApiTransform
    return None
