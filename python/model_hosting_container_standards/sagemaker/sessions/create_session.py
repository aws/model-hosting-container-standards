from http import HTTPStatus
from typing import Any, Dict, Optional

import jmespath
from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

from ...common.fastapi.utils import serialize_response
from ...common.handler import handler_registry
from ...common.transforms.base_api_transform2 import (
    BaseApiTransform2,
    BaseTransformRequestOutput,
)
from ...common.transforms.defaults_config import _transform_defaults_config
from ...logging_config import logger
from .models import SageMakerSessionHeader


class CreateSessionApiTransform(BaseApiTransform2):
    def __init__(
        self,
        original_function,
        engine_request_paths: Dict[str, Any],
        engine_response_session_id_path: str,
        engine_request_model_cls: BaseModel,
        engine_request_defaults: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            original_function,
            engine_request_paths,
            engine_request_model_cls,
            engine_request_defaults,
        )
        self.engine_response_session_id_jmesexpr = jmespath.compile(
            engine_response_session_id_path
        )

    async def validate_request(self, raw_request):
        return {}

    def _extract_additional_fields(self, validated_request, raw_request: Request):
        # no additional fields needed
        return {}

    def _generate_successful_response_content(
        self,
        raw_response: Response,
        transform_request_output: BaseTransformRequestOutput,
    ):
        session_id = transform_request_output.additional_fields.get("session_id")
        content_prefix = "Successfully created session"
        if session_id:
            return f"{content_prefix}: {session_id}"
        else:
            return f"{content_prefix}"

    def _transform_ok_response(
        self,
        raw_response: Response,
        transform_request_output: BaseTransformRequestOutput,
    ):
        # Overwrite base class method _transform_ok_response
        serialized_response = serialize_response(raw_response)
        logger.debug(
            f"Transforming engine response to SageMaker format. Input: {serialized_response}"
        )
        session_id = self.engine_response_session_id_jmesexpr.search(
            serialized_response
        )
        if not session_id:
            logger.warning(f"Session ID not found in response: {serialized_response}")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="Session ID not found in response",
            )
        transform_request_output.additional_fields["session_id"] = session_id
        return Response(
            status_code=HTTPStatus.OK.value,
            headers={
                SageMakerSessionHeader.NEW_SESSION_ID: session_id,
            },
            content=self._generate_successful_response_content(
                raw_response, transform_request_output
            ),
        )


def create_create_session_transform():
    handler_type = "create_session"

    def create_session_decorator_with_params(
        engine_request_paths: Dict[str, Any],
        engine_response_session_id_path: Optional[str] = None,
        engine_request_model_cls: Optional[BaseModel] = None,
        engine_request_defaults: Optional[Dict[str, Any]] = None,
    ):
        def create_session_decorator(original_func):
            create_session_transform = CreateSessionApiTransform(
                original_func,
                engine_request_paths,
                engine_response_session_id_path,
                engine_request_model_cls,
                engine_request_defaults=engine_request_defaults,
            )

            async def create_session_transform_wrapper(raw_request: Request):
                return await create_session_transform.transform(raw_request)

            handler_registry.set_handler(handler_type, create_session_transform_wrapper)
            logger.info(
                f"[{handler_type.upper()}] Registered transform handler for {original_func.__name__}"
            )
            return create_session_transform_wrapper

        return create_session_decorator

    return create_session_decorator_with_params


def _register_create_session_handler(
    engine_response_session_id_path: str,
    engine_request_model_cls: Optional[BaseModel] = None,
):
    logger.info("Registering create session handler")
    logger.debug(
        f"Handler parameters - response_session_id_path: {engine_response_session_id_path}"
    )
    return create_create_session_transform()(
        engine_request_paths={},
        engine_response_session_id_path=engine_response_session_id_path,
        engine_request_model_cls=engine_request_model_cls,
        engine_request_defaults=_transform_defaults_config.create_session_defaults,
    )
