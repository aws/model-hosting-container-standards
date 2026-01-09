from http import HTTPStatus
from typing import Any, Dict, Optional

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ...common.handler import handler_registry
from ...common.transforms.base_api_transform2 import (
    BaseApiTransform2,
    BaseTransformRequestOutput,
)
from ...common.transforms.defaults_config import _transform_defaults_config
from ...logging_config import logger
from .models import SageMakerSessionHeader

SAGEMAKER_HEADER_PREFIX = "X-Amzn-SageMaker-"


def to_hyphens(field_name: str) -> str:
    return field_name.replace("_", "-")


def to_sagemaker_headers(field_name: str) -> str:
    field_name = to_hyphens(field_name)
    field_name = SAGEMAKER_HEADER_PREFIX + field_name.title()
    return field_name


class SageMakerSessionRequestHeader(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_sagemaker_headers,
        populate_by_name=True,
        extra="ignore",
    )
    session_id: str

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if v == "":
            raise ValueError("Session ID cannot be empty")
        return v


class CloseSessionApiTransform(BaseApiTransform2):
    async def validate_request(self, raw_request):
        try:
            return SageMakerSessionRequestHeader.model_validate(raw_request.headers)
        except ValidationError:
            logger.error("No session ID found in request headers for close session")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Session ID is required in request headers to close a session",
            )

    def _extract_additional_fields(
        self, validated_request: SageMakerSessionRequestHeader, raw_request: Request
    ):
        return dict(
            session_id=validated_request.session_id,
        )

    def _init_validate_sagemaker_params(self, sagemaker_param):
        if sagemaker_param not in SageMakerSessionRequestHeader.model_fields.keys():
            raise ValueError(
                f"Invalid sagemaker_param: {sagemaker_param}. "
                f"Allowed value(s): {SageMakerSessionRequestHeader.model_fields.keys()}"
            )

    def _generate_successful_response_content(
        self,
        raw_response: Response,
        transform_request_output: BaseTransformRequestOutput,
    ):
        session_id = transform_request_output.additional_fields.get("session_id")
        content_prefix = "Successfully closed session"
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
        # since we already know the session_id value
        # and where to store it
        return Response(
            status_code=HTTPStatus.OK.value,
            headers={
                SageMakerSessionHeader.CLOSED_SESSION_ID: transform_request_output.additional_fields.get(
                    "session_id"
                ),
            },
            content=self._generate_successful_response_content(
                raw_response, transform_request_output
            ),
        )


def create_close_session_transform(
    engine_request_paths: Optional[Dict[str, Any]] = None,
    engine_request_model_cls: BaseModel = None,
    engine_request_defaults: Optional[Dict[str, Any]] = None,
):
    handler_type = "close_session"

    def close_session_decorator(original_func):
        close_session_transform = CloseSessionApiTransform(
            original_func,
            engine_request_paths,
            engine_request_model_cls,
            engine_request_defaults=engine_request_defaults,
        )

        async def close_session_transform_wrapper(raw_request: Request):
            return await close_session_transform.transform(raw_request)

        handler_registry.set_handler(handler_type, close_session_transform_wrapper)
        logger.info(
            f"[{handler_type.upper()}] Registered transform handler for {original_func.__name__}"
        )
        return close_session_transform_wrapper

    return close_session_decorator


def _register_close_session_handler(
    engine_request_session_id_path: str,
    engine_request_model_cls: Optional[BaseModel] = None,
):
    logger.info("Registering close session handler")
    logger.debug(
        f"Handler parameters - request_session_id_path: {engine_request_session_id_path}"
    )
    engine_request_paths = {"session_id": engine_request_session_id_path}
    return create_close_session_transform(
        engine_request_paths,
        engine_request_model_cls,
        engine_request_defaults=_transform_defaults_config.close_session_defaults,
    )
