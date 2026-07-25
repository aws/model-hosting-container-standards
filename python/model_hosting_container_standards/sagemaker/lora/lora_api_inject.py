from typing import Any, Dict, Optional, Union

from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, field_validator

from ...common.handler.registry import handler_registry
from ...common.transforms.base_api_inject import (
    BaseApiInject,
    BaseInjectValidateOutput,
    InjectDefinition,
)
from ...common.transforms.utils import to_sagemaker_headers
from ...logging_config import logger
from .constants import LoRAHandlerType, SageMakerLoRAApiHeader


class SageMakerLoRARequestHeader(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_sagemaker_headers,
        populate_by_name=True,
        extra="ignore",
    )
    adapter_identifier: str
    adapter_alias: Optional[str] = None

    @field_validator("adapter_identifier")
    @classmethod
    def validate_adapter_identifier(cls, v: str) -> str:
        if v == "":
            raise ValueError("Adapter identifier cannot be empty")
        return v


class LoRAApiInject(BaseApiInject):
    def _init_validate_sagemaker_params(self, sagemaker_param: str) -> None:
        if sagemaker_param not in SageMakerLoRARequestHeader.model_fields.keys():
            raise ValueError(
                f"Invalid sagemaker_param: {sagemaker_param}. "
                f"Allowed value(s): {SageMakerLoRARequestHeader.model_fields.keys()}"
            )

    async def validate_request_should_inject(
        self, raw_request: Request
    ) -> BaseInjectValidateOutput:
        request_body = await raw_request.json()
        if raw_request.headers.get(SageMakerLoRAApiHeader.ADAPTER_IDENTIFIER) is None:
            return BaseInjectValidateOutput(
                should_inject=False,
                request_body=request_body,
                sagemaker_values=None,
            )
        else:
            return BaseInjectValidateOutput(
                should_inject=True,
                request_body=request_body,
                sagemaker_values=SageMakerLoRARequestHeader.model_validate(
                    raw_request.headers
                ),
            )

    def _extract_additional_fields(
        self,
        sagemaker_values: Union[SageMakerLoRARequestHeader, Dict[str, Any]],
        request_body: Dict[str, Any],
        raw_request,
    ) -> Dict[str, Any]:
        if isinstance(sagemaker_values, BaseModel):
            sagemaker_values = sagemaker_values.model_dump()
        return dict(
            adapter_identifier=sagemaker_values.get("adapter_identifier"),
            adapter_alias=sagemaker_values.get("adapter_alias"),
        )


def create_lora_api_inject(
    engine_request_inject_definitions: Dict[str, InjectDefinition],
    engine_request_model_cls: Optional[BaseModel] = None,
    engine_request_defaults: Optional[Dict[str, Any]] = None,
):
    handler_type = LoRAHandlerType.INJECT_ADAPTER_ID

    def lora_inject_decorator(original_func):
        lora_inject = LoRAApiInject(
            original_func,
            engine_request_inject_definitions,
            engine_request_model_cls,
            engine_request_defaults=engine_request_defaults,
        )

        async def lora_inject_wrapper(raw_request: Request) -> Response:
            return await lora_inject.inject(raw_request)

        handler_registry.set_handler(handler_type, lora_inject_wrapper)
        logger.info(
            f"[{handler_type.upper()}] Registered transform handler for {original_func.__name__}"
        )

        return lora_inject_wrapper

    return lora_inject_decorator
