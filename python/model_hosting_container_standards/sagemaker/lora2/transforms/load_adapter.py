from http import HTTPStatus
from typing import Any, Dict, Optional

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from ...lora.constants import ResponseMessage
from ..lora_api_transform import LoRARequestBaseModel, BaseLoRAApiTransform, BaseLoRATransformRequestOutput


class SageMakerLoadAdapterRequest(LoRARequestBaseModel):
    # already has `name` from LoRARequestBaseModel
    src: str
    preload: Optional[bool] = True
    pinned: Optional[bool] = False


class LoadLoraApiTransform(BaseLoRAApiTransform):
    def __init__(
        self,
        original_function,
        engine_request_paths: Dict[str, Any],
        engine_request_model_cls: BaseModel,
        engine_request_defaults: Dict[str, Any] = None,
    ):
        super().__init__(
            original_function,
            engine_request_paths,
            engine_request_model_cls,
            engine_request_defaults,
        )

    async def validate_request(self, raw_request: Request) -> SageMakerLoadAdapterRequest:
        raw_query_params = raw_request.query_params
        if not raw_query_params:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value
            )
        try:
            return SageMakerLoadAdapterRequest.model_validate(raw_query_params)
        except ValidationError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value, detail=e.json(include_url=False)
            )

    def _generate_successful_response_content(self, raw_response: Response, transform_request_output: BaseLoRATransformRequestOutput) -> str:
        adapter_alias = transform_request_output.additional_fields.get("adapter_alias")
        adapter_name = transform_request_output.additional_fields.get("adapter_name")
        return ResponseMessage.ADAPTER_REGISTERED.format(
            alias=adapter_alias or adapter_name
        )
