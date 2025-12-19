from http import HTTPStatus
from typing import Any, Dict

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from ...lora.constants import ResponseMessage
from ...lora.utils import get_adapter_name_from_request_path
from ..lora_api_transform import LoRARequestBaseModel, BaseLoRAApiTransform, BaseLoRATransformRequestOutput


SageMakerUnloadAdapterRequest = LoRARequestBaseModel
# already has `name` from LoRARequestBaseModel

class UnloadLoraApiTransform(BaseLoRAApiTransform):
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

    async def validate_request(self, raw_request: Request) -> SageMakerUnloadAdapterRequest:
        try:
            return SageMakerUnloadAdapterRequest.model_validate({
                "name": get_adapter_name_from_request_path(raw_request),
            })
        except (KeyError, ValidationError):
            raise HTTPException(
                status_code=HTTPStatus.FAILED_DEPENDENCY.value,
                detail=f"Malformed request.",  # TODO: improve error message
            )

    def _generate_successful_response_content(self, raw_response: Response, transform_request_output: BaseLoRATransformRequestOutput) -> str:
        adapter_alias = transform_request_output.additional_fields.get("adapter_alias")
        adapter_name = transform_request_output.additional_fields.get("adapter_name")
        return ResponseMessage.ADAPTER_UNREGISTERED.format(
            alias=adapter_alias or adapter_name
        )
