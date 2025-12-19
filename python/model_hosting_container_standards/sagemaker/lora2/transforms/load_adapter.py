from http import HTTPStatus
from typing import Any, Dict, Optional

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from ....logging_config import logger
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
        logger.debug("Initialized LoadLoraApiTransform")

    async def validate_request(self, raw_request: Request) -> SageMakerLoadAdapterRequest:
        logger.debug("Validating load adapter request")
        
        raw_query_params = raw_request.query_params
        logger.debug(f"Received query parameters: {dict(raw_query_params)}")
        
        if not raw_query_params:
            logger.error("No query parameters provided for load adapter request")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Query parameters are required for load adapter request"
            )
        try:
            validated_request = SageMakerLoadAdapterRequest.model_validate(raw_query_params)
            logger.info(f"Load adapter request validated successfully for adapter: {validated_request.name}")
            logger.debug(f"Validated parameters - name: {validated_request.name}, src: {validated_request.src}, "
                        f"preload: {validated_request.preload}, pinned: {validated_request.pinned}")
            return validated_request
        except ValidationError as e:
            logger.error(f"Load adapter request validation failed: {e}")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value, detail=e.json(include_url=False)
            )

    def _generate_successful_response_content(self, raw_response: Response, transform_request_output: BaseLoRATransformRequestOutput) -> str:
        adapter_alias = transform_request_output.additional_fields.get("adapter_alias")
        adapter_name = transform_request_output.additional_fields.get("adapter_name")
        
        response_content = ResponseMessage.ADAPTER_REGISTERED.format(
            alias=adapter_alias or adapter_name
        )
        
        logger.info(f"Generated successful load response for adapter: {adapter_alias or adapter_name}")
        return response_content
