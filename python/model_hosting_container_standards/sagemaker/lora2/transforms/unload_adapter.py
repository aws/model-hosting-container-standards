from http import HTTPStatus
from typing import Any, Dict

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

from ....logging_config import logger
from ...lora.constants import ResponseMessage
from ...lora.utils import get_adapter_name_from_request_path
from ..lora_api_transform import (
    BaseLoRAApiTransform,
    BaseLoRATransformRequestOutput,
    LoRARequestBaseModel,
)

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
        logger.debug("Initialized UnloadLoraApiTransform")

    async def validate_request(
        self, raw_request: Request
    ) -> SageMakerUnloadAdapterRequest:
        logger.debug("Validating unload adapter request")

        try:
            adapter_name = get_adapter_name_from_request_path(raw_request)
            logger.debug(f"Extracted adapter name from request path: {adapter_name}")

            validated_request = SageMakerUnloadAdapterRequest.model_validate(
                {
                    "name": adapter_name,
                }
            )

            logger.info(
                f"Unload adapter request validated successfully for adapter: {adapter_name}"
            )
            return validated_request

        except (KeyError, ValidationError) as e:
            logger.error(f"Unload adapter request validation failed: {str(e)}")
            raise HTTPException(
                status_code=HTTPStatus.FAILED_DEPENDENCY.value,
                detail="Malformed request: Unable to extract adapter name from request path",
            )

    def _generate_successful_response_content(
        self,
        raw_response: Response,
        transform_request_output: BaseLoRATransformRequestOutput,
    ) -> str:
        adapter_alias = transform_request_output.additional_fields.get("adapter_alias")
        adapter_name = transform_request_output.additional_fields.get("adapter_name")

        response_content = ResponseMessage.ADAPTER_UNREGISTERED.format(
            alias=adapter_alias or adapter_name
        )

        logger.info(
            f"Generated successful unload response for adapter: {adapter_alias or adapter_name}"
        )
        return response_content
