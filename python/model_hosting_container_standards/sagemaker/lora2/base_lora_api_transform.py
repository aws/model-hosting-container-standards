from typing import Any, Dict, Optional, TypedDict

from fastapi import Request
from pydantic import BaseModel
from typing_extensions import NotRequired

from ...common.transforms.base_api_transform2 import BaseApiTransform2, BaseTransformRequestOutput
from ...logging_config import logger
from ..lora.utils import get_adapter_alias_from_request_header


class LoRAAdditionalFields(TypedDict):
    adapter_name: NotRequired[str]
    adapter_alias: NotRequired[Optional[str]]


class LoRATransformRequestOutput(BaseTransformRequestOutput):
    # raw_request: Any - inherited
    # transformed_request: Optional[Dict] = None - inherited
    additional_fields: LoRAAdditionalFields = {}


class LoRARequestBaseModel(BaseModel):
    name: str


class BaseLoRAApiTransform(BaseApiTransform2):
    def _extract_additional_fields(
        self, validated_request: LoRARequestBaseModel, raw_request: Request
    ) -> LoRAAdditionalFields:
        adapter_name = validated_request.name
        adapter_alias = get_adapter_alias_from_request_header(raw_request)

        logger.debug(
            f"Extracted adapter fields - name: {adapter_name}, alias: {adapter_alias}"
        )

        return LoRAAdditionalFields(
            adapter_name=adapter_name,
            adapter_alias=adapter_alias,
        )
