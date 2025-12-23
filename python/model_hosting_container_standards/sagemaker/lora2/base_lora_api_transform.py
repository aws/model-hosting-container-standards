from typing import Any, Dict

from fastapi import Request
from pydantic import BaseModel

from ...common.transforms.base_api_transform2 import BaseApiTransform2
from ...logging_config import logger
from ..lora.utils import get_adapter_alias_from_request_header


class LoRARequestBaseModel(BaseModel):
    name: str


class BaseLoRAApiTransform(BaseApiTransform2):
    def _extract_additional_fields(
        self, validated_request: LoRARequestBaseModel, raw_request: Request
    ) -> Dict[str, Any]:
        adapter_name = validated_request.name
        adapter_alias = get_adapter_alias_from_request_header(raw_request)

        logger.debug(
            f"Extracted adapter fields - name: {adapter_name}, alias: {adapter_alias}"
        )

        return dict(
            adapter_name=adapter_name,
            adapter_alias=adapter_alias,
        )
