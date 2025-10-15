from http import HTTPStatus
from typing import Optional

from ..base_lora_api_transform import BaseLoRAApiTransform
from ..models import BaseLoRATransformRequestOutput
from ..constants import ResponseMessage, RequestField
from ..utils import get_adapter_name_from_request_path, get_adapter_alias_from_request_header

from fastapi import Request, Response, HTTPException


def validate_sagemaker_unregister_request(raw_request: Request):
    adapter_name = get_adapter_name_from_request_path(raw_request)
    if not adapter_name:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Malformed request path; missing path parameter: {RequestField.ADAPTER_NAME}"
        )
    return adapter_name


class UnregisterLoRAApiTransform(BaseLoRAApiTransform):
    async def transform_request(self, raw_request: Request) -> BaseLoRATransformRequestOutput:
        """
        :param Optional[pydantic.BaseModel] request: Not used because the Unregister LoRA API does not take a request body.
        :param fastapi.Request raw_request:
        """
        adapter_name = validate_sagemaker_unregister_request(raw_request)
        transformed_request = self._transform_request(None, raw_request)
        return BaseLoRATransformRequestOutput(
            request=transformed_request,
            raw_request=raw_request,
            adapter_name=adapter_name,
        )

    def _transform_ok_response(self, response: Response, adapter_name: str, adapter_alias: Optional[str] = None):
        return Response(
            status_code=HTTPStatus.OK,
            content=ResponseMessage.ADAPTER_UNREGISTERED.format(
                alias=adapter_alias or adapter_name)
        )

    def _transform_error_response(self, response: Response, adapter_name: str, adapter_alias: Optional[str] = None):
        # TODO: add error handling
        return response
