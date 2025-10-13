from http import HTTPStatus
from typing import Optional

from ..base.api_transformer import BaseLoRAApiTransformer, BaseLoRATransformRequestOutput
from ..constants import ResponseMessage

from fastapi import Request, Response
from pydantic import BaseModel


class UnregisterLoRAApiTransformer(BaseLoRAApiTransformer):
    def transform_request(self, request: Optional[BaseModel], raw_request: Request) -> BaseLoRATransformRequestOutput:
        """
        :param Optional[pydantic.BaseModel] request: Not used because the Unregister LoRA API does not take a request body.
        :param fastapi.Request raw_request:
        """
        transformed_request = self._transform_request(None, raw_request)
        return BaseLoRATransformRequestOutput(
            request=transformed_request,
            raw_request=raw_request,
        )

    def _transform_ok_response(self, response: Response, adapter_name: str):
        return Response(
            status_code=HTTPStatus.OK,
            content=ResponseMessage.ADAPTER_UNREGISTERED.format(
                alias=adapter_name)
        )

    def _transform_error_response(self, response: Response, adapter_name: str):
        # TODO: add error handling
        return response