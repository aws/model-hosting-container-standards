from http import HTTPStatus

from ..base.api_transformer import BaseLoRAApiTransformer
from ..models.transformer import BaseLoRATransformRequestOutput
from ..constants import ResponseMessage
from ..models.request import SageMakerRegisterLoRAAdapterRequest

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ValidationError


def validate_sagemaker_register_request(request_data: dict) -> SageMakerRegisterLoRAAdapterRequest:
    if not request_data.get("name"):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="The parameter name is required")
    if not request_data.get("src"):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="The parameter src is required")
    try:
        sagemaker_request: SageMakerRegisterLoRAAdapterRequest = \
        SageMakerRegisterLoRAAdapterRequest.model_validate(request_data)
        return sagemaker_request
    except ValidationError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="The request body is invalid.")


class RegisterLoRAApiTransformer(BaseLoRAApiTransformer):
    def transform_request(self, request: SageMakerRegisterLoRAAdapterRequest, raw_request: Request) -> BaseLoRATransformRequestOutput:
        transformed_request = self._transform_request(request, raw_request)
        return BaseLoRATransformRequestOutput(
            request=transformed_request,
            raw_request=raw_request,
            adapter_name=request.name,
        )

    def _transform_ok_response(self, response: Response, adapter_name: str):
        return Response(
            status_code=HTTPStatus.OK,
            content=ResponseMessage.ADAPTER_REGISTERED.format(
                alias=adapter_name)
        )

    def _transform_error_response(self, response: Response, adapter_name: str):
        # TODO: add error handling
        return response