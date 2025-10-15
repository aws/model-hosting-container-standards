from http import HTTPStatus

from fastapi import Request, Response
from fastapi.exceptions import HTTPException
from pydantic import ValidationError

from ..base_lora_api_transform import BaseLoRAApiTransform
from ..constants import ResponseMessage
from ..models import BaseLoRATransformRequestOutput, SageMakerRegisterLoRAAdapterRequest


def validate_sagemaker_register_request(
    request_data: dict,
) -> SageMakerRegisterLoRAAdapterRequest:
    """Validate and parse a SageMaker register LoRA adapter request.

    :param dict request_data: Raw request data to validate
    :return SageMakerRegisterLoRAAdapterRequest: Validated request model
    :raises HTTPException: If required parameters are missing or validation fails
    """
    try:
        sagemaker_request: SageMakerRegisterLoRAAdapterRequest = (
            SageMakerRegisterLoRAAdapterRequest.model_validate(request_data)
        )
        return sagemaker_request
    except ValidationError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=e.errors(include_url=False)
        )


class RegisterLoRAApiTransform(BaseLoRAApiTransform):
    """Transformer for LoRA adapter registration API requests and responses.

    Handles the transformation of register adapter requests, including validation
    of required parameters (name, src) and generation of appropriate success/error responses.
    """

    async def transform_request(
        self, raw_request: Request
    ) -> BaseLoRATransformRequestOutput:
        """Transform and validate a register LoRA adapter request.

        :param Request raw_request: The incoming request containing adapter registration data
        :return BaseLoRATransformRequestOutput: Validated and transformed request with adapter name
        :raises HTTPException: If request validation fails
        """
        request_data = await raw_request.json()
        request = validate_sagemaker_register_request(request_data)
        transformed_request = self._transform_request(request, raw_request)
        return BaseLoRATransformRequestOutput(
            request=transformed_request,
            raw_request=raw_request,
            adapter_name=request.name,
        )

    def _transform_ok_response(self, response: Response, **kwargs):
        """Transform successful registration response with adapter confirmation message.

        :param Response response: The original successful response
        :param str adapter_name: Name of the successfully registered adapter
        :return Response: Response with adapter registration confirmation message
        """
        adapter_name = kwargs.get("adapter_name")
        adapter_alias = kwargs.get("adapter_alias")
        return Response(
            status_code=HTTPStatus.OK,
            content=ResponseMessage.ADAPTER_REGISTERED.format(
                alias=adapter_alias or adapter_name
            ),
        )

    def _transform_error_response(self, response: Response, **kwargs):
        """Transform error response for failed registration attempts.

        :param Response response: The original error response
        :param str adapter_name: Name of the adapter that failed to register
        :return Response: Transformed error response
        """
        # TODO: add error handling
        response_body = response.body.decode()
        # Register adapter failed (adapter weights related error): 424.
        if ResponseMessage.ADAPTER_INVALID_WEIGHTS in response_body:
            return Response(
                status_code=HTTPStatus.FAILED_DEPENDENCY,
                content=response_body,
            )
        if ResponseMessage.ADAPTER_MAX_LORA_RANK in response_body:
            return Response(
                status_code=HTTPStatus.FAILED_DEPENDENCY,
                content=response_body,
            )
        return response
