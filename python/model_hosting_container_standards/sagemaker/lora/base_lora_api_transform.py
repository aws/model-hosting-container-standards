import abc
from http import HTTPStatus

from fastapi import Request, Response

from ...common import BaseApiTransform
from .models import BaseLoRATransformRequestOutput
from .utils import get_adapter_alias_from_request_header, get_adapter_name_from_request


class BaseLoRAApiTransform(BaseApiTransform):
    """Base abstract class for LoRA API request/response transformations.

    This class provides the foundation for transforming HTTP requests and responses
    using JMESPath expressions defined in request_shape and response_shape dictionaries.
    Subclasses must implement the abstract methods to handle specific transformation logic.
    """

    @abc.abstractmethod
    async def transform_request(
        self, raw_request: Request
    ) -> BaseLoRATransformRequestOutput:
        """Transform an incoming HTTP request for LoRA adapter operations.

        Subclasses must implement this method to handle request parsing, validation,
        and transformation according to their specific LoRA operation requirements.

        :param Request raw_request: The incoming FastAPI request object
        :return BaseLoRATransformRequestOutput: Transformed request data and metadata
        :raises NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    def transform_response(
        self,
        response: Response,
        transform_request_output: BaseLoRATransformRequestOutput,
    ):
        """Transform the response based on the request processing results.

        Routes to appropriate response transformation method based on HTTP status code.

        :param Response response: The response to transform
        :param transform_request_output: Output from the request transformation containing adapter info
        :return Response: Transformed response
        """
        adapter_name = get_adapter_name_from_request(transform_request_output)
        adapter_alias = get_adapter_alias_from_request_header(
            transform_request_output.raw_request
        )
        if response.status_code == HTTPStatus.OK:
            return self._transform_ok_response(
                response, adapter_name=adapter_name, adapter_alias=adapter_alias
            )
        else:
            return self._transform_error_response(
                response, adapter_name=adapter_name, adapter_alias=adapter_alias
            )

    def _transform_ok_response(self, response: Response, **kwargs):
        """Transform successful (200 OK) responses.

        :param Response response: The successful response to transform
        :param str adapter_name: Name of the LoRA adapter being processed
        :return Response: Transformed response
        :raises NotImplementedError: Must be implemented by subclasses if needed
        """
        raise NotImplementedError()

    def _transform_error_response(self, response: Response, **kwargs):
        """Transform error responses.

        :param Response response: The error response to transform
        :param str adapter_name: Name of the LoRA adapter being processed
        :return Response: Transformed response
        :raises NotImplementedError: Must be implemented by subclasses if needed
        """
        raise NotImplementedError()
