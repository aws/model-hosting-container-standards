import json
from typing import Any, Dict

from fastapi import Request, Response

from ....common.fastapi.utils import serialize_request
from ....common.utils import set_value
from ....logging_config import logger
from ..base_lora_api_transform import BaseLoRAApiTransform
from ..models import BaseLoRATransformRequestOutput


class InjectToBodyApiTransform(BaseLoRAApiTransform):
    """Transformer that injects adapter information to request body."""

    def __init__(
        self, request_shape: Dict[str, Any], response_shape: Dict[str, Any] = {}
    ):
        for _, v in request_shape.items():
            if not isinstance(v, str):
                raise ValueError(
                    f"Nested objects and non-str values are not allowed for {self.__class__}, but {request_shape=} "
                )
        if response_shape:
            logger.warning(
                f"{self.__class__} does not take response_shape, but {response_shape=}"
            )
        super().__init__(request_shape, response_shape)

    async def transform_request(
        self, raw_request: Request
    ) -> BaseLoRATransformRequestOutput:
        request_data = await raw_request.json()
        source_data = serialize_request(request_data, raw_request)

        for key_path, value_path in self._request_shape.items():
            value = value_path.search(source_data)
            request_data = set_value(request_data, key_path, value)

        logger.debug(f"Updated request body: {request_data}")
        raw_request._body = json.dumps(request_data).encode("utf-8")
        return BaseLoRATransformRequestOutput(
            request=None,
            raw_request=raw_request,
        )

    def transform_response(self, response: Response, transform_request_output):
        """Pass through the response without any transformations.

        This transformer only modifies requests by moving header data to the body.
        Responses are returned unchanged as a passthrough operation.

        :param Response response: The response to pass through
        :param transform_request_output: Request transformation output (unused)
        :return Response: Unmodified response
        """
        return response
