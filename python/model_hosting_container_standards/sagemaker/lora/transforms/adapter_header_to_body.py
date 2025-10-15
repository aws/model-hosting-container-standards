import json
from typing import Any, Dict, Optional

from ....logging_config import logger
from ..base_lora_api_transform import BaseLoRAApiTransform
from ..models import BaseLoRATransformRequestOutput

from fastapi import Request, Response


class AdapterHeaderToBodyApiTransform(BaseLoRAApiTransform):
    """Transformer that moves adapter information from HTTP headers to request body.

    This transformer extracts adapter-related data from request headers and injects
    it into the request body, allowing downstream handlers to access adapter information
    directly from the request payload.
    """

    async def transform_request(self, raw_request: Request) -> BaseLoRATransformRequestOutput:
        """Transform request by adding header-derived data to the request body.

        Extracts data from request headers based on the configured request_shape mappings
        and adds this data to the request body. If keys already exist in the body,
        they will be overwritten.

        :param Request raw_request: The incoming request with headers to transform
        :return BaseLoRATransformRequestOutput: Transformed request with modified body
        """
        # Parse the original request body
        request_data = await raw_request.json()

        # Extract data from headers using JMESPath transformations
        add_to_body = self._transform_request(None, raw_request)
        # Merge header-derived data into request body
        logger.debug(f"Extracted headers: {add_to_body}")
        request_data.update(add_to_body)
        logger.debug(f"Updated request body with extracted headers: {request_data}")

        # Update the raw request body with the modified data
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
