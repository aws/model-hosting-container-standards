import json
from typing import Any, Dict, Optional

from ....logging_config import logger
from ..base.api_transformer import BaseLoRAApiTransformer, BaseLoRATransformRequestOutput

from fastapi import Request, Response


class AdapterHeaderToBodyApiTransformer(BaseLoRAApiTransformer):
    def transform_request(self, request: Dict[str, Any], raw_request: Request) -> BaseLoRATransformRequestOutput:
        """
        :param Dict[str, Any] request:
        :param fastapi.Request raw_request:
        """
        add_to_body = self._transform_request(None, raw_request)
        overwritten_keys = [k for k in add_to_body.keys() if k in request.keys()]
        if overwritten_keys:
            logger.warning(f"Overwriting the following field(s) in the request: {overwritten_keys}")

        request.update(add_to_body)
        raw_request._body = json.dumps(request).encode("utf-8")
        return BaseLoRATransformRequestOutput(
            request=None,
            raw_request=request,
        )


    def transform_response(self, response: Response):
        """
        Passthrough.
        """
        return response