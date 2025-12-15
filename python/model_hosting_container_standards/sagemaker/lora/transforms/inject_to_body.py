import json
from http import HTTPStatus
from typing import Any, Dict, Optional

import jmespath
from fastapi import HTTPException, Request, Response
from pydantic import BaseModel

from ....common.fastapi.utils import serialize_request
from ....common.transforms.utils import set_value
from ....logging_config import logger
from ..base_lora_api_transform import BaseLoRAApiTransform
from ..models import AppendOperation, BaseLoRATransformRequestOutput


class InjectToBodyApiTransform(BaseLoRAApiTransform):
    """Transformer that injects adapter information to request body."""

    def __init__(
        self,
        request_shape: Dict[str, Any],
        response_shape: Dict[str, Any] = {},
        engine_request_model_cls: Optional[BaseModel] = None,
    ):
        # Validate and extract AppendOperation instances before passing to parent
        self._append_operations: Dict[str, AppendOperation] = {}
        cleaned_request_shape: Dict[str, str] = {}

        for key_path, value_config in request_shape.items():
            if isinstance(value_config, AppendOperation):
                # Store the append operation separately
                self._append_operations[key_path] = value_config
            elif isinstance(value_config, str):
                # Regular JMESPath string - will be compiled by parent
                cleaned_request_shape[key_path] = value_config
            else:
                raise ValueError(
                    f"Only strings and AppendOperation instances are allowed for {self.__class__}, but got {type(value_config)} for {key_path}"
                )

        if response_shape:
            logger.warning(
                f"{self.__class__} does not take response_shape, but {response_shape=}"
            )

        # Pass cleaned request_shape to parent for JMESPath compilation
        super().__init__(
            cleaned_request_shape,
            response_shape=response_shape,
            engine_request_model_cls=engine_request_model_cls,
        )

    async def transform_request(
        self, raw_request: Request
    ) -> BaseLoRATransformRequestOutput:
        try:
            request_data = await raw_request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"JSON decode error: {e}",
            ) from e
        source_data = serialize_request(request_data, raw_request)

        # Handle regular replace operations (compiled JMESPath in parent's _request_shape)
        for key_path, value_path in self._request_shape.items():
            value = value_path.search(source_data)
            request_data = set_value(request_data, key_path, value)

        # Handle append operations (stored separately with pre-compiled expressions)
        for key_path, append_op in self._append_operations.items():
            # Use the pre-compiled expression from AppendOperation
            adapter_id = append_op.compiled_expression.search(source_data)

            if adapter_id:
                # Get existing value at key_path
                existing_value = jmespath.search(key_path, request_data)
                if existing_value is None:
                    # If no existing value, just use the adapter_id
                    new_value = adapter_id
                    logger.debug(
                        f"No existing value at {key_path}, using adapter_id: {adapter_id}"
                    )
                else:
                    # Append with separator
                    new_value = f"{existing_value}{append_op.separator}{adapter_id}"
                    logger.debug(
                        f"Appending adapter_id to {key_path}: {existing_value} -> {new_value}"
                    )

                # Set the new value
                request_data = set_value(request_data, key_path, new_value)

        logger.debug(f"Updated request body: {request_data}")
        raw_request._body = json.dumps(request_data).encode("utf-8")
        transformed_request = (
            None
            if self._transformed_request_model is None
            else self._convert_to_model(request_data)
        )
        return BaseLoRATransformRequestOutput(
            request=transformed_request,
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
