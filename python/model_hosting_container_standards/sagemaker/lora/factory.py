import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict

from fastapi import Request

from ...common.handler import handler_registry
from .transforms import get_transform_cls_from_handler_type

logger = logging.getLogger(__name__)


def _resolve_transforms(
    handler_type: str, request_shape: Dict[str, Any], response_shape: Dict[str, Any]
):
    """Resolve and instantiate the appropriate transformer class for the given handler type.

    :param str handler_type: The LoRA handler type (e.g., 'register_adapter', 'unregister_adapter')
    :param Dict[str, Any] request_shape: JMESPath expressions for request transformation
    :param Dict[str, Any] response_shape: JMESPath expressions for response transformation
    :return: Instantiated transformer class for the specified handler type
    :raises ValueError: If handler_type is not supported
    """
    logger.debug(f"Resolving transformer for handler type: {handler_type}")
    # TODO: figure out how to validate that request shape's path specifications for sagemaker are valid
    # TODO: figure out how to validate that response shape's keys for sagemaker are valid
    _transformer_cls = get_transform_cls_from_handler_type(handler_type)
    logger.debug(
        f"Creating transformer instance: {getattr(_transformer_cls, '__name__', str(_transformer_cls))}"
    )
    return _transformer_cls(request_shape, response_shape)


def create_transform_decorator(handler_type: str):
    """Create a decorator factory for LoRA API transform handlers.

    This function creates decorators that automatically apply request/response transformations
    to handler functions based on JMESPath expressions. The decorated function will have
    request data transformed according to the request_shape and responses transformed
    according to the response_shape.

    :param str handler_type: The type of LoRA handler (e.g., 'register_adapter', 'unregister_adapter')
    :return: Decorator factory function that accepts request_shape and response_shape parameters
    """

    def decorator_with_params(
        request_shape: Dict[str, Any] = {}, response_shape: Dict[str, Any] = {}
    ):
        """Configure the transformation shapes for the decorator.

        :param Dict[str, Any] request_shape: JMESPath expressions defining request data extraction
        :param Dict[str, Any] response_shape: JMESPath expressions defining response transformation
        :return: Actual decorator function that wraps the handler
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """Decorator that wraps a handler function with transformation logic.

            :param Callable[..., Any] func: The handler function to wrap
            :return Callable[..., Any]: Wrapped function with transformation applied
            """
            # if no transform shapes specified, register as passthrough handler
            if not request_shape and not response_shape:
                logger.info("No transform shapes defined, using passthrough")
                handler_registry.set_handler(f"{handler_type}", func)
                logger.info(
                    f"[{handler_type.upper()}] Registered transform handler for {func.__name__}"
                )
                return func

            # Resolve transforms as needed
            transformer = _resolve_transforms(
                handler_type, request_shape, response_shape
            )

            # Create wrapped function that applies transforms
            logger.info(
                f"[{handler_type.upper()}] Transform decorator applied to: {func.__name__}"
            )

            async def decorated_func(raw_request: Request):
                """The actual wrapped handler function that applies transformations."""
                logger.debug(f"Applying request transformation for {handler_type}")
                # Apply request transformations using the configured transformer
                transform_request_output = await transformer.transform_request(
                    raw_request
                )
                transformed_request = transform_request_output.request
                transformed_raw_request = transform_request_output.raw_request
                logger.debug(f"Request transformation complete for {handler_type}")

                if not transformed_request:
                    logger.debug(
                        "No transformed request data, passing raw request only"
                    )
                    # If transformed_request is None, only pass the modified raw request
                    response = await func(transformed_raw_request)
                else:
                    logger.debug(
                        "Passing transformed request data and raw request to handler"
                    )
                    # Pass both transformed data and original request for context
                    # Convert dict to SimpleNamespace for attribute access
                    response = await func(
                        SimpleNamespace(**transformed_request), transformed_raw_request
                    )

                logger.debug(f"Applying response transformation for {handler_type}")
                # Apply response transformations and return final response
                final_response = transformer.transform_response(
                    response, transform_request_output
                )
                logger.debug(f"Response transformation complete for {handler_type}")
                return final_response

            # Register the wrapped function in the handler registry
            handler_registry.set_handler(handler_type, decorated_func)
            logger.info(
                f"[{handler_type.upper()}] Registered transform handler for {func.__name__}"
            )

            return decorated_func

        return decorator

    return decorator_with_params
