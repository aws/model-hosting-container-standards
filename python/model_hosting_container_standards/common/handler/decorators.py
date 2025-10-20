"""Utility functions for creating handler decorators."""

from typing import Any, Callable

from ...logging_config import logger


def create_override_decorator(
    handler_type: str, handler_registry
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create a simple override decorator for handler functions.

    Args:
        handler_type: The type of handler to override (e.g., 'ping', 'invoke' for SageMaker).
        handler_registry: Registry instance that stores and manages handler functions.
                         Must implement a set_handler method.

    Returns:
        A decorator that immediately registers the decorated function as a customer
        override handler, replacing any default implementation.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Override the handler (decorator version)."""
        logger.debug(
            "[%s] @%s decorator called on function: %s",
            handler_type.upper(),
            handler_type,
            func.__name__,
        )
        handler_registry.set_handler(handler_type, func)
        logger.info(
            "[%s] Customer override registered: %s", handler_type.upper(), func.__name__
        )
        # Return the original function unchanged - it will be called directly
        return func

    return decorator
