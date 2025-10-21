"""Utility functions for creating handler decorators."""

from typing import Any, Callable, Optional

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


def create_register_decorator(
    handler_type: str,
    handler_registry,
) -> Callable[[Optional[Callable[..., Any]]], Callable[..., Any]]:
    """Create a register decorator that automatically sets up routes for handlers.

    This decorator registers a function as a framework handler and automatically
    sets up the appropriate routes when used with SageMaker handlers.

    Args:
        handler_type: The type of handler to register (e.g., 'ping', 'invoke' for SageMaker).
        handler_registry: Registry instance that stores the final resolved handler.
                        Must implement a set_handler method.

    Returns:
        A decorator that registers the function and sets up routes automatically.
    """

    def register_decorator(
        func: Optional[Callable[..., Any]] = None,
    ) -> Callable[..., Any]:
        if func is None:
            return register_decorator  # type: ignore

        logger.debug(
            "[%s] @register_%s_handler decorator called on function: %s",
            handler_type.upper(),
            handler_type,
            func.__name__,
        )

        # Register the handler in the registry with framework prefix to maintain priority order
        handler_registry.set_handler("framework_" + handler_type, func)

        logger.info(
            "[%s] Framework handler registered: %s", handler_type.upper(), func.__name__
        )

        return func

    return register_decorator
