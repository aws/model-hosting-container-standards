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
    resolver_func: Callable[[], Optional[Callable[..., Any]]],
    handler_registry,
) -> Callable[[Optional[Callable[..., Any]]], Callable[..., Any]]:
    """Create a register decorator that resolves handler precedence at startup.

    Args:
        handler_type: The type of handler to register (e.g., 'ping', 'invocation' for SageMaker).
        resolver_func: Function that checks for existing customer handlers from
                      environment variables or customer scripts. Returns the handler
                      if found, None otherwise.
        handler_registry: Registry instance that stores the final resolved handler.
                         Must implement a set_handler method.

    Returns:
        A decorator that either uses an existing customer handler (if found by
        resolver_func) or registers the decorated function as the default handler.
        Customer handlers always take precedence over defaults.
    """

    def register_decorator(
        func: Optional[Callable[..., Any]] = None,
    ) -> Callable[..., Any]:
        """Register an async handler function, resolved at startup time."""

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            logger.debug(
                "[DECORATOR] register_%s_handler called on function: %s",
                handler_type,
                f.__name__,
            )

            # Resolve the final handler at decoration time (startup)
            final_handler = resolver_func()
            logger.debug(
                "[DECORATOR] Resolved final handler: %s",
                final_handler.__name__ if final_handler else "None",
            )

            if final_handler:
                # Customer script or env var handler takes precedence
                logger.info(
                    "[DECORATOR] Using customer handler: %s", final_handler.__name__
                )
                return final_handler
            else:
                # No existing handler found, register and use the decorated function
                handler_registry.set_handler(handler_type, f)
                logger.debug("Using default handler: %s", f.__name__)
                return f

        if func is None:
            return decorator
        else:
            return decorator(func)

    return register_decorator
