"""SageMaker integration decorators."""

from typing import Any, Callable, List

from ..utils import create_override_decorator, create_register_decorator


# Placeholder registry - will be replaced with actual registry in next CR
class _PlaceholderRegistry:
    """Placeholder registry for demonstration."""

    def __init__(self):
        self.handlers = {}

    def set_handler(self, handler_type: str, handler_func):
        """Placeholder set_handler method."""
        pass  # No-op for now


_handler_registry = _PlaceholderRegistry()


# Placeholder resolver functions - will be replaced with actual resolvers in next CR
def _get_ping_handler():
    """Placeholder ping handler resolver."""
    return None  # No existing handler found


def _get_invocation_handler():
    """Placeholder invocation handler resolver."""
    return None  # No existing handler found


# SageMaker decorator instances - created using utility functions

# Override decorators - immediately register customer handlers
ping = create_override_decorator("ping", _handler_registry)
invoke = create_override_decorator("invoke", _handler_registry)

# Register decorators - created using create_register_decorator
register_ping_handler = create_register_decorator(
    "ping", _get_ping_handler, _handler_registry
)
register_invocation_handler = create_register_decorator(
    "invocation", _get_invocation_handler, _handler_registry
)


__all__: List[str] = [
    "ping",
    "invoke",
    "register_ping_handler",
    "register_invocation_handler",
]