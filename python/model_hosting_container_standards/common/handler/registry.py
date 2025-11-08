"""Handler registry for managing Custom handlers."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class HandlerInfo:
    """Container for handler function and its route configuration.

    Attributes:
        func: The handler function to be registered
        route_kwargs: Optional FastAPI route parameters (dependencies, responses, etc.)
    """

    func: Callable[..., Any]
    route_kwargs: Dict[str, Any] = field(default_factory=dict)


class HandlerRegistry:
    """General registry for managing handlers by name with proper priority."""

    def __init__(self) -> None:
        self._handlers: Dict[str, HandlerInfo] = {}
        self._framework_default_handlers: Dict[str, HandlerInfo] = {}
        self._decorator_handlers: Dict[str, HandlerInfo] = {}

    def set_handler(self, name: str, handler: HandlerInfo) -> None:
        """Set a handler by name.

        Args:
            name: The handler name/type
            handler: HandlerInfo object containing the handler function and route config
        """
        self._handlers[name] = handler

    def set_framework_default(
        self,
        handler_type: str,
        handler: Callable[..., Any],
        route_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a framework default handler with optional route configuration.

        Args:
            handler_type: The type of handler (e.g., 'ping', 'invoke')
            handler: The handler function to register as framework default
            route_kwargs: Optional FastAPI route parameters (dependencies, responses, etc.)
        """
        handler_info = HandlerInfo(func=handler, route_kwargs=route_kwargs or {})
        self._framework_default_handlers[handler_type] = handler_info

    def set_decorator_handler(
        self,
        handler_type: str,
        handler: Callable[..., Any],
        route_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a decorator handler with optional route configuration.

        Args:
            handler_type: The type of handler (e.g., 'ping', 'invoke')
            handler: The handler function to register as decorator handler
            route_kwargs: Optional FastAPI route parameters (dependencies, responses, etc.)
        """
        handler_info = HandlerInfo(func=handler, route_kwargs=route_kwargs or {})
        self._decorator_handlers[handler_type] = handler_info

    def get_handler(self, name: str) -> Optional[HandlerInfo]:
        """Get handler info (function + route_kwargs) by name."""
        return self._handlers.get(name)

    def get_framework_default(self, handler_type: str) -> Optional[HandlerInfo]:
        """Get framework default handler info.

        Args:
            handler_type: The type of handler (e.g., 'ping', 'invoke')

        Returns:
            The framework default HandlerInfo, or None if not found
        """
        return self._framework_default_handlers.get(handler_type)

    def get_decorator_handler(self, handler_type: str) -> Optional[HandlerInfo]:
        """Get decorator handler info.

        Args:
            handler_type: The type of handler (e.g., 'ping', 'invoke')

        Returns:
            The decorator HandlerInfo, or None if not found
        """
        return self._decorator_handlers.get(handler_type)

    def has_framework_default(self, handler_type: str) -> bool:
        """Check if a framework default handler is registered.

        Args:
            handler_type: The type of handler (e.g., 'ping', 'invoke')

        Returns:
            True if the framework default handler is registered, False otherwise
        """
        return handler_type in self._framework_default_handlers

    def has_decorator_handler(self, handler_type: str) -> bool:
        """Check if a decorator handler is registered.

        Args:
            handler_type: The type of handler (e.g., 'ping', 'invoke')

        Returns:
            True if the decorator handler is registered, False otherwise
        """
        return handler_type in self._decorator_handlers

    def has_handler(self, name: str) -> bool:
        """Check if a handler is registered by name."""
        return name in self._handlers

    def remove_handler(self, name: str) -> None:
        """Remove a handler by name."""
        self._handlers.pop(name, None)

    def remove_framework_default(self, handler_type: str) -> None:
        """Remove a framework default handler by type."""
        self._framework_default_handlers.pop(handler_type, None)

    def remove_decorator_handler(self, handler_type: str) -> None:
        """Remove a decorator handler by type."""
        self._decorator_handlers.pop(handler_type, None)

    def list_handlers(self) -> List[str]:
        """List all registered handler names."""
        return list(self._handlers.keys())

    def list_framework_defaults(self) -> List[str]:
        """List all registered framework default handler types."""
        return list(self._framework_default_handlers.keys())

    def list_decorator_handlers(self) -> List[str]:
        """List all registered decorator handler types."""
        return list(self._decorator_handlers.keys())

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._framework_default_handlers.clear()
        self._decorator_handlers.clear()


# Global registry instance
handler_registry = HandlerRegistry()
