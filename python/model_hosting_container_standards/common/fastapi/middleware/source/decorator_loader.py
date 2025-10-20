"""Decorator middleware loader."""

from typing import Callable, Union

from .....logging_config import logger
from .base import BaseMiddlewareLoader


class MiddlewareDecoratorLoader(BaseMiddlewareLoader):
    """Loads middleware from decorators and manages formatters."""

    def load(self) -> None:
        """Process loaded middlewares and handle combinations."""
        # Handle pre_fn + post_fn combination into pre_post_middleware
        self._handle_pre_post_combination()

    def set_middleware(self, name: str, middleware: Union[Callable, type]) -> None:
        """Set middleware directly from decorators.

        Args:
            name: Middleware name ("throttle", "pre_post_process")
            middleware: Middleware function or class
        """
        # Validate allowed middleware names (same as registry)
        allowed_names = ["throttle", "pre_post_process"]
        if name not in allowed_names:
            allowed_names_str = ", ".join(sorted(allowed_names))
            raise ValueError(
                f"Middleware name '{name}' is not allowed. Allowed names: {allowed_names_str}"
            )

        # Check for duplicate registration
        if name == "throttle" and self.throttle_middleware is not None:
            raise ValueError(
                f"Middleware '{name}' is already registered. Cannot register duplicate middleware."
            )
        elif name == "pre_post_process" and self.pre_post_middleware is not None:
            raise ValueError(
                f"Middleware '{name}' is already registered. Cannot register duplicate middleware."
            )

        # Set the middleware
        if name == "throttle":
            self.throttle_middleware = middleware
        elif name == "pre_post_process":
            self.pre_post_middleware = middleware

        logger.info(f"[MIDDLEWARE_DEC] Set {name} middleware: {middleware.__name__}")

    def set_input_formatter(self, formatter: Callable) -> None:
        """Set the input formatter function."""
        if self.pre_fn is not None:
            raise ValueError(
                "Input formatter is already registered. Only one input formatter is allowed."
            )
        self.pre_fn = formatter

    def set_output_formatter(self, formatter: Callable) -> None:
        """Set the output formatter function."""
        if self.post_fn is not None:
            raise ValueError(
                "Output formatter is already registered. Only one output formatter is allowed."
            )
        self.post_fn = formatter

    def _handle_pre_post_combination(self) -> None:
        """Handle combination of pre_fn + post_fn into pre_post_middleware if needed."""
        # Only combine if we don't already have a direct pre_post_middleware
        if not self.pre_post_middleware and (self.pre_fn or self.post_fn):
            self.pre_post_middleware = self._combine_pre_post_middleware(
                "MIDDLEWARE_DEC"
            )
            logger.info(
                "[MIDDLEWARE_DEC] Created combined pre_post_process middleware from input/output formatters"
            )

    def clear(self) -> None:
        """Clear all loaded middlewares and formatters."""
        self.throttle_middleware = None
        self.pre_post_middleware = None
        self.pre_fn = None
        self.post_fn = None


# Global decorator loader instance
decorator_loader = MiddlewareDecoratorLoader()
