"""Middleware registry for storing registered middlewares."""

from typing import Callable, Dict, List, Optional, Union

from ....logging_config import logger

# Allowed middleware names in execution order
ALLOWED_MIDDLEWARE_NAMES = ["throttle", "pre_post_process"]


class MiddlewareInfo:
    """Information about registered middleware."""

    def __init__(self, name: str, middleware: Union[Callable, type]):
        self.name = name
        self.middleware = middleware
        self.is_class = isinstance(middleware, type)


class MiddlewareRegistry:
    """Simple registry for storing middlewares."""

    def __init__(self) -> None:
        # Simple structure: name -> MiddlewareInfo
        self._middlewares: Dict[str, MiddlewareInfo] = {}
        # Only allow these specific middleware names
        self._allowed_middleware_names = set(ALLOWED_MIDDLEWARE_NAMES)

    def register_middleware(
        self,
        name: str,
        middleware: Union[Callable, type],
    ) -> None:
        """
        Register a middleware by name.

        Args:
            name: Middleware name (must be one of: throttle, pre_post_process)
            middleware: Middleware function or class

        Raises:
            ValueError: If name is not allowed or already registered
        """
        # Check if name is allowed
        if name not in self._allowed_middleware_names:
            allowed_names = ", ".join(sorted(self._allowed_middleware_names))
            raise ValueError(
                f"Middleware name '{name}' is not allowed. Allowed names: {allowed_names}"
            )

        # Check if already registered
        if name in self._middlewares:
            existing_type = "class" if self._middlewares[name].is_class else "function"
            new_type = "class" if isinstance(middleware, type) else "function"
            raise ValueError(
                f"Middleware '{name}' is already registered as a {existing_type}. Cannot register as {new_type}."
            )

        # Register the middleware
        self._middlewares[name] = MiddlewareInfo(name, middleware)

    def get_middleware(self, name: str) -> Optional[MiddlewareInfo]:
        """Get middleware info by name."""
        return self._middlewares.get(name)

    def has_middleware(self, name: str) -> bool:
        """Check if a middleware is registered by name."""
        return name in self._middlewares

    def list_middlewares(self) -> List[str]:
        """List all registered middleware names."""
        return list(self._middlewares.keys())

    def get_allowed_middleware_names(self) -> List[str]:
        """Get list of allowed middleware names."""
        return list(self._allowed_middleware_names)

    def clear_middlewares(self) -> None:
        """Clear all registered middlewares."""
        self._middlewares.clear()

    def load_middlewares(self, function_loader) -> None:
        """Load and resolve middlewares from all sources (env vars, decorators, formatters).

        Args:
            function_loader: Function loader for environment variable middleware.
        """
        # Import here to avoid circular imports
        from .source.decorator_loader import decorator_loader
        from .source.environment_loader import MiddlewareEnvironmentLoader

        env_loader = MiddlewareEnvironmentLoader()

        # Load from both sources
        logger.debug("[REGISTRY] Loading middlewares from environment variables")
        env_loader.load(function_loader)

        logger.debug("[REGISTRY] Loading middlewares from decorators")
        decorator_loader.load()

        # Register with priority (env overrides decorator)
        self._register_middleware_with_priority(
            "throttle", env_loader, decorator_loader
        )
        self._register_middleware_with_priority(
            "pre_post_process", env_loader, decorator_loader
        )

        logger.info("[REGISTRY] Middleware resolution and registration complete")

    def _register_middleware_with_priority(
        self, middleware_name: str, env_loader, dec_loader
    ) -> None:
        """Register middleware with env > decorator priority."""

        # Get middleware with priority: env > decorator
        middleware = env_loader.get_middleware(
            middleware_name
        ) or dec_loader.get_middleware(middleware_name)

        if middleware is None:
            logger.debug(
                f"[REGISTRY] No {middleware_name} middleware found in any source"
            )
            return

        # Register in registry
        try:
            self.register_middleware(middleware_name, middleware)
            source = (
                "env" if env_loader.get_middleware(middleware_name) else "decorator"
            )
            logger.info(
                f"[REGISTRY] Registered {middleware_name} middleware from {source}"
            )
        except ValueError as e:
            logger.error(f"[REGISTRY] Failed to register {middleware_name}: {e}")


# Global registry instance
middleware_registry = MiddlewareRegistry()
