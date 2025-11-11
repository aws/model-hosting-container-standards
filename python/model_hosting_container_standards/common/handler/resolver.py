"""Generic handler resolution logic for container standards.

This module provides a reusable template for resolving handlers with a consistent
priority order across different container types (SageMaker, etc.).

## Handler Resolution Priority Order

1. **Environment variable** specified function
2. **Registry** @decorated function
3. **Customer script** function
4. **Default** handler (if any)

## Usage Examples

```python
from model_hosting_container_standards.handler_resolver import GenericHandlerResolver

# Define handler-specific configuration
class MyHandlerConfig:
    def get_env_handler(self, handler_type: str):
        # Return handler from environment variable
        pass

    def get_customer_script_handler(self, handler_type: str):
        # Return handler from customer script
        pass

# Create resolver with configuration
resolver = GenericHandlerResolver(MyHandlerConfig())

# Resolve specific handlers
ping_handler = resolver.resolve_handler("ping")
invoke_handler = resolver.resolve_handler("invoke")
```
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

from model_hosting_container_standards.exceptions import (
    HandlerFileNotFoundError,
    HandlerNotFoundError,
    HandlerResolutionError,
    InvalidHandlerSpecError,
)

from .registry import handler_registry

logger = logging.getLogger(__name__)


class HandlerConfig(ABC):
    """Abstract base class for handler configuration.

    Subclasses must implement methods to load handlers from different sources.
    """

    @abstractmethod
    def get_env_handler(
        self, handler_type: str
    ) -> Union[Callable[..., Any], str, None]:
        """Get handler from environment variable.

        Args:
            handler_type: Type of handler (e.g., "ping", "invoke")

        Returns:
            Callable: Function handler
            str: Router path
            None: No handler found

        Raises:
            HandlerResolutionError: If env var is set but invalid
            InvalidHandlerSpecError: If handler spec is malformed
        """
        pass

    @abstractmethod
    def get_customer_script_handler(
        self, handler_type: str
    ) -> Optional[Callable[..., Any]]:
        """Get handler from customer script.

        Args:
            handler_type: Type of handler (e.g., "ping", "invoke")

        Returns:
            Callable function or None if not found
        """
        pass


class GenericHandlerResolver:
    """Generic handler resolver that implements consistent priority order.

    The resolver uses a clean separation of concerns with individual methods
    for each resolution step, making the code more maintainable and testable.
    """

    def __init__(self, config: HandlerConfig, registry=None) -> None:
        """Initialize the handler resolver.

        Args:
            config: Handler configuration for loading from different sources
            registry: Handler registry (defaults to global registry)
        """
        self.config = config
        self.registry = registry or handler_registry

    def _try_env_handler(self, handler_type: str) -> Optional[Callable]:
        """Try to resolve handler from environment variable.

        Args:
            handler_type: Type of handler to resolve

        Returns:
            Handler function if found, None otherwise

        Raises:
            HandlerResolutionError: If env var is set but invalid
            InvalidHandlerSpecError: If handler spec is malformed
        """
        try:
            env_handler = self.config.get_env_handler(handler_type)
            if callable(env_handler):
                # Function handler (already validated by config)
                handler_name = getattr(env_handler, "__name__", str(env_handler))
                logger.info(f"Found env {handler_type} handler: {handler_name}")
                return env_handler
            elif env_handler:
                # Router path - not a callable handler
                # redirection not implemented yet
                logger.debug(
                    f"Env {handler_type} handler is router path: {env_handler}"
                )
        except (HandlerResolutionError, InvalidHandlerSpecError):
            # If env var is set but invalid, this is a configuration error - don't continue
            logger.error(
                f"Environment variable {handler_type} handler configuration error"
            )
            raise

        logger.debug(f"No env {handler_type} handler found")
        return None

    def _try_decorator_handler(self, handler_type: str) -> Optional[Callable]:
        """Try to resolve handler from registry decorators.

        Args:
            handler_type: Type of handler to resolve

        Returns:
            Handler function if found, None otherwise
        """
        decorated_handler = self.registry.get_decorator_handler(handler_type)
        if decorated_handler:
            handler_name = getattr(
                decorated_handler, "__name__", str(decorated_handler)
            )
            logger.debug(f"Found decorator {handler_type} handler: {handler_name}")
            return decorated_handler

        logger.debug(f"No decorator {handler_type} handler found")
        return None

    def _try_customer_script_handler(self, handler_type: str) -> Optional[Callable]:
        """Try to resolve handler from customer script.

        Args:
            handler_type: Type of handler to resolve

        Returns:
            Handler function if found, None otherwise
        """
        try:
            customer_handler = self.config.get_customer_script_handler(handler_type)
            if customer_handler:
                handler_name = getattr(
                    customer_handler, "__name__", str(customer_handler)
                )
                logger.debug(f"Found customer module {handler_type}: {handler_name}")
                return customer_handler
        except (HandlerFileNotFoundError, HandlerNotFoundError) as e:
            # File doesn't exist or handler not found - continue to next priority
            logger.debug(
                f"No customer script {handler_type} function found: {type(e).__name__}"
            )
        except Exception:
            # File exists but has errors (syntax, import, etc.) - this is a real error
            logger.error(f"Customer script {handler_type} function failed to load")
            raise

        logger.debug(f"No customer module {handler_type} found")
        return None

    def resolve_handler(self, handler_type: str) -> Optional[Callable]:
        """
        Resolve handler with priority order:
        1. Environment variable specified function
        2. Registry @decorated function
        3. Customer script function
        4. Default handler registered by the framework

        Args:
            handler_type: Type of handler to resolve (e.g., "ping", "invoke")

        Returns:
            Resolved handler function or None if not found
        """
        logger.debug(f"resolve_{handler_type}_handler called")

        # Try each resolution method in priority order
        for resolver_method in [
            self._try_env_handler,
            self._try_decorator_handler,
            self._try_customer_script_handler,
        ]:
            handler = resolver_method(handler_type)
            if handler:
                return handler

        # No handler found anywhere, let us just do nothing
        # handler = self.registry.get_framework_default(handler_type)
        # if handler:
        #     logger.info(f"Use {handler_type} handler registered in framework")
        #     return handler

        logger.debug(f"No {handler_type} handler found anywhere")
        return None
