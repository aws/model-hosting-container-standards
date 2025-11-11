"""Handler resolution logic for SageMaker container standards.

This module provides SageMaker-specific handler resolution using the generic
handler resolver framework.

## Handler Resolution Priority Order

### For Ping Handlers:
1. **Environment variable** specified function (CUSTOM_FASTAPI_PING_HANDLER)
2. **Registry** @custom_ping_handler decorated function
3. **Customer script** def custom_sagemaker_ping_handler function
4. **Default** handler (if any)

### For Invoke Handlers:
1. **Environment variable** specified function (CUSTOM_FASTAPI_INVOCATION_HANDLER)
2. **Registry** @custom_invocation_handler decorated function
3. **Customer script** def custom_sagemaker_invocation_handler function
4. **Default** handler (if any)

## Error Handling

- Environment variable errors are raised immediately (configuration errors)
- Customer script errors are raised if script exists but can't be loaded
- Missing handlers return None (graceful degradation)
"""

import logging
from typing import Any, Callable, Optional, Union

from ..common.handler.registry import handler_registry
from ..common.handler.resolver import GenericHandlerResolver, HandlerConfig
from ..exceptions import HandlerFileNotFoundError, HandlerNotFoundError
from .sagemaker_loader import SageMakerFunctionLoader

logger = logging.getLogger(__name__)


class SageMakerHandlerConfig(HandlerConfig):
    """SageMaker-specific handler configuration."""

    # Mapping from handler type to custom function name
    HANDLER_TYPE_TO_FUNCTION_NAME = {
        "invoke": "custom_sagemaker_invocation_handler",
        "ping": "custom_sagemaker_ping_handler",
    }

    def get_env_handler(
        self, handler_type: str
    ) -> Union[Callable[..., Any], str, None]:
        """Get handler from SageMaker environment variables."""
        if handler_type == "ping":
            return SageMakerFunctionLoader.get_ping_handler_from_env()
        elif handler_type == "invoke":
            return SageMakerFunctionLoader.get_invocation_handler_from_env()
        else:
            return None

    def get_customer_script_handler(
        self, handler_type: str
    ) -> Optional[Callable[..., Any]]:
        """Get handler from SageMaker customer script."""
        custom_function_name = self.HANDLER_TYPE_TO_FUNCTION_NAME.get(handler_type)
        if not custom_function_name:
            logger.debug(f"No mapping found for handler type: {handler_type}")
            return None

        try:
            return SageMakerFunctionLoader.load_function_from_spec(
                f"model:{custom_function_name}"
            )
        except (HandlerFileNotFoundError, HandlerNotFoundError) as e:
            # Function not found - continue to next priority
            logger.debug(
                f"No customer script {custom_function_name} function found: {type(e).__name__}"
            )
            return None
        except Exception:
            # File exists but has errors (syntax, import, etc.) - this is a real error
            logger.error(
                f"Customer script {custom_function_name} function failed to load"
            )
            raise


class SageMakerHandlerResolver(GenericHandlerResolver):
    """SageMaker-specific handler resolver inheriting from generic resolution logic."""

    def __init__(self) -> None:
        """Initialize the SageMaker handler resolver."""
        super().__init__(SageMakerHandlerConfig())


# Global resolver instance
_resolver = SageMakerHandlerResolver()


def register_sagemaker_overrides():
    def set_handler(handler_type):
        handler = _resolver.resolve_handler(handler_type)
        if handler:
            handler_registry.set_handler(handler_type, handler)

    set_handler("invoke")
    set_handler("ping")
